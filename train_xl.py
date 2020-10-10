from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time
import random

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf
import tensorflow_addons as tfa
import data_utils
import transformer_xl as txl

import numpy as np

# GPU config
flags.DEFINE_integer("num_hosts", default=1,
      help="Number of TPU hosts")
flags.DEFINE_integer("num_core_per_host", default=8,
      help="Number of cores per host")
flags.DEFINE_bool("use_tpu", default=False,
      help="Use TPUs rather than plain CPUs.")

# Experiment (data/checkpoint/directory) config
flags.DEFINE_string("data_dir", default="",
      help="Path to tf-records directory.")
flags.DEFINE_string("record_info_dir", default="",
      help="Path to local directory containing filenames.txt.")
flags.DEFINE_string("corpus_info_path", default="",
      help="Path to corpus-info.json file.")
flags.DEFINE_string("checkpoint_dir", default=None,
      help="directory for saving checkpoint.")
flags.DEFINE_bool("warm_start", default=False,
      help="Whether to warm start training from checkpoint.")
flags.DEFINE_bool("do_train", default=True,
      help="Whether to run training.")
flags.DEFINE_bool("do_eval", default=False,
      help="Whether to run eval on the dev set and test set during training.")

# Optimization config
flags.DEFINE_float("learning_rate", default=2.5e-4,
      help="Maximum learning rate.")
flags.DEFINE_float("clip", default=0.25,
      help="Gradient clipping value.")
# for cosine decay
flags.DEFINE_float("min_lr_ratio", default=0.004,
      help="Minimum ratio learning rate.")
flags.DEFINE_integer("warmup_steps", default=0,
      help="Number of steps for linear lr warmup.")

# Training config
flags.DEFINE_integer("train_batch_size", default=60,
      help="Size of train batch.")
flags.DEFINE_integer("eval_batch_size", default=60,
      help="Size of valid batch.")
flags.DEFINE_integer("test_batch_size", default=60,
      help="Size of test batch.")
flags.DEFINE_integer("train_steps", default=100000,
      help="Total number of training steps.")
flags.DEFINE_integer("iterations", default=500,
      help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps", default=10000,
      help="number of steps for model checkpointing.")

# Evaluation config
flags.DEFINE_bool("do_test", default=False,
      help="Run on the test set.")
flags.DEFINE_integer("max_eval_batch", default=-1,
      help="Set -1 to turn off.")
flags.DEFINE_string("eval_split", "valid",
      help="Which data split to evaluate.")

# Model config
flags.DEFINE_integer("tgt_len", default=70,
      help="Number of steps to predict")
flags.DEFINE_integer("mem_len", default=70,
      help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
      help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length")
flags.DEFINE_integer("test_tgt_len", default=70,
      help="Number of steps to predict for test set")
flags.DEFINE_integer("test_mem_len", default=70,
      help="Number of steps to cache for test set")

flags.DEFINE_integer("n_layer", default=6,
      help="Number of layers.")
flags.DEFINE_integer("d_model", default=500,
      help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=500,
      help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=10,
      help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=50,
      help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=1000,
      help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.1,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
      help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False,
      help="untie r_w_bias and r_r_bias")
flags.DEFINE_bool("pre_lnorm", default=False,
      help="Use pre layer normalization in the transformer blocks")

# Adaptive Softmax / Embedding
flags.DEFINE_integer("div_val", default=1,
      help="Divide the embedding size by this val for each bin")
flags.DEFINE_bool("proj_share_all_but_first", default=False,
      help="True to share all but first projs, False not to share.")
flags.DEFINE_bool("proj_same_dim", default=True,
      help="Project the bin with the same dimension.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
      enum_values=["normal", "uniform"],
      help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
      help="Initialization std when init is normal.")
flags.DEFINE_float("proj_init_std", default=0.01,
      help="Initialization std for embedding projection.")
flags.DEFINE_float("init_range", default=0.1,
      help="Initialization std when init is uniform.")

FLAGS = flags.FLAGS


class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, tr_steps, wu_steps=0, min_lr_ratio=0.0):
        self.max_lr=max_lr
        self.tr_steps=tr_steps
        self.wu_steps=wu_steps
        self.min_lr_ratio=min_lr_ratio
    def __call__(self, step):
        step_float = tf.cast(step, tf.float32)
        wu_steps_float = tf.cast(self.wu_steps, tf.float32)
        tr_steps_float = tf.cast(self.tr_steps, tf.float32)
        max_lr_float =tf.cast(self.max_lr, tf.float32)
        min_lr_ratio_float = tf.cast(self.min_lr_ratio, tf.float32)

        # warmup learning rate using linear schedule
        wu_lr = (step_float/wu_steps_float) * max_lr_float

        # decay the learning rate using the cosine schedule
        global_step = tf.math.minimum(step_float-wu_steps_float, tr_steps_float-wu_steps_float)
        decay_steps = tr_steps_float-wu_steps_float
        pi = tf.constant(math.pi)
        cosine_decay = .5 * (1. + tf.math.cos(pi * global_step / decay_steps))
        decayed = (1.-min_lr_ratio_float) * cosine_decay + min_lr_ratio_float
        decay_lr = max_lr_float * decayed
        return tf.cond(step < self.wu_steps, lambda: wu_lr, lambda: decay_lr)


def create_model(n_token, cutoffs):
    if FLAGS.init == "uniform":
      initializer = tf.compat.v1.initializers.random_uniform(
          minval=-FLAGS.init_range,
          maxval=FLAGS.init_range,
          seed=None)
      proj_initializer = None
    elif FLAGS.init == "normal":
      initializer = tf.compat.v1.initializers.random_normal(
          stddev=FLAGS.init_std,
          seed=None)
      proj_initializer = tf.compat.v1.initializers.random_normal(
          stddev=FLAGS.proj_init_std,
          seed=None)

    tie_projs = [False for _ in range(len(cutoffs) + 1)]
    if FLAGS.proj_share_all_but_first:
      for i in range(1, len(tie_projs)):
        tie_projs[i] = True

    model = txl.TransformerXL(
        n_token=n_token,
        n_layer = FLAGS.n_layer,
        d_model = FLAGS.d_model,
        d_embed = FLAGS.d_embed,
        n_head = FLAGS.n_head,
        d_head = FLAGS.d_head,
        d_inner = FLAGS.d_inner,
        dropout = FLAGS.dropout,
        dropatt = FLAGS.dropatt,
        initializer = initializer,
        proj_initializer = proj_initializer,
        pre_lnorm = FLAGS.pre_lnorm,
        tgt_len = FLAGS.tgt_len,
        mem_len = FLAGS.mem_len,
        cutoffs = cutoffs,
        div_val = FLAGS.div_val,
        tie_projs = tie_projs,
        same_length = FLAGS.same_length,
        clamp_len = FLAGS.clamp_len,
        untie_r = FLAGS.untie_r,
        proj_same_dim = FLAGS.proj_same_dim,
        use_tpu = FLAGS.use_tpu
    )

    return model


def create_dist_mems(strategy, n_layer, d_model, mem_len, bsz):
    # mem_len == 0 does not work properly on TPU,
    # assign it to a positive value.
    num_replica = strategy.num_replicas_in_sync
    mems = tf.zeros((num_replica, n_layer, mem_len, bsz, d_model))
    mems_dataset = tf.data.Dataset.from_tensor_slices(mems).batch(num_replica)
    dist_mems_dataset = strategy.experimental_distribute_dataset(mems_dataset)
    return next(iter(dist_mems_dataset))


def train(n_token, cutoffs, train_data, valid_data, test_data, strategy, chk_name):
  train_input_fn = train_data.get_dataset
  train_record_info = train_data.get_record_info()
  num_train_batch = train_record_info["num_batch"]

  tf.compat.v1.logging.info("num of train batches {}".format(train_record_info["num_batch"]))

  if FLAGS.do_eval:
    valid_input_fn = valid_data.get_dataset
    valid_record_info = valid_data.get_record_info()
    num_valid_batch = valid_record_info["num_batch"]

    tf.compat.v1.logging.info("num of valid batches {}".format(valid_record_info["num_batch"]))

    test_input_fn = test_data.get_dataset
    test_record_info = test_data.get_record_info()
    num_test_batch = test_record_info["num_batch"]

    tf.compat.v1.logging.info("num of test batches {}".format(test_record_info["num_batch"]))

  # Ensure that number of replicas in sync is same as 'FLAGS.num_core_per_host'
  assert(FLAGS.num_core_per_host == strategy.num_replicas_in_sync)

  ##### Create computational graph for train dataset
  train_dataset = train_input_fn()
  train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

  if FLAGS.do_eval:
    ##### Create computational graph for valid dataset
    valid_dataset = valid_input_fn()
    valid_dist_dataset = strategy.experimental_distribute_dataset(valid_dataset)

    ##### Create computational graph for test dataset
    test_dataset = test_input_fn()
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

  if FLAGS.save_steps == 0:
    FLAGS.save_steps = None
  else:
    # Set the FLAGS.save_steps to a value multiple of FLAGS.iterations
    if FLAGS.save_steps < FLAGS.iterations:
        FLAGS.save_steps = FLAGS.iterations
    else:
        FLAGS.save_steps = (FLAGS.save_steps // FLAGS.iterations) * \
                                                          FLAGS.iterations
  ##### Instantiate learning rate scheduler object
  lr_sch = LRSchedule(
          FLAGS.learning_rate, FLAGS.train_steps, \
          FLAGS.warmup_steps, FLAGS.min_lr_ratio
  )

  ##### Create computational graph for model
  with strategy.scope():
    model = create_model(n_token, cutoffs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sch)
    #optimizer = tfa.optimizers.LAMB(learning_rate=lr_sch)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    eval_loss = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)
    grad_norm = tf.keras.metrics.Mean('grad_norms', dtype=tf.float32)

    if FLAGS.warm_start:
      options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
      chk_path = os.path.join(FLAGS.checkpoint_dir, chk_name)
      try:
        checkpoint.read(chk_path, options=options)
        tf.compat.v1.logging.info("Restored checkpoint: {}".format(chk_path))
      except:
        tf.compat.v1.logging.info("Could not restore checkpoint, starting training from beginning")

  @tf.function
  def train_steps(iterator, steps, dist_mems, bsz):
    ###### Reset the states of the update variables
    train_loss.reset_states()
    grad_norm.reset_states()
    ###### The step function for one training step
    def step_fn(inps, lbls, mems):
      mems = tf.squeeze(mems, axis=0)
      with tf.GradientTape() as tape:
        loss, new_mems = model(inps, lbls, mems, training=True)
        per_example_loss = tf.reduce_mean(loss, axis=1)
        avg_loss = tf.nn.compute_average_loss(per_example_loss, \
                                            global_batch_size=bsz)
      variables = tape.watched_variables()
      gradients = tape.gradient(avg_loss, variables)
      clipped, gnorm = tf.clip_by_global_norm(gradients, FLAGS.clip)
      optimizer.apply_gradients(list(zip(clipped, variables)))
      train_loss.update_state(avg_loss * strategy.num_replicas_in_sync)
      grad_norm.update_state(gnorm)
      return tf.expand_dims(new_mems, axis=0)
    for _ in range(steps):
      inps, lbls, _ = next(iterator)
      dist_mems = strategy.run(step_fn, args=(inps, lbls, dist_mems,))
    return dist_mems

  @tf.function
  def eval_steps(iterator, steps, dist_mems, bsz):
    ###### The step function for one evaluation step
    def step_fn(inps, lbls, mems):
      mems = tf.squeeze(mems, axis=0)
      loss, new_mems = model(inps, lbls, mems, training=False)
      per_example_loss = tf.reduce_mean(loss, axis=1)
      avg_loss = tf.nn.compute_average_loss(per_example_loss, \
                                            global_batch_size=bsz)
      eval_loss.update_state(avg_loss * strategy.num_replicas_in_sync)
      return tf.expand_dims(new_mems, axis=0)
    for _ in range(steps):
      inps, lbls, _ = next(iterator)
      dist_mems = strategy.run(step_fn, args=(inps, lbls, dist_mems,))
    return dist_mems

  per_replica_bsz = FLAGS.train_batch_size // FLAGS.num_hosts // strategy.num_replicas_in_sync
  dist_mems = create_dist_mems(
          strategy, FLAGS.n_layer, FLAGS.d_model, 
          FLAGS.mem_len, per_replica_bsz
  )
  tf.compat.v1.logging.info('Starting training ... ')
  train_iter = iter(train_dist_dataset)

  cur_step = optimizer.iterations.numpy()
  while cur_step < FLAGS.train_steps:
    dist_mems = train_steps(train_iter, tf.convert_to_tensor(FLAGS.iterations), \
                            dist_mems, FLAGS.train_batch_size)

    cur_step = optimizer.iterations.numpy()
    cur_loss = train_loss.result()
    gnorm = grad_norm.result()
    lr_rate = lr_sch(cur_step)

    tf.compat.v1.logging.info("[{:6d}] | gnorm {:5.2f} lr {:9.6f} "
            "| loss {:>5.2f} | pplx {:>7.2f}, bpc {:>8.4f}".format(
            cur_step, gnorm, lr_rate, cur_loss, math.exp(cur_loss), 
            cur_loss / math.log(2)))

    if FLAGS.do_eval:
      eval_tr_bsz = FLAGS.train_batch_size // FLAGS.num_hosts // strategy.num_replicas_in_sync
      eval_tr_dist_mems = create_dist_mems(
              strategy, FLAGS.n_layer, FLAGS.d_model, 
              FLAGS.mem_len, eval_tr_bsz
      )

      if FLAGS.max_eval_batch <= 0:
        num_eval_iters = min(num_valid_batch, num_test_batch)
      else: 
        num_eval_iters = min(min(FLAGS.max_eval_batch, num_valid_batch), num_test_batch)

      eval_tr_iter = iter(train_dist_dataset)
      eval_loss.reset_states()
      eval_steps(eval_tr_iter, tf.convert_to_tensor(num_eval_iters), \
                 eval_tr_dist_mems, FLAGS.train_batch_size)

      cur_eval_loss = eval_loss.result()
      tf.compat.v1.logging.info("Train batches[{:5d}]                |"
                " loss {:>5.2f} | pplx {:>7.2f}, bpc {:>8.4f}".format(
                 num_eval_iters, cur_eval_loss, math.exp(cur_eval_loss), 
                 cur_eval_loss / math.log(2)))

      eval_va_bsz = FLAGS.eval_batch_size // FLAGS.num_hosts // strategy.num_replicas_in_sync
      eval_va_dist_mems = create_dist_mems(
              strategy, FLAGS.n_layer, FLAGS.d_model, 
              FLAGS.mem_len, eval_va_bsz
      )

      eval_va_iter = iter(valid_dist_dataset)
      eval_loss.reset_states()
      eval_steps(eval_va_iter, tf.convert_to_tensor(num_eval_iters), \
                 eval_va_dist_mems, FLAGS.eval_batch_size)

      cur_eval_loss = eval_loss.result()
      tf.compat.v1.logging.info("Valid batches[{:5d}]                |"
                " loss {:>5.2f} | pplx {:>7.2f}, bpc {:>8.4f}".format(
                 num_eval_iters, cur_eval_loss, math.exp(cur_eval_loss), 
                 cur_eval_loss / math.log(2)))

      eval_te_bsz = FLAGS.test_batch_size // FLAGS.num_hosts // strategy.num_replicas_in_sync
      eval_te_dist_mems = create_dist_mems(
              strategy, FLAGS.n_layer, FLAGS.d_model, 
              FLAGS.test_mem_len, eval_te_bsz
      )

      eval_te_iter = iter(test_dist_dataset)
      eval_loss.reset_states()
      eval_steps(eval_te_iter, tf.convert_to_tensor(num_eval_iters), \
                 eval_te_dist_mems, FLAGS.test_batch_size)

      cur_eval_loss = eval_loss.result()
      tf.compat.v1.logging.info("Test batches[{:5d}]                 |"
                " loss {:>5.2f} | pplx {:>7.2f}, bpc {:>8.4f}".format(
                 num_eval_iters, cur_eval_loss, math.exp(cur_eval_loss), 
                 cur_eval_loss / math.log(2)))

    if FLAGS.save_steps is not None and cur_step > 0 and cur_step % FLAGS.save_steps == 0:
      chk_path = os.path.join(FLAGS.checkpoint_dir, chk_name)
      options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
      save_path = checkpoint.write(chk_path, options=options)
      tf.compat.v1.logging.info("Model saved in path: {}".format(save_path))


def evaluate(n_token, cutoffs, valid_data, test_data, strategy, chk_name):
  valid_input_fn = valid_data.get_dataset
  valid_record_info = valid_data.get_record_info()
  num_valid_batch = valid_record_info["num_batch"]

  tf.compat.v1.logging.info("num of valid batches {}".format(valid_record_info["num_batch"]))

  test_input_fn = test_data.get_dataset
  test_record_info = test_data.get_record_info()
  num_test_batch = test_record_info["num_batch"]

  tf.compat.v1.logging.info("num of test batches {}".format(test_record_info["num_batch"]))

  # Ensure that number of replicas in sync is same as 'FLAGS.num_core_per_host'
  assert(FLAGS.num_core_per_host == strategy.num_replicas_in_sync)

  ##### Create computational graph for valid dataset
  valid_dataset = valid_input_fn()
  valid_dist_dataset = strategy.experimental_distribute_dataset(valid_dataset)

  ##### Create computational graph for test dataset
  test_dataset = test_input_fn()
  test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

  ##### Create computational graph for model
  with strategy.scope():
    model = create_model(n_token, cutoffs)
    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    eval_loss = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)

    options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
    chk_path = os.path.join(FLAGS.checkpoint_dir, chk_name)
    try:
      checkpoint.read(chk_path, options=options).expect_partial()
      tf.compat.v1.logging.info("Restored checkpoint: {}".format(chk_path))
    except Exception as e:
      tf.compat.v1.logging.info("Exception: " + str(e))
      tf.compat.v1.logging.info("Could not restored checkpoint: {}".format(chk_path))
      return

  @tf.function
  def eval_steps(iterator, steps, dist_mems, bsz):
    ###### The step function for one evaluation step
    def step_fn(inps, lbls, mems):
      mems = tf.squeeze(mems, axis=0)
      loss, new_mems = model(inps, lbls, mems, training=False)
      per_example_loss = tf.reduce_mean(loss, axis=1)
      avg_loss = tf.nn.compute_average_loss(per_example_loss, \
                                            global_batch_size=bsz)
      eval_loss.update_state(avg_loss * strategy.num_replicas_in_sync)
      return tf.expand_dims(new_mems, axis=0)
    for _ in range(steps):
      inps, lbls, _ = next(iterator)
      dist_mems = strategy.run(step_fn, args=(inps, lbls, dist_mems,))
    return dist_mems

  tf.compat.v1.logging.info('Starting evaluation ... ')

  per_replica_bsz = FLAGS.eval_batch_size // FLAGS.num_hosts // strategy.num_replicas_in_sync
  valid_dist_mems = create_dist_mems(
        strategy, FLAGS.n_layer, FLAGS.d_model, 
        FLAGS.mem_len, per_replica_bsz
  )

  eval_iter = iter(valid_dist_dataset)
  eval_steps(eval_iter, tf.convert_to_tensor(num_valid_batch),
             valid_dist_mems, FLAGS.eval_batch_size)

  cur_eval_loss = eval_loss.result()
  tf.compat.v1.logging.info("")
  tf.compat.v1.logging.info("Final evaluation results on valid data ...")
  tf.compat.v1.logging.info("Eval batches[{:5d}]                 |"
          " loss {:>5.2f} | pplx {:>7.2f}, bpc {:>8.4f}".format(
          num_valid_batch, cur_eval_loss, math.exp(cur_eval_loss), 
          cur_eval_loss / math.log(2)))

  per_replica_bsz = FLAGS.test_batch_size // FLAGS.num_hosts // strategy.num_replicas_in_sync
  test_dist_mems = create_dist_mems(
        strategy, FLAGS.n_layer, FLAGS.d_model, 
        FLAGS.test_mem_len, per_replica_bsz
  )
  eval_loss.reset_states()

  eval_iter = iter(test_dist_dataset)
  eval_steps(eval_iter, tf.convert_to_tensor(num_test_batch), \
             test_dist_mems, FLAGS.test_batch_size)

  cur_eval_loss = eval_loss.result()
  tf.compat.v1.logging.info("")
  tf.compat.v1.logging.info("Final evaluation results on test data ...")
  tf.compat.v1.logging.info("Eval batches[{:5d}]                 |"
          " loss {:>5.2f} | pplx {:>7.2f}, bpc {:>8.4f}".format(
          num_test_batch, cur_eval_loss, math.exp(cur_eval_loss), 
          cur_eval_loss / math.log(2)))


def main(unused_argv):
  del unused_argv  # Unused

  # Currently implemented for only one host
  assert(FLAGS.num_hosts == 1)

  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
 
  # Get corpus info
  corpus_info = data_utils.get_corpus_info(FLAGS.corpus_info_path)
  n_token = corpus_info["vocab_size"]
  cutoffs = corpus_info["cutoffs"][1:-1]
  tf.compat.v1.logging.info("n_token {}".format(n_token))

  if FLAGS.do_train:
    # Get train input function
    train_data = data_utils.Dataset(data_dir=FLAGS.data_dir,
                         record_info_dir=FLAGS.record_info_dir,
                         split="train",
                         per_host_bsz=FLAGS.train_batch_size // FLAGS.num_hosts,
                         tgt_len=FLAGS.tgt_len,
                         num_core_per_host=FLAGS.num_core_per_host,
                         num_hosts=FLAGS.num_hosts)

  if FLAGS.do_eval or FLAGS.do_test:
    # Get valid input function
    valid_data = data_utils.Dataset(data_dir=FLAGS.data_dir,
                         record_info_dir=FLAGS.record_info_dir,
                         split="valid",
                         per_host_bsz=FLAGS.eval_batch_size // FLAGS.num_hosts,
                         tgt_len=FLAGS.tgt_len,
                         num_core_per_host=FLAGS.num_core_per_host,
                         num_hosts=FLAGS.num_hosts)

    test_data = data_utils.Dataset(data_dir=FLAGS.data_dir,
                         record_info_dir=FLAGS.record_info_dir,
                         split="test",
                         per_host_bsz=FLAGS.test_batch_size // FLAGS.num_hosts,
                         tgt_len=FLAGS.test_tgt_len,
                         num_core_per_host=FLAGS.num_core_per_host,
                         num_hosts=FLAGS.num_hosts)
  else:
    valid_data = None
    test_data = None

  if FLAGS.use_tpu:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
  else:
    strategy = tf.distribute.get_strategy()
  print("Number of accelerators: ", strategy.num_replicas_in_sync)

  # Ensure that number of replicas in sync is same as 'FLAGS.num_core_per_host'
  assert(FLAGS.num_core_per_host == strategy.num_replicas_in_sync)


  chk_name = 'texl_chk'
  if FLAGS.do_train:
    train(n_token, cutoffs, train_data, valid_data, test_data, strategy, chk_name)
  if FLAGS.do_test:
    evaluate(n_token, cutoffs, valid_data, test_data, strategy, chk_name)


if __name__ == "__main__":
  tf.compat.v1.app.run()
