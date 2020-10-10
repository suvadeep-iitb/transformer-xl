from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from functools import partial

from collections import Counter, OrderedDict
import pickle
import json
import multiprocessing as mp

import numpy as np

from absl import flags
import tensorflow as tf
from vocabulary import Vocab

from tensorflow.compat.v1.gfile import Exists as exists
from tensorflow.compat.v1.gfile import MakeDirs as makedirs
from tensorflow.compat.v1.gfile import Glob as glob


def _preprocess(shard, train, vocab, save_dir, bsz, tgt_len, num_shuffle):
  file_names = []
  num_batch = 0

  path = train[shard]
  data_shard = vocab.encode_file(path, ordered=False, add_double_eos=True)

  for shuffle in range(num_shuffle):
    basename = "train-{:03d}-{:02d}".format(shard, shuffle)
    print("Processing shard {} shuffle {}".format(shard, shuffle))

    np.random.shuffle(data_shard)
    file_name, num_batch_shuffle = create_ordered_tfrecords(
        save_dir, basename, np.concatenate(data_shard), bsz, tgt_len)
    file_names.append(file_name)
    num_batch += num_batch_shuffle

  return file_names, num_batch


class Corpus(object):
  def __init__(self, path, dataset, *args, **kwargs):
    self.dataset = dataset
    self.vocab = Vocab(*args, **kwargs)

    if self.dataset in ["ptb", "wt2", "enwik8", "text8", "sb2", "sb92"]:
      self.vocab.count_file(os.path.join(path, "train.txt"))
      self.vocab.count_file(os.path.join(path, "valid.txt"))
      self.vocab.count_file(os.path.join(path, "test.txt"))
    elif self.dataset in ["wt103", "wt103small"]:
      self.vocab.count_file(os.path.join(path, "train.txt"))
    elif self.dataset == "lm1b":
      train_path_pattern = os.path.join(
          path, "1-billion-word-language-modeling-benchmark-r13output",
          "training-monolingual.tokenized.shuffled", "news.en-*")
      train_paths = glob(train_path_pattern)

      # the vocab will load from file when build_vocab() is called
      # for train_path in sorted(train_paths):
      #   self.vocab.count_file(train_path, verbose=True)

    self.vocab.build_vocab()

    if self.dataset in ["ptb", "sb2", "sb92"]:
      self.train = self.vocab.encode_file(
          os.path.join(path, "train.txt"), ordered=True)
      self.valid = self.vocab.encode_file(
          os.path.join(path, "valid.txt"), ordered=True)
      self.test  = self.vocab.encode_file(
          os.path.join(path, "test.txt"), ordered=True)
    elif self.dataset in ["wt2", "wt103", "wt103small"]:
      self.train, self.train_boundary = self.vocab.encode_file(
          os.path.join(path, "train.txt"), ordered=True, 
          ret_doc_boundary=True, pattern="\=[^=]+\=")
      self.valid, self.valid_boundary = self.vocab.encode_file(
          os.path.join(path, "valid.txt"), ordered=True, 
          ret_doc_boundary=True, pattern="\=[^=]+\=")
      self.test, self.test_boundary = self.vocab.encode_file(
          os.path.join(path, "test.txt"), ordered=True, 
          ret_doc_boundary=True, pattern="\=[^=]+\=")
    elif self.dataset in ["enwik8", "text8"]:
      self.train = self.vocab.encode_file(
          os.path.join(path, "train.txt"), ordered=True, add_eos=False)
      self.valid = self.vocab.encode_file(
          os.path.join(path, "valid.txt"), ordered=True, add_eos=False)
      self.test  = self.vocab.encode_file(
          os.path.join(path, "test.txt"), ordered=True, add_eos=False)
    elif self.dataset == "lm1b":
      self.train = train_paths
      valid_path = os.path.join(path, "valid.txt")
      test_path = valid_path
      self.valid = self.vocab.encode_file(
          valid_path, ordered=True, add_double_eos=True)
      self.test  = self.vocab.encode_file(
          test_path, ordered=True, add_double_eos=True)

    if self.dataset == "sb92":
      self.cutoffs = [0, 10000, 20000] + [len(self.vocab)]
    elif self.dataset == "wt103small":
      self.cutoffs = [0, 20000, 40000] + [len(self.vocab)]
    elif self.dataset == "wt103":
      self.cutoffs = [0, 20000, 40000, 200000] + [len(self.vocab)]
    elif self.dataset == "lm1b":
      self.cutoffs = [0, 60000, 100000, 640000] + [len(self.vocab)]
    else:
      self.cutoffs = []


  def convert_to_tfrecords(self, split, save_dir, bsz, tgt_len, **kwargs):
    FLAGS = kwargs.get('FLAGS')

    file_names = []

    record_name = "record_info-{}.bsz-{}.tlen-{}.json".format(
           split, bsz, tgt_len)

    record_info_path = os.path.join(save_dir, record_name)

    if self.dataset in ["ptb", "enwik8", "text8", "sb2", "sb92"]:
      data = getattr(self, split)
      file_name, num_batch = create_ordered_tfrecords(
          save_dir, split, data, bsz, tgt_len, num_passes=FLAGS.num_passes)
      file_names.append(file_name)
    if self.dataset in ["wt2", "wt103", "wt103small"]:
      data = getattr(self, split)
      boundary = getattr(self, split+"_boundary")
      file_name, num_batch = create_ordered_tfrecords(
          save_dir, split, data, bsz, tgt_len, num_passes=FLAGS.num_passes, 
          boundary=boundary)
      file_names.append(file_name)
    elif self.dataset == "lm1b":
      if split == "train":
        np.random.seed(123456)
        num_batch = 0

        if FLAGS.num_procs > 1:
          _preprocess_wrapper = partial(_preprocess,
              train=self.train, vocab=self.vocab, save_dir=save_dir,
              bsz=bsz, tgt_len=tgt_len, num_shuffle=FLAGS.num_shuffle)

          pool = mp.Pool(processes=FLAGS.num_procs)
          results = pool.map(_preprocess_wrapper, range(len(self.train)))
          for res in results:
            file_names.extend(res[0])
            num_batch += res[1]
        else:
          for shard, path in enumerate(self.train):
            data_shard = self.vocab.encode_file(path, ordered=False,
                                                add_double_eos=True)

            num_shuffle = FLAGS.num_shuffle

            for shuffle in range(num_shuffle):
              print("Processing shard {} shuffle {}".format(shard, shuffle))
              basename = "train-{:03d}-{:02d}".format(shard, shuffle)
              np.random.shuffle(data_shard)
              file_name, num_batch_ = create_ordered_tfrecords(
                  save_dir, basename, np.concatenate(data_shard), bsz, tgt_len)
              file_names.append(file_name)
              num_batch += num_batch_

      else:
        file_name, num_batch = create_ordered_tfrecords(
            save_dir, split, getattr(self, split), bsz, tgt_len)
        file_names.append(file_name)

    with open(record_info_path, "w") as fp:
      record_info = {
        "filenames": file_names,
        "num_batch": num_batch
      }
      if self.dataset in ["wt2", "wt103", "wt103small"]:
        record_info["boundary"] = True
      else:
        record_info["boundary"] = False
      json.dump(record_info, fp)


def _int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _float_feature(values):
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def batchify(data, batch_size, num_passes, boundary=None):
  """
    Since TPU training requires entire [bsz x tgt_len] chunks, it can discard
    as many as `bsz * tgt_len` tokens in training. When `bsz` and `tgt_len` are 
    both large, as in the case of TPU training for Transformer-XL, the problem
    may lead to detectable performance drop. 

    Here, we use multiple randomly shifted copies to deal with this problem.
  """
  if num_passes > 1:
    data_len = len(data)
    double_data = np.concatenate([data, data])
    data_list = []
    if boundary is not None:
      assert len(boundary) == data_len
      double_b = np.concatenate([boundary, boundary])
      b_list = []
    for i in range(num_passes):
      start = np.random.randint(0, data_len)
      data_list.append(double_data[start:start+data_len])
      if boundary is not None:
        b_list.append(double_b[start:start+data_len])
    data = np.concatenate(data_list)
    if boundary is not None:
      boundary = np.concatenate(b_list)

  num_step = len(data) // batch_size
  data = data[:batch_size * num_step]
  data = data.reshape(batch_size, num_step)
  if boundary is not None:
    boundary = boundary[:batch_size * num_step]
    boundary = boundary.reshape(batch_size, num_step)
    return data, boundary

  return data


def create_ordered_tfrecords(save_dir, basename, data, batch_size, tgt_len,
                             num_passes=1, boundary=None):

  file_name = "{}.bsz-{}.tlen-{}.tfrecords".format(
        basename, batch_size, tgt_len)

  save_path = os.path.join(save_dir, file_name)
  record_writer = tf.compat.v1.python_io.TFRecordWriter(save_path)

  if boundary is not None:
    batched_data, batched_boundary = batchify(data, batch_size, num_passes, boundary)
  else:
    batched_data = batchify(data, batch_size, num_passes)

  num_batch = 0
  # for t in range(0, batched_data.shape[1] - tgt_len - 1, tgt_len):
  for t in range(0, batched_data.shape[1] - 1, tgt_len):
    cur_tgt_len = min(batched_data.shape[1] - 1 - t, tgt_len)
    if cur_tgt_len < tgt_len: 
      break
    if num_batch % 500 == 0:
      print("  processing batch {}".format(num_batch))
    for idx in range(batch_size):
      inputs = batched_data[idx, t:t + cur_tgt_len]
      labels = batched_data[idx, t + 1:t + cur_tgt_len + 1]
      if boundary is not None:
        bnds = batched_boundary[idx, t:t + cur_tgt_len]
        # features dict
        feature = {
          "inputs": _int64_feature(inputs),
          "labels": _int64_feature(labels),
          "bnds": _int64_feature(bnds),
        }
      else:
        # features dict
        feature = {
          "inputs": _int64_feature(inputs),
          "labels": _int64_feature(labels),
        }

      example = tf.train.Example(features=tf.train.Features(feature=feature))
      record_writer.write(example.SerializeToString())

    num_batch += 1

  record_writer.close()
  print("Done writing {}. batches: {}".format(file_name, num_batch))

  return file_name, num_batch


def get_lm_corpus(data_dir, dataset):
  fn = os.path.join(data_dir, "cache.pkl")

  if exists(fn):
    print("Loading cached dataset...")
    with open(fn, "rb") as fp:
      corpus = pickle.load(fp, encoding="latin1")
  else:
    print("Producing dataset...")
    kwargs = {}
    if dataset in ["wt103", "wt2", "sb2"]:
      kwargs["special"] = ["<eos>"]
      kwargs["lower_case"] = False
    elif dataset == "sb92":
      kwargs["special"] = ["<eos>"]
      kwargs["lower_case"] = False
    elif dataset == "wt103small":
      kwargs["special"] = ["<UNK>", "<eos>"]
      kwargs["lower_case"] = False
      kwargs["min_freq"] = 30
    elif dataset == "ptb":
      kwargs["special"] = ["<eos>"]
      kwargs["lower_case"] = True
    elif dataset == "lm1b":
      kwargs["special"] = []
      kwargs["lower_case"] = False
      kwargs["vocab_file"] = os.path.join(data_dir, "1b_word_vocab.txt")
    elif dataset in ["enwik8", "text8"]:
      pass

    corpus = Corpus(data_dir, dataset, **kwargs)

    print("Saving dataset...")
    with open(fn, "wb") as fp:
      pickle.dump(corpus, fp, protocol=2)

    corpus_info = {
      "vocab_size" : len(corpus.vocab),
      "cutoffs" : corpus.cutoffs,
      "dataset" : corpus.dataset
    }
    with open(os.path.join(data_dir, "corpus-info.json"), "w") as fp:
      json.dump(corpus_info, fp)

  return corpus


def main(unused_argv):
  del unused_argv  # Unused

  corpus = get_lm_corpus(FLAGS.data_dir, FLAGS.dataset)

  save_dir = os.path.join(FLAGS.data_dir, "tfrecords")
  if not exists(save_dir):
    makedirs(save_dir)

  # test mode
  if FLAGS.per_host_test_bsz > 0:
    corpus.convert_to_tfrecords("test", save_dir, FLAGS.per_host_test_bsz,
                                FLAGS.tgt_len, FLAGS=FLAGS)
    return

  for split, batch_size in zip(
      ["train", "valid"],
      [FLAGS.per_host_train_bsz, FLAGS.per_host_valid_bsz]):

    if batch_size <= 0: continue
    print("Converting {} set...".format(split))
    corpus.convert_to_tfrecords(split, save_dir, batch_size, FLAGS.tgt_len,
                                FLAGS=FLAGS)


def load_record_info(record_info_dir, split, per_host_bsz, tgt_len):
  record_name = "record_info-{}.bsz-{}.tlen-{}.json".format(
        split, per_host_bsz, tgt_len)

  record_info_path = os.path.join(record_info_dir, record_name)
  with open(record_info_path, "r") as fp:
    record_info = json.load(fp)

  return record_info


class Dataset:
  def __init__(self, data_dir, record_info_dir, split, per_host_bsz, 
               tgt_len, num_core_per_host, num_hosts=1):
    self.data_dir = data_dir
    self.record_info_dir = record_info_dir
    self.split = split
    self.per_host_bsz = per_host_bsz
    self.tgt_len = tgt_len
    self.num_core_per_host = num_core_per_host
    self.num_hosts = num_hosts

    # Currently implemented for single host
    assert (num_hosts == 1)
    assert (per_host_bsz % num_core_per_host == 0)

    record_info = load_record_info(record_info_dir, split, per_host_bsz, tgt_len)

    file_names = record_info["filenames"]
    num_batch = record_info["num_batch"]
    boundary = record_info["boundary"]

    tf.compat.v1.logging.info("[{}] File names {}".format(split, file_names))

    def parser(record):
      record_spec = {
          "inputs": tf.compat.v1.FixedLenFeature([tgt_len], tf.int64),
          "labels": tf.compat.v1.FixedLenFeature([tgt_len], tf.int64),
      }
      if boundary:
        record_spec["bnds"] = tf.compat.v1.FixedLenFeature([tgt_len], tf.int64)

      # retrieve serialized example
      example = tf.compat.v1.parse_single_example(
          serialized=record,
          features=record_spec)

      # cast int64 into int32
      for key in list(example.keys()):
        val = example[key]
        if key == "bnds":
          val = tf.cast(val, tf.float32)
        elif val.dtype == tf.int64:
          val = tf.compat.v1.to_int32(val)
        example[key] = val

      return example

    file_paths = []
    for file_name in file_names:
      file_path = os.path.join(data_dir, file_name)
      file_paths.append(file_path)

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    if split == "train" and len(file_paths) > 1:
      # TO DO: shuffle().repeat() needs to be supported
      dataset = dataset.shuffle(len(file_paths))
    dataset = tf.data.TFRecordDataset(dataset)
    dataset = dataset.map(parser)

    def get_as_data_dict(dataset):
      batch_size = per_host_bsz

      ds_size = dataset.reduce(np.int64(0), lambda x, _: x+1).numpy()
      ds_size = (ds_size // batch_size) * batch_size
      print('Dataset size: '+str(ds_size))

      data_dict = {}
      elem = next(dataset.as_numpy_iterator())
      for k, v in elem.items():
        data_dict[k] = np.zeros((ds_size,)+v.shape, dtype=v.dtype)
        print(k+': shape '+str(data_dict[k].shape)+': type '+str(data_dict[k].dtype.name))

      dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(5)

      cur_idx = 0
      for example in dataset.as_numpy_iterator():
        for k, v in example.items():
          if len(example[k].shape) == 2:
            dim = example[k].shape[1]
            data_dict[k][cur_idx: cur_idx+batch_size, :dim] = example[k]
          else:
            data_dict[k][cur_idx: cur_idx+batch_size, :, :] = example[k]
        cur_idx = cur_idx+batch_size
        
      return data_dict

    # Read data into memory
    print('Reading tfrecords file into memory...')
    self.data_dict = get_as_data_dict(dataset)
    print('Reading complete')

    if split == "train" and num_hosts > 1:
      record_info["num_batch"] = num_batch // num_hosts
    self.record_info = record_info


  def get_dataset(self):
    # assert that the batch_size is integer multiple of num_core_per_host
    assert (self.per_host_bsz % self.num_core_per_host == 0)

    dataset = tf.data.Dataset.from_tensor_slices(self.data_dict)

    def parser(example):
      inps = example["inputs"]
      lbls = example["labels"]
      if self.record_info["boundary"]:
        bnds = example["bnds"]
      else:
        bnds = tf.zeros(inps.shape, dtype=tf.float32)
      return inps, lbls, bnds
    dataset = dataset.map(parser)

    if self.split == "train":
      dataset = dataset.batch(self.per_host_bsz, drop_remainder=True).repeat()
      dataset = dataset.prefetch(self.per_host_bsz)
    else:
      dataset = dataset.batch(self.per_host_bsz, drop_remainder=True)

    return dataset


  def get_record_info(self):
    return self.record_info



def get_corpus_info(corpus_info_path):
  with open(corpus_info_path, "r") as fp:
    corpus_info = json.load(fp)
  return corpus_info


if __name__ == "__main__":
  FLAGS = flags.FLAGS
  flags.DEFINE_string("data_dir", None,
        help="Location of the data corpus")
  flags.DEFINE_enum("dataset", "wt103",
        ["ptb", "wt2", "wt103", "wt103small", "lm1b", "enwik8", "text8", "sb2", "sb92"],
        help="Dataset name.")
  flags.DEFINE_integer("per_host_train_bsz", 60,
        help="train batch size each host")
  flags.DEFINE_integer("per_host_valid_bsz", 60,
        help="valid batch size each host")
  flags.DEFINE_integer("per_host_test_bsz", 0,
        help="If > 0, enter test mode and process test set only."
             "Otherwise, process train and dev sets only.")
  flags.DEFINE_integer("tgt_len", 70,
        help="number of tokens to predict")
  flags.DEFINE_integer("max_batch", -1,
        help="run in debug mode")
  flags.DEFINE_integer("num_core_per_host", 8,
        help="Currently unused, kept for backward compatibility")
  flags.DEFINE_bool("debug", default=False,
        help="Process only the first batch without shuffle for lm1b.")
  flags.DEFINE_integer("num_procs", 1,
        help="number of processes")
  flags.DEFINE_integer("num_passes", 10,
        help="number of passes")
  flags.DEFINE_integer("num_shuffle", 4,
        help="number of shuffles for lm1b")
  flags.DEFINE_bool("use_tpu", False,
        help="now reduntant, kept for backward compatibility")

  tf.compat.v1.app.run(main)
