# Keras implementation of [Transformer-XL](https://arxiv.org/abs/1901.02860) Model. 
The implementated model can be trained and evaluated on both TPU and GPU. Our implementation is different from the [implementation of Zhilin Yang](https://github.com/kimiyoung/transformer-xl.git) in two aspects:
* Yang's tensorflow TPU pipeline uses the high level [TPUEstimator](https://www.tensorflow.org/api_docs/python/tf/compat/v1/estimator/tpu/TPUEstimator) API which does not support recurrent memory. Consecuently, Yang has modified the source code of TPUEstimator to make it work with recurrent memory. We have implemented a custom training and evaluation pipeline using [tf.function](https://www.tensorflow.org/api_docs/python/tf/function) and [tf.distribute](https://www.tensorflow.org/api_docs/python/tf/distribute) APIs directly.
* A typical TPU pipeline reads the data from [Google Cloud Storage Bucket](https://cloud.google.com/storage/docs/json_api/v1/buckets) during training and evaluation. Thus, they require the data to be resided in Google Cloud Storage Bucket. However, in our implementation, we read the data into memory at the beginning of training/evaluation iterations. Thus, no Google Cloud Storage account is needed for our pipeline. On the downside, the data should be small enough to be fully stored in memory.