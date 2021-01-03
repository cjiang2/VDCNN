# VDCNN
Tensorflow Implementation of Very Deep Convolutional Neural Network for Text Classification.

## Update
As it turns out, the original repo has too many implementation error (fresh eyes after many years), and it is extremely outdated (Tensorflow has updated to 2.0+ with eager support). Works are being done slowly on updating this code repo.

(Side Note: PyTorch is so much better. :O )

To-Dos:

 - [x] Re-implement VDCNN, correctly. 
    - [x] 3 types of pooling operations. (Convolutional pooling, k-max pooling, max pooling)
    - [x] Dotted line shortcuts. (Conv1x1 projection, identity with zero-padding)
    - [] Double-check re-implementation.

 - [] Dataset codebase. Need to refer to newer tf.datasets implementation.

 - [] Training and evaluating codebase. This should be simpler with tf.keras, once dataset objects are done.

## Note
This repository is a Tensorflow implementation of VDCNN model proposed by Conneau et al. [Paper](https://arxiv.org/abs/1606.01781) for VDCNN.

## Prerequisites

 - Python3
 - Tensorflow >= 2.0
 - Numpy

## Reference
[Original preprocessing codes and VDCNN Implementation By geduo15](https://github.com/geduo15/Very-Deep-Convolutional-Networks-for-Natural-Language-Processing-in-tensorflow)

[Train Script and data iterator from Convolutional Neural Network for Text Classification](https://github.com/dennybritz/cnn-text-classification-tf)

[NLP Datasets Gathered by ArdalanM and Others](https://github.com/ArdalanM/nlp-benchmarks)
