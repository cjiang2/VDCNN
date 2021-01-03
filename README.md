# VDCNN
Tensorflow Implementation of Very Deep Convolutional Neural Network for Text Classification, proposed by [Conneau et al](https://arxiv.org/abs/1606.01781).

Archiecture for VDCNN is now **correctly re-implemented with Tensorflow 2 and tf.keras support**. A simple training interface is implemented following [Tensorflow 2 Expert Tutorial](https://www.tensorflow.org/tutorials/quickstart/advanced). Feel free to contribute additional utilities like TensorBoard support.

**Side Note, if you are a newcomer for NLP text classification:** 
 - Please checkout new SOTA NLP methods like [transformers](https://github.com/huggingface/transformers) or [Bert](https://github.com/google-research/bert). 
 
 - Check out [PyTorch](https://pytorch.org/) for **MUCH BETTER** dynamic graphing and dataset object support. 
   - Current VDCNN implementation is also extremely easy to be ported onto PyTorch.

## Prerequisites

 - Python3
 - Tensorflow >= 2.0
 - tensorflow-datasets
 - numpy
 
## Datasets
The original paper tests several NLP datasets, including DBPedia, AG's News, Sogou News and etc. 

[tensorflow-datasets](https://www.tensorflow.org/datasets/catalog/ag_news_subset) is used to support AG's News dataset.

Downloads of those NLP text classification datasets can be found here (Many thanks to ArdalanM):

| Dataset                | Classes | Train samples | Test samples | source |
|------------------------|:---------:|:---------------:|:--------------:|:--------:|
| AG’s News              |    4    |    120 000    |     7 600    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Sogou News             |    5    |    450 000    |    60 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| DBPedia                |    14   |    560 000    |    70 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Yelp Review Polarity   |    2    |    560 000    |    38 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Yelp Review Full       |    5    |    650 000    |    50 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Yahoo! Answers         |    10   |   1 400 000   |    60 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Amazon Review Full     |    5    |   3 000 000   |    650 000   |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Amazon Review Polarity |    2    |   3 600 000   |    400 000   |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|

## Parameters Setting
The original paper suggests the following details for training:
 - SGD optimizer with lr 1e-2, decay 0.9. 
 - 10 - 15 epochs for convergence.
 - He Initialization.


Some additional parameter settings for this repo:
 - Gradient clipping with norm_value of 7.0, to stablize the training.


Skip connections and pooling are correctly implemented now:
 - k-maxpooling.
 - maxpooling with kernel size of 3 and strides 2.
 - conv pooling with K_i convolutional layer.
 
 
For dotted skip connections:
 - Identity with zero padding.
 - Conv1D with kernel size of 1.
 
 
Please refer to Conneau et al for their methodology and experiment section in more detail.

## Experiments
Results are reported as follows:  (i) / (ii)
 - (i): Test set accuracy reported by the paper (acc = 100% - error_rate)
 - (ii): Test set accuracy reproduced by this Keras implementation

TODO: Feel free to report your own experimental results in the following format:

Results for "Identity" Shortcut, "k-max" Pooling:
|      Depth      |      ag_news      |      DBPedia      |     Sogou News    |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|
|     9 layers    |  90.17 / xx.xxxx  |  98.44 / xx.xxxx  |  96.42 / xx.xxxx  |  
|    17 layers    |  90.61 / xx.xxxx  |  98.39 / xx.xxxx  |  96.49 / xx.xxxx  |
|    29 layers    |  91.33 / xx.xxxx  |  98.59 / xx.xxxx  |  96.82 / xx.xxxx  |
|    49 layers    |  xx.xx / xx.xxxx  |  xx.xx / xx.xxxx  |  xx.xx / xx.xxxx  |

## Reference
[Original preprocessing codes and VDCNN Implementation By geduo15](https://github.com/geduo15/Very-Deep-Convolutional-Networks-for-Natural-Language-Processing-in-tensorflow)

[Train Script and data iterator from Convolutional Neural Network for Text Classification](https://github.com/dennybritz/cnn-text-classification-tf)

[NLP Datasets Gathered by ArdalanM and Others](https://github.com/ArdalanM/nlp-benchmarks)
