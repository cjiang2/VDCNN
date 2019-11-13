# VDCNN #
*Tensorflow 2.0 Implementation of Very Deep Convolutional Neural Network for Text Classification.*

## Note ##
This repository is a simple Tensorflow 2.0 implementation of the VDCNN model proposed by Conneau et al. in [their 2016 paper](https://arxiv.org/abs/1606.01781). It is based off of [this Keras implementation by zonetrooper32](https://github.com/zonetrooper32/VDCNN).

Note: Temporal batch norm not implemented. "Temp batch norm applies same kind of regularization as batch norm, except that the activations in a mini-batch are jointly normalized over temporal instead of spatial locations." Right now this project is using regular Tensorflow batch normalization only.

It should be noted that the VDCNN paper states that the implementation is done originally in Touch 7.

## Prerequisites ##

 - Python 3
 - Tensorflow 2.0
 - numpy

## Datasets ##
The original paper tests several NLP datasets, including DBPedia, AG's News, Sogou News and etc. "data_helper.py" operates with CSV format train and test files.

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

## Parameters Setting ##
For all versions of VDCNN, training and testing is done on a Ubuntu 16.04 Server with Tesla K80, with Momentum Optimizer of decay 0.9, exponential learning rate decay, a evaluation interval of 25, a batch size of 128. Weights are initialized by He initialization proposed in [He et al](https://arxiv.org/pdf/1502.01852). Batch normalizations are using a decay of 0.999.

## References ##

[Keras implementation by zonetrooper32](https://github.com/zonetrooper32/VDCNN)

[Original preprocessing codes and VDCNN Implementation By geduo15](https://github.com/geduo15/Very-Deep-Convolutional-Networks-for-Natural-Language-Processing-in-tensorflow)

[Train Script and data iterator from Convolutional Neural Network for Text Classification](https://github.com/dennybritz/cnn-text-classification-tf)

[NLP Datasets Gathered by ArdalanM and Others](https://github.com/ArdalanM/nlp-benchmarks)
