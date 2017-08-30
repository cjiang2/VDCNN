# VDCNN
Tensorflow Implementation of Very Deep Convolutional Neural Network for Text Classification.

## Note
This project is a simple Tensorflow implementation of VDCNN model proposed by Conneau et al. [Paper](https://arxiv.org/abs/1606.01781) for VDCNN.
TODO: Temporal batch norm: "Temp batch norm applies same kind of regularization as batch norm, except that the activations in a mini-batch are jointly normalized over temporal instead of spatial locations." Right now this project is using regular batch normalization only.
TODO: Testing of more NLP benchmark datasets, testing of 17 layers VDCNN, implementation of 28 layers VDCNN. 

## Prerequisites

 - Python3
 - Tensorflow 1.0 or higher
 - Numpy

## Datasets
The original paper tests several NLP datasets, including DBPedia, AG's News, Sogou News and etc. "data_helper.py" operates with CSV format train and test files.
I'm recommanding another [VDCNN implementation in Pytorch](https://github.com/ArdalanM/nlp-benchmarks), in which the author is also providing all those NLP datasets in CSV format that data iterator can process. (Honestly it's much easier to finish up a VDCNN implementation in Pytorch I think -_-)

## Experiment
TODO: Testing of more NLP benchmark datasets, testing of 17 layers VDCNN, finish-up implementation of 28 layers VDCNN. 
For 9 layer VDCNN, training and testing is done on a Ubuntu 16.04 Server with Tesla K80 with DBPedia(around 98%) and AG's News(around 89%), with Momentum Optimizer of decay 0.9, exponential learning rate decay, a evaluation interval of 100, a batch size of 128. Weights are initialized by He initialization proposed in [He et al](https://arxiv.org/pdf/1502.01852).

## Reference
[Original VDCNN Implementation By geduo15](https://github.com/geduo15/Very-Deep-Convolutional-Networks-for-Natural-Language-Processing-in-tensorflow)
[Train Script and data iterator from Convolutional Neural Network for Text Classification](https://github.com/dennybritz/cnn-text-classification-tf)
[Datasets from ArdalanM](https://github.com/ArdalanM/nlp-benchmarks)