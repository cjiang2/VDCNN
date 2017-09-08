# VDCNN
Tensorflow Implementation of Very Deep Convolutional Neural Network for Text Classification.

## Note
This project is a simple Tensorflow implementation of VDCNN model proposed by Conneau et al. [Paper](https://arxiv.org/abs/1606.01781) for VDCNN.

Note: Temporal batch norm not implemented. "Temp batch norm applies same kind of regularization as batch norm, except that the activations in a mini-batch are jointly normalized over temporal instead of spatial locations." Right now this project is using regular Tensorflow batch normalization only.

TODO: Testing of NLP benchmark Datasets, including AG's News, DBPedia, Sogou News, Imdb, etc.

## Prerequisites

 - Python3
 - Tensorflow 1.0 or higher
 - Numpy

## Datasets
The original paper tests several NLP datasets, including DBPedia, AG's News, Sogou News and etc. "data_helper.py" operates with CSV format train and test files.

I'm recommanding another [VDCNN implementation in Pytorch](https://github.com/ArdalanM/nlp-benchmarks), in which the author is also providing all those NLP datasets in CSV format that data iterator can process. (Honestly it's much easier to finish up a VDCNN implementation in Pytorch I think -_-)

## Parameters Setting
TODO: Testing of more NLP benchmark datasets, testing of 17 layers VDCNN, finish-up implementation of 28 layers VDCNN. 

For all versions of VDCNN, training and testing is done on a Ubuntu 16.04 Server with Tesla K80, with Momentum Optimizer of decay 0.9, exponential learning rate decay, a evaluation interval of 50, a batch size of 128. Weights are initialized by He initialization proposed in [He et al](https://arxiv.org/pdf/1502.01852). (There are tons of factors that can influence the testing accuracy of the model, but overall this project should be good to go. Training of a deep CNN model is not a easy task, patience is everything. -_-)

## Experiments
Results are reported as follows:  (i) / (ii)
 - (i): Test set accuracy reported by the paper  
 - (ii): Test set accuracy reproduced here  

|                  | ag_news |  dbPedia  |   Sogou News   |  Imdb  |
|:----------------:|:-------:|:---------:|:--------------:|:------:|
|VDCNN (9 layers)  |    /    |     /     |        /       |    /   |
|VDCNN (17 layers) |    /    |     /     |        /       |    /   |
|VDCNN (29 layers  |    /    |     /     |        /       |    /   |

## Reference
[Original VDCNN Implementation By geduo15](https://github.com/geduo15/Very-Deep-Convolutional-Networks-for-Natural-Language-Processing-in-tensorflow)

[Train Script and data iterator from Convolutional Neural Network for Text Classification](https://github.com/dennybritz/cnn-text-classification-tf)

[All Those NLP Datasets Gathered by ArdalanM and Others](https://github.com/ArdalanM/nlp-benchmarks)
