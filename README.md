# VDCNN
Tensorflow Implementation of Very Deep Convolutional Neural Network for Text Classification.

## Note
This project is a simple Tensorflow implementation of VDCNN model proposed by Conneau et al. [Paper](https://arxiv.org/abs/1606.01781) for VDCNN.

Note: Temporal batch norm not implemented. "Temp batch norm applies same kind of regularization as batch norm, except that the activations in a mini-batch are jointly normalized over temporal instead of spatial locations." Right now this project is using regular Tensorflow batch normalization only.

See another [VDCNN implementation in Pytorch](https://github.com/ArdalanM/nlp-benchmarks) if you feel more comfortable with Pytorch, in which the author is having detailed reproduced results as well. 

It should be noted that the VDCNN paper states that the implementation is done originally in Touch 7.

## Prerequisites

 - Python3
 - Tensorflow 1.0 or higher
 - Numpy

## Datasets
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

## Parameters Setting
For all versions of VDCNN, training and testing is done on a Ubuntu 16.04 Server with Tesla K80, with Momentum Optimizer of decay 0.9, exponential learning rate decay, a evaluation interval of 25, a batch size of 128. Weights are initialized by He initialization proposed in [He et al](https://arxiv.org/pdf/1502.01852). Batch normalizations are using a decay of 0.999.

(There are tons of factors that can influence the testing accuracy of the model, but overall this project should be good to go. Training of a deep CNN model is not a easy task, patience is everything. -_-)

## Experiments

TODO: Testing of more NLP benchmark datasets and presenting detailed results.

Results are reported as follows:  (i) / (ii)
 - (i): Test set accuracy reported by the paper (acc = 100% - error_rate)
 - (ii): Test set accuracy reproduced by this Tensorflow implementation

Results for Max Pooling:

|      Depth      |      ag_news      |      DBPedia      |     Sogou News    |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|
|     9 layers    |  90.83 / xx.xxxx  |  98.65 / xx.xxxx  |  96.30 / xx.xxxx  |  
|    17 layers    |  91.12 / xx.xxxx  |  98.60 / xx.xxxx  |  96.46 / xx.xxxx  |
|    29 layers    |  91.27 / xx.xxxx  |  98.71 / xx.xxxx  |  96.64 / xx.xxxx  |

Results for K-max Pooling(k = 8):

|      Depth      |      ag_news      |      DBPedia      |     Sogou News    |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|
|     9 layers    |  90.17 / xx.xxxx  |  98.44 / xx.xxxx  |  96.42 / xx.xxxx  |  
|    17 layers    |  90.61 / xx.xxxx  |  98.39 / xx.xxxx  |  96.49 / xx.xxxx  |
|    29 layers    |  91.33 / xx.xxxx  |  98.59 / xx.xxxx  |  96.82 / xx.xxxx  |

## TODOs

 - (i): Optional Shortcut
 - (ii): Three types of downsampling between blocks: 
         (a) maxpooling(Done)
         (b) k-maxpooling(Tensorflow doesn't support this operation natively so I'll keep looking for a way :( )
         (c) Convolution with Stride 2
 - (iii) Testing for all datasets and a detailed accuracy for 9, 17 and 28 depth.

## Reference
[Original preprocessing codes and VDCNN Implementation By geduo15](https://github.com/geduo15/Very-Deep-Convolutional-Networks-for-Natural-Language-Processing-in-tensorflow)

[Train Script and data iterator from Convolutional Neural Network for Text Classification](https://github.com/dennybritz/cnn-text-classification-tf)

[NLP Datasets Gathered by ArdalanM and Others](https://github.com/ArdalanM/nlp-benchmarks)
