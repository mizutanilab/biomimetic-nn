# Schizophrenia-mimcking layer
Schizophrenia-mimicking layer is based on our [study on nanometer-scale 3D structure of neuronal network in schizophrenia cases](https://www.nature.com/articles/s41398-019-0427-4). The synchrotron raditation nano-CT analysis revealed that neurites of the anterior cingulate cortex are thin and tortuous in schizophrenia compared to control cases. Analysis of another area called temporal cortex delineated [individual differences even between healthy controls](https://arxiv.org/abs/2007.00212). So we translated these findings into newly designed layers that mimic connection impairment in schizophrenia. We call them 'schizophrenia layer'. Test calculations using the schizophrenia layer indicated that 80% of weights can be eliminated without any changes in training procedures or network configuration. Very interestingly the schizophrenia layer completely suppresses overfitting and outperforms fully connected layer. Here is a typical example obtained using [python code](paperfig/) (num_epoch=200, idlist=\[0.5, 0.0\], num_repeat=10). <BR><BR>
![training example](paperfigs/CIFAR_CNN_ConcurrTraj200913.png)

## How to implement schizophrenia-mimicking layer in your network
Our original code runs on Tensorflow/Keras. 
1. Download 'schizo.py' file to your working directory where your *.py file is placed. 
2. The following is an example code using a 'SzDense' layer in place of 'Dense' layer: 
```
from tensorflow import keras
from tensorflow.keras import layers
import schizo

model = keras.Sequential([
  layers.Flatten(),
  # layers.Dense(512, activation='relu', kernel_initializer='he_normal'),
  schizo.SzDense(512, param_reduction=0.5, activation='relu', kernel_initializer='he_normal'),
  layers.Dense(num_class, activation='softmax')
])
```
In this example, the `layers.Dense` layer was commented out to replace it with a `schizo.SzDense` layer of 50% parameter reduction, which is defined with argument `param_reduction`. The reduction ratio best fit to your network depends on the network configuration, but in most cases 50-70% seems to give good results. We recommend 50% as a first choice. 

## Code used for preparing our paper figures
under construction.

## References
Mizutani et al. (2020) Schizophrenia-mimicking layers outperform conventional neural network layers. [arXiv](https://arxiv.org/abs/2009.10887)<BR>
Mizutani et al. (2019) Three-dimensional alteration of neurites in schizophrenia. <i>Transl Psychiatry</i> <b>9</b>, 85. [nature.com](https://www.nature.com/articles/s41398-019-0427-4)<BR>
Mizutani et al. (2020) Structural diverseness of neurons between brain areas and between cases. [arXiv](https://arxiv.org/abs/2007.00212)<BR>

