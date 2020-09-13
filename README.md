# Schizophrenia-mimcking layer
Schizophrenia-mimicking layer was inspired from the [analysis of humran brain tissues of schizophrenia patients](https://www.nature.com/articles/s41398-019-0427-4). The difference was found also between healthy individuals ([arXiv](https://arxiv.org/abs/2007.00212)). Very interestingly this layer completely suppresses overfitting and outperforms fully connected layer. We have experienced it in many examples.

## How to implement schizophrenia layer in your network
1. Download 'schizo.py' file to your working directory where your *.py file is placed.
2. Replace 'Dense' layer with Schizo layer like this: 
```
from tensorflow import keras
from tensorflow.keras import layers
import schizo as Schizo
model = keras.Sequential([
  layers.Flatten(),
  # layers.Dense(512, activation='relu', kernel_initializer='he_normal'),
  Schizo(512, reduction_ratio=0.5, form='diagonal', activation='relu', kernel_initializer='he_normal'),
  layers.Dense(num_class, activation='softmax')
])
```
In this example, the Dense hidden layer were commented out to replace it with a Schizo layer of 50% parameter reduction. You can set the reduction ratio using argument `reduction_ratio`. Parameter reduction of 50-80% seems to give good results, so we recommend 50% as a first choice and try a higher level. 

## Code used for our paper figures
under construction.

## Reference
will be presented
