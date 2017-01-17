# README #


### The All Convolutional Net ###

This is the implementation of The All Convolutional Net Paper where all layers of the neural net are conv layers no pool layer added. We compare three implementations of a convolutional neural net. 

1. Baseline, regular conv->pool->conv->pool... architecture 
2. Removing all pool layers from (1) but with increased stride in each of the conv layers right before the pool layers. 
3. Replace all pool layers with a conv layer with increased stride. 

#### Version:

Version 0.1

#### Data Used

_Ciphar-10 and Ciphar-100_ image datasets are used

### Dependencies ###

To install the required libraries, run the following from the terminal

``` pip -r install requirements.txt``` 

To run experiments, use the experiments.notebook.ipynb notebook and follow the examples to train and compare the all convolutional model with increased stride against the standard convolutional architectures with pool layers.

### Contributors ###

Elias Hussen and Dan Rosenthal contributed to the project
