# README #


### The All Convolutional Net ###

* This is the implementation of The All Convolutional Net Paper where all layers of the neural net are conv layers no pool layer added. We compare three implementations of a convolutional neural net. 1) Baseline, regular conv->pool->conv->pool... architecture 2) Removing all pool layers from (1) but with increased stride in each of the conv layers right before the pool layers. 2) Replace all pool layers with a conv layer with increased stride. 

(http://arxiv.org/abs/1412.6806)
Version 0.1

_Ciphar-10 and Ciphar-100_ image datasets are used

### Dependencies ###

``` pip -r install requirements.txt``` 

will install the dependancies listed in the requirements.txt file

### Contribution guidelines ###

To run experiments, run the experiments.notebook.ipynb notebook and follow the examples to train each type of model

### Contributors ###

Elias Hussen and Dan Rosenthal
