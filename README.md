## Sequential Learning Network with Residual Blocks: Incorporating Temporal Convolutional Information into Recurrent Neural Networks

### Overview

This is the code for SLN model. Temporal convolutional networks (TCNs) have shown remarkable performance in sequence modeling and surpassed recurrent neural networks (RNNs) in a number of tasks. However, performing exceptionally on extremely long sequences remains an obstacle due to the restrained receptive field of temporal convolutions and a lack of forgetting mechanism. Although RNNs can carry out state transmission down the full sequence length and latch information by means of a forgetting gate, the issue of information saturation and vanishing or exploding gradients that occur during back-propagation due to the effect of multiplicative accumulation, still persist. To benefit from both temporal convolutions and RNNs, we propose a neural architecture that merge temporal convolutional data into recurrent networks. The temporal convolutions are employed intermittently and fused into the hidden states of RNNs with the assistance of attention for providing long-term information. With this architecture, it is not needed for convolutional networks to cover the total length of the sequence, thus gradient and saturation issues in RNNs are ameliorated since convolutions are integrated into its cells and the state is updated with convolutional information. 

### Data

See `data_generator` in `utils.py`. You only need to download the data once. The default path
to store the data is at `./data/mnist`.

Original source of the data can be found [here](http://yann.lecun.com/exdb/mnist/).

### Note

- Because a TCN's receptive field depends on depth of the network and the filter size, we need
to make sure these the model we use can cover the sequence length 784. 

- While this is a sequence model task, we only use the very last output (i.e. at time T=784) for 
the eventual classification.
