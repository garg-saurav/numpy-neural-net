# numpy-neural-net
Vectorised implementation of forward and backward propagation of different neural network layers in numpy

# Layers Implemented
- Fully Connected Layer
- Convolution Layer
- Max Pooling Layer
- Average Pooling Layer
- Flatten Layer

# Setting up
- Give execute permission to [download_data.sh](./datasets/download_data.sh) file using the following command:<br>
`chmod +x download_data.sh`
- Now run the file to download the datasets into the correct directory
- You can now run the main file to train the network for different datasets using the following command:<br>
`$ python3 main.py −−dataset <dataset> −−verbose −−seed <seed>`<br>
Here dataset is one of MNIST, CIFAR-10, XOR, circle.

# Network Architecture
- You can change the network architecture in [trainer.py](./trainer.py) file
- [Here](./report.pdf) is the report mentioning different architectures and obtained accuracies

# Description of files
- [layers.py](./layers.py): This file contains the classes corresponding to the layers <br>
  - FullyConnectedLayer: A simple feed forward layer having weights of size in nodes x out nodes and a bias parameter
  - ConvolutionLayer: This layer has filters of size (out depth, in depth, filter rows, filter cols). The sizes of input and output matrices are mentioned in the code. The forward pass involves a patchwise cross-correlation of the input with the filters
  - MaxPooling: This layer pools the values in a window of size (filter row,filter col) and replaces them by the maximum value in that window. The backward pass through this for each such window would be a sparse matrix having exactly 1 value equal to 1, multiplied by the incoming gradient
  - AvgPooling: Similar to MaxPooling this layer replaces the values in a window by the average of all values in that window
  - Flatten - This is a helper layer which converts a 3-D array into a 1-D array

- [nn.py](./nn.py): This file contains the class corresponding to the neural net and uses the functions defined in layers.py

- [trainer.py](./trainer.py): This file contains the code to train the network on multiple datasets
