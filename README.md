# volterra-net
VolterraNet implementation and Spherical MNIST experiment. 

Please cite the following paper if you use this code: 

Banerjee, M., Chakraborty, R., Bouza, J., & Vemuri, B. C. (2020). VolterraNet: A higher order convolutional network with group equivariance for homogeneous manifolds. IEEE Transactions on Pattern Analysis and Machine Intelligence.

## Requirements

* Pytorch (0.4 <= )
* Tensorflow
* [s2cnn](https://github.com/jonas-koehler/s2cnn)

## Running Spherical MNIST

Please download the [MNIST dataset] (http://yann.lecun.com/exdb/mnist/) and place it in a folder named "mnist_data" in the project directory.
Next, run the gendata.py script to generate the Spherical MNIST dataset. Now the seperate models can be trained by calling 
the corresponding file. 
