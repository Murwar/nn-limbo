import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        self.conv1 = ConvolutionalLayer(input_shape[2], conv1_channels, 3, 1)
        self.relu1 = ReLULayer()
        self.max_pl1 = MaxPoolingLayer(4, 4)
        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1)
        self.relu2 = ReLULayer()
        self.max_pl2 = MaxPoolingLayer(4, 4)
        self.flat = Flattener()
        self.fc = FullyConnectedLayer(4 * conv2_channels, n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        for i_param in self.params():
            param = self.params()[i_param]
            param.grad = np.zeros_like(param.grad)
        
        step1 = self.conv1.forward(X)
        step2 = self.relu1.forward(step1)
        step3 = self.max_pl1.forward(step2)
        step4 = self.conv2.forward(step3)
        step5 = self.relu2.forward(step4)
        step6 = self.max_pl2.forward(step5)
        step7 = self.flat.forward(step6)
        step8 = self.fc.forward(step7)
        loss, loss_grad = softmax_with_cross_entropy(step8, y)

        d8 = self.fc.backward(loss_grad)
        d7 = self.flat.backward(d8)
        d6 = self.max_pl2.backward(d7)
        d5 = self.relu2.backward(d6)
        d4 = self.conv2.backward(d5)
        d3 = self.max_pl1.backward(d4)
        d2 = self.relu1.backward(d3)
        d1 = self.conv1.backward(d2)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        step1 = self.conv1.forward(X)
        step2 = self.relu1.forward(step1)
        step3 = self.max_pl1.forward(step2)
        step4 = self.conv2.forward(step3)
        step5 = self.relu2.forward(step4)
        step6 = self.max_pl2.forward(step5)
        step7 = self.flat.forward(step6)
        step8 = self.fc.forward(step7)
        
        
        pred = step8.argmax(axis = 1)
        
        return pred

    def params(self):
        result =  {'conv1.W': self.conv1.W, 'conv1.B': self.conv1.B, 'conv2.W': self.conv2.W, 'conv2.B': self.conv2.B, 'fc.W': self.fc.W, 'fc.B': self.fc.B}

        # TODO: Aggregate all the params from all the layers
        # which have parameters

        return result
