import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    loss = reg_strength * np.sum(W**2)
    grad = reg_strength * 2 * W

    return loss, grad

def softmax(predictions):

    prediction = predictions.copy()
    len_pred = len(predictions.shape)

    if (len_pred == 1):
        e_prob = np.exp(prediction - np.max(prediction))
        probs = np.array(list(map(lambda x: x / np.sum(e_prob), e_prob)))
    else:
        pred = list(map(lambda x: x - np.max(x), prediction))
        e_prob = np.exp(pred)
        probs = np.array(list(map(lambda x: x / np.sum(x), e_prob)))

    return probs

def cross_entropy_loss(probs, target_index):

    len_probs = len(probs.shape)

    if (len_probs == 1):
        loss = -np.log(probs[target_index])
    else:
        batch_size = np.arange(target_index.shape[0])
        loss = np.sum(-np.log(probs[batch_size,target_index.flatten()])) / target_index.shape[0]

    return loss

def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    prediction = predictions.copy()
    len_pred = len(predictions.shape)
    len_target = 1
    probs = softmax(predictions)
    dprediction = probs
    loss = cross_entropy_loss(probs, target_index)

    if (len_pred == 1):
        dprediction[target_index] -= 1
    else:        
        batch_size = np.arange(target_index.shape[0])
        dprediction[batch_size, target_index.flatten()] -= 1
        len_target = target_index.shape[0]

    return loss, dprediction/len_target
    


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X = X
        
        return np.maximum(X, 0)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = np.where(self.X > 0, d_out, 0)
        
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}




class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        result = np.dot(X, self.W.value) + self.B.value
        self.X = X
        return result

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.B.grad += np.sum(d_out, axis = 0)
        self.W.grad += np.dot(self.X.T, d_out)
        d_input = np.dot(d_out, (self.W.value).T)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}


    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))
        self.X = None
        self.result = None
        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        pad = self.padding
        self.X = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

        out_height = width - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1
        self.result = np.zeros((batch_size, out_height, out_width,  self.out_channels))
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        self.X_pad = np.zeros((batch_size, height + 2 * self.padding, width + 2 * self.padding, self.in_channels))
        self.X_pad[:, self.padding: height + self.padding, self.padding: width + self.padding, :] = X
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                x_slice = self.X_pad[:, x: x + self.filter_size, y: y + self.filter_size, :, np.newaxis]
                self.result[:, x, y, :] = np.sum(x_slice * self.W.value[np.newaxis, :], axis=(1, 2, 3)) +  self.B.value
        
        return self.result


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

       
        dX = np.zeros_like(self.X)
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                grad = d_out[:, x, y, np.newaxis, np.newaxis, np.newaxis, :]
                dX[:, x: x + self.filter_size, y: y + self.filter_size, :] += np.sum(self.W.value * grad, axis=-1)
                self.W.grad += np.sum(grad * self.X[:, x: x + self.filter_size, y: y + self.filter_size, :, np.newaxis], axis=0)
       
        self.B.grad += np.sum(d_out, axis=(0, 1, 2))
         
        d_input = dX[:, self.padding: height -self.padding, self.padding: width -self.padding, :]
        
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        out_height = round((height - self.pool_size) / self.stride + 1)
        out_width = round((width - self.pool_size) / self.stride + 1)
        result = np.zeros((batch_size, out_height, out_width, channels))
        
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        
        for y in range(out_height):
            for x in range(out_width):
                x_slice = self.X[:, x: x + self.pool_size, y: y + self.pool_size, :]
                result[:, x, y, :] = x_slice.max(axis=(1,2))
        
        return result

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape
        result = np.zeros_like(self.X)
        
        for y in range(out_height):
            for x in range(out_width):
                x_slice = self.X[:, y:(y + self.pool_size), x:(x + self.pool_size), :]
                dX = (d_out[:, y, x, :])[:, np.newaxis, np.newaxis, :]
                result[:, y:(y + self.pool_size), x:(x + self.pool_size), :] += dX * (x_slice == x_slice.max(axis=(1, 2))[:, np.newaxis, np.newaxis, :])

        return result

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
