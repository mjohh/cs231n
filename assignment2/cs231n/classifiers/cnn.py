from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        #pass

        ###I move conv_param and pool_param from loss func to here, for W2 dim vals#### 
        # pass conv_param to the forward pass for the convolutional layer
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        self.conv_param = conv_param
        self.pool_param = pool_param

        C, H, W = input_dim
        F = num_filters
        HH = WW = filter_size
             
        H2 = 1 + (H + 2 * conv_param['pad'] - HH) // conv_param['stride']
        W2 = 1 + (W + 2 * conv_param['pad'] - WW) // conv_param['stride']
       
        H3 = 1 + (H2 - pool_param['pool_height']) // pool_param['stride']
        W3 = 1 + (H2 - pool_param['pool_width']) // pool_param['stride']

        self.params['W1'] = weight_scale * np.random.randn(F, C, HH, WW)
        self.params['b1'] = np.zeros(F,)

        self.params['W2'] = weight_scale * np.random.randn(F*H3*W3, hidden_dim)
        self.params['b2'] = np.float64(0)
        
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.float64(0)
        

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']


        #### I move conv_param and pool_param to __init__ func ####
        # pass conv_param to the forward pass for the convolutional layer
        #filter_size = W1.shape[2]
        #conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        #pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        conv_param = self.conv_param
        pool_param = self.pool_param

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        #pass
        #conv - relu - 2x2 max pool - affine - relu - affine - softmax
        out, conv_cache = conv_forward_naive(X, W1, b1, conv_param)
        out, relu_cache = relu_forward(out)
        out, pool_cache = max_pool_forward_naive(out, pool_param)
        out, affine_cache = affine_forward(out, W2, b2)
        out, relu_cache2 = relu_forward(out)
        out, affine_cache2 = affine_forward(out, W3, b3)
        scores = out

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        #pass
        loss, dout =  softmax_loss(scores, y)
        dout, grads['W3'], grads['b3'] = affine_backward(dout, affine_cache2)
        dout = relu_backward(dout, relu_cache2) 
        dout, grads['W2'], grads['b2'] = affine_backward(dout, affine_cache)
        dout = max_pool_backward_naive(dout, pool_cache)
        dout = relu_backward(dout, relu_cache)
        dout, grads['W1'], grads['b1'] = conv_backward_naive(dout, conv_cache)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
