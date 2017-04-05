import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ResConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - SBN- relu - 2x2 max pool - conv - SBN - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[16,16], filter_size = 3,
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
    (C, H, W) = input_dim
    F1 = num_filters[0]
    F2 = num_filters[1]
    HH = filter_size
    WW = filter_size

    # padding
    P1 = (filter_size - 1)/2
    # stride
    S = 1
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
    H_out1 = 1 + (H - HH + 2*P1)/S
    W_out1 = 1 + (W - WW + 2*P1)/S
    H_out2 = 1 + (H_out1/2 - HH + 2*P1)/S
    W_out2 = 1 + (W_out1/2 - WW + 2*P1)/S
    P2 = 2
    S2 = 2
    HHH = 4
    WWW = 4
    self.params['W1'] = weight_scale * np.random.randn(F1,C,HH,WW)
    self.params['b1'] = np.zeros(F1)
    self.params['W2'] = weight_scale * np.random.randn(F2,F1,HH,WW)
    self.params['b2'] = np.zeros(F2)
    self.params['W2_2'] = weight_scale * np.random.randn(F2,C,HHH,WWW)
    self.params['b2_2'] = np.zeros(F2)
    self.params['W3'] = weight_scale * np.random.randn(H_out2*W_out2*F2/4,hidden_dim)
    self.params['b3'] = np.zeros(hidden_dim)
    self.params['W4'] = weight_scale * np.random.randn(hidden_dim,num_classes)
    self.params['b4'] = np.zeros(num_classes)
    self.params['gamma1'] = np.random.rand(F1)
    self.params['beta1'] = np.random.rand(F1)
    self.params['gamma2'] = np.random.rand(F2)
    self.params['beta2'] = np.random.rand(F2)
    self.sbn_params = [{'mode':'train'},{'mode':'train'}]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    
    mode = 'test' if y is None else 'train'
    for sbn_param in self.sbn_params:
        sbn_param[mode] = mode
    
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W2_2, b2_2 = self.params['W2_2'], self.params['b2_2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    reg = self.reg
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param1 = {'stride': 1, 'pad': (filter_size - 1) / 2}
    conv_param2 = {'stride': 2, 'pad': (W2_2.shape[2] - 1) / 2}
    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    sbn1 = self.sbn_params[0]
    sbn2 = self.sbn_params[1]
    gamma1 = self.params['gamma1']
    beta1 = self.params['beta1']
    gamma2 = self.params['gamma2']
    beta2 = self.params['beta2']
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out1, cache1 = conv_forward_fast(X,W1,b1,conv_param1)
    out2, cache2 = spatial_batchnorm_forward(out1,gamma1, beta1, sbn1)
    out3, cache3 = relu_forward(out2)
    out4, cache4 = max_pool_forward_fast(out3,pool_param)
    #print out4.shape
    #print W2.shape
    out5, cache5 = conv_forward_fast(out4,W2,b2,conv_param1)
    out6, cache6 = spatial_batchnorm_forward(out5,gamma2, beta2, sbn2)
    out7, cache7 = conv_forward_fast(X,W2_2,b2_2,conv_param2)
    out8 = out6 + out7
    out9, cache9 = relu_forward(out8)
    out10, cache10 = max_pool_forward_fast(out9,pool_param)
    out11, cache11 = affine_relu_forward(out10.reshape(out10.shape[0],-1),W3,b3)
    out12, cache12 = affine_forward(out11,W4,b4)
    scores = out12
    
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
    loss, dscores = softmax_loss(scores,y) 
    loss += 0.5*reg*(np.sum(W1**2)+np.sum(W2**2)+np.sum(W3**2) + np.sum(W4**2) + np.sum(W2_2**2))
    dout12, dW4, db4 = affine_backward(dscores,cache12)
    dout11, dW3, db3 = affine_relu_backward(dout12,cache11)
    dout10 = max_pool_backward_fast(dout11.reshape(*out10.shape),cache10)
    dout9 = relu_backward(dout10,cache9)
    dout6, grads['gamma2'],grads['beta2'] = spatial_batchnorm_backward(dout9,cache6)
    dout7,dW2_2,db2_2 = conv_backward_fast(dout9,cache7)
    #print dout5_1.shape
    dout5, dW2, db2 = conv_backward_fast(dout6,cache5)
    #print dout5.shape
    dout4 = max_pool_backward_fast(dout5,cache4)
    dout3 = relu_backward(dout4,cache3)
    dout2, grads['gamma1'],grads['beta1'] = spatial_batchnorm_backward(dout3,cache2)
    dout1, dW1, db1 = conv_backward_fast(dout2,cache1)
    grads['W4'] = dW4 + reg * W4
    grads['b4'] = db4
    grads['W3'] = dW3 + reg * W3
    grads['b3'] = db3
    grads['W2'] = dW2 + reg * W2
    grads['b2'] = db2
    grads['W2_2'] = dW2_2 + reg * W2_2
    grads['b2_2'] = db2_2
    grads['W1'] = dW1 + reg * W1
    grads['b1'] = db1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
