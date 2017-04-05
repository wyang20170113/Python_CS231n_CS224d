import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = X.dot(W)
  score = score - np.max(score,1)[:,np.newaxis]
  score = np.exp(score)
  score_sum = np.sum(score,1)
  score = score/score_sum[:,np.newaxis]
  num_train = X.shape[0]
  #print score[1,:].shape
  #print X[1,:].shape
  for i in range(num_train):
    loss += -np.log(score[i,y[i]])
    dW += X[i,:][:,np.newaxis].dot(score[i,:][np.newaxis,:])
    dW[:,y[i]] -= X[i,:]
  dW = dW / num_train
  dW = dW + reg * W
  loss =loss/num_train +  reg * np.sum(W**2)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  
  
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = X.dot(W)
  score = score - np.max(score,1)[:,np.newaxis]
  score = np.exp(score)
  score_sum = np.sum(score,1)
  score = score/score_sum[:,np.newaxis]
  num_train = X.shape[0]
  #print score[np.arange(num_train),y].shape
  loss += np.sum(-np.log(score[np.arange(num_train),y]))/num_train + 0.5*reg*np.sum(W**2)
  score[np.arange(num_train),y] -= 1
  dW = dW + X.T.dot(score)/num_train + reg*W
  #print score[np.arange(num_train),y].shape
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

