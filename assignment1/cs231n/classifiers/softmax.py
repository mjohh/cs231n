import numpy as np
from random import shuffle
from past.builtins import xrange

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
  #pass
  return softmax_loss_vectorized(W, X, y, reg)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

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
  #pass
  #attentio: the val of loss and dW should be correct both, because they affect
  #each other!!!!
  N, D = X.shape
  C = W.shape[1]
  scores = np.dot(X, W)                 #(N, C)
  maxs = np.max(scores, axis=1, keepdims=True)
  scores -= maxs
  exps = np.exp(scores)                 #(N, C) 
  sums = np.sum(exps, axis=1, keepdims=True)           #(N, C)
  P = exps / sums                       #(N, C)

  data_loss = -np.log(P[np.arange(N), y]).sum()
  #data_loss = (P[np.arange(N), y]).sum()
  loss = data_loss / N + 0.5 * reg * np.sum(W*W)

  P[np.arange(N), y] -= 1               #(N, C)
  dW = np.dot(X.T, P)                   #(D, C)
  dW = 1/N * dW + reg * W
  
  #loss = -scores[np.arange(N), y] + np.log(sums) #(N,1)
  #loss = 1/N * np.sum(loss) + 0.5*reg*np.sum(W*W)

  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

