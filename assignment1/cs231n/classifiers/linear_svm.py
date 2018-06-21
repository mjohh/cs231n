import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).
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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  h = 0.00001
  hW = W - h
  for i in xrange(num_train):
    scores = X[i].dot(W)
    hscores = X[i].dot(hW)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,y[i]] += -X[i]
        dW[:, j] += X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W         # (1)
   
  return loss, dW



def svm_loss_vectorized(W, X, y, reg):
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #pass

  num_train = X.shape[0]
  score_mat = X.dot(W)   #N x c
  correct_vals = score_mat[np.arange(num_train), y] #N x 1
  correct_vals = correct_vals.reshape([num_train, 1])
  margins = score_mat - correct_vals + 1.0
  margins[np.arange(num_train), y] = 0.0 
  margins[margins<0] = 0.0 
  loss = margins.sum() / num_train
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #pass
  margins[margins>0] = 1.0
  
  row_sum = np.sum(margins, axis=1)
  margins[np.arange(num_train), y] = -row_sum
       
  dW = np.dot(X.T, margins) / num_train + reg * W #D x C

       
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW