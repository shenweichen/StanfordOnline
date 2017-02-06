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
  dW_each = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  n_train,n_class = X.shape[0],W.shape[1]
  f = np.dot(X,W) 
  f_max = np.max(f,axis = 1).reshape(n_train,1)
  f = f - f_max
  for i in range(n_train):
    score = np.exp(f[i,y[i]])/np.sum(np.exp(f[i]))
    loss += -np.log(score)
    for k in range(n_class):
    	if k == y[i]:
    		dW_each[:,k] = X[i].T * (score - 1)
    	else:
    		score_k = np.exp(f[i,k])/np.sum(np.exp(f[i]))
    		dW_each[:,k] = X[i].T * score_k
    dW += dW_each
  loss /= n_train
  loss += 0.5 * reg * np.sum(np.square(W))
  dW /= n_train
  dW += reg * W 
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
  pass
  n_train,n_feature = X.shape
  f = np.dot(X,W)
  f_max = np.max(f,axis=1).reshape(n_train,1)
  f = f - f_max
  prob = np.exp(f)/np.sum(np.exp(f),axis=1,keepdims = True)
  y_true = np.zeros_like(prob)
  y_true[range(n_train),y] = 1.0
  loss += -np.sum(y_true * np.log(prob))/n_train + 0.5 * reg * np.sum(W*W)
  dW += -np.dot(X.T, y_true - prob) / n_train + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

