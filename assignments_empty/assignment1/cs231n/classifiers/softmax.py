import numpy as np
from random import shuffle
#from past.builtins import xrange

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
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
  	fi = X[i,:].dot(W)
  	log_c = np.max(fi)
  	fi = fi - log_c
  	sum = 0.0
  	for j in range(num_class):
  		sum += np.exp(fi[j])
  	loss += (-fi[y[i]] + np.log(sum))
  	dW[:,y[i]] += (-X[i,:])
  	for j in range(num_class):
  		dW[:,j] += (1./sum * np.exp(fi[j]) * X[i,:])
  loss = loss / float(num_train) + 0.5 * reg * np.sum(W ** 2)
  dW = dW / num_train + reg * W
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
  num_train = X.shape[0]
  f = X.dot(W)
  #print (f.ndim)
  f_max = np.max(f,axis=1)
  f = f - f_max.reshape(-1,1)
  f = np.exp(f) / np.sum(np.exp(f),axis=1).reshape(-1,1)
  #loss = np.sum(np.log(np.sum(np.exp(f)),axis=1)) - np.sum(f[np.arange(num_train),y])
  #dW = 1. / np.sum(np.exp(f),axis=1)
  corr = f[np.arange(num_train),y]
  loss = -np.mean(np.log(corr / np.sum(f,axis=1).reshape(-1,1)))
  #p = np.exp(f) * 1./ np.sum(np.exp(f),axis=1).reshape(-1,1)
  p = f.copy()
  ind = np.zeros(p.shape)
  ind[np.arange(num_train),y] = 1
  dW = np.dot(X.T,(p-ind))
  dW /= float(num_train)
  ##############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  return loss, dW

