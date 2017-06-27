import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
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
  num_classes = W.shape[0]
  num_train = X.shape[1]

  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    scores -= np.max(scores)
    normalize_scores = np.exp(scores) / np.sum(np.exp(scores))
    loss += - np.log(normalize_scores[y[i]])

    dscore = np.reshape(normalize_scores, (num_classes, 1)) * X[:, i]
    dscore[y[i], :] -= X[:, i]
    dW += dscore

  loss = loss / num_train + 0.5 * reg * np.sum(W * W)
  dW = dW /num_train + reg * W
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
  num_classes = W.shape[0]
  num_train = X.shape[1]

  scores = W.dot(X)
  scores -= np.max(scores, axis=0)
  normalize_scores = np.exp(scores) / np.sum(np.exp(scores), axis=0)
  loss = np.sum(-np.log(normalize_scores[y, range(num_train)]))
  loss = loss / num_train + 0.5 * reg * np.sum(W * W)

  dscore = normalize_scores
  dscore[y, range(num_train)] -= 1
  dW = dscore.dot(X.T) / num_train + reg * W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
