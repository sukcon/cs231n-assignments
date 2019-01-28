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

  num_classes = W.shape[1]
  num_train, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss = 0

  for i in range(num_train):
    # calc all the scores
    scores = np.dot(X[i], W)

    # subtract the max for numerical stability 
    scores -= np.max(scores)
    
    # do exp(score)
    exp_scores = np.exp(scores)
    
    denominator = np.sum(exp_scores)
    # Cross entropy loss
    loss_i = -np.log(exp_scores[y[i]] / denominator) 
    loss += loss_i
              
    # Update weights each step, because we're always driving p(y | X, 0) to 1
    for j in range(num_classes):
      if j != y[i]:
        dW[:, j] += (exp_scores[j] / denominator) * X[i]
    dW[:,y[i]] += -X[i] + (exp_scores[y[i]] / denominator) * X[i]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  dW /= num_train

  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # X (N, D)
  # W (D, C)
  # y (N, )

  # For each training image
  # Compute the scores
  # Exponentiate all the scores (substracting max purposefully omitted)
  exp_scores = np.exp(np.matmul(X, W)) # (N, C)

  # compute sum of exponentiated scores along each image
  sum_exp_scores = np.sum(exp_scores, axis=-1)

  # compute loss as exponentiated score of correct image / denominator
  # tally up loss
  loss = np.sum(-np.log(exp_scores[np.arange(num_train), y] / sum_exp_scores))
     # Recall -log(1) = 0
     # and -log(.001) = some big boy pos number
  
  # compute dWeights
  # X is (N, D)
  # RHS is #(N, C)
  dW = np.matmul(X.T, (exp_scores / sum_exp_scores.reshape(-1,1)))  # 
  # 
  # TODO VECTORIZE THIS
  for i in range(num_train):
      dW[:, y[i]] -= X[i]


  # average out loss
  loss /= num_train
  # average out dWeights
  dW /= num_train

  # add regularization to loss
  loss += reg * np.sum(W * W)

  # add derviative of regulariztion to dWeights
  dW += 2 * reg * W



  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

