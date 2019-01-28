import numpy as np
from random import shuffle

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
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1

      # dont update when correct cuz svm is max( 0, asldkfjlaksdjf) 
      # so grad of 0 is just 0.

      # Guessed incorrect class
      if margin > 0:
        # Update weights
        dW[:,j] += X[i]     # adjust weights for guessing class j
        dW[:,y[i]] -= X[i]  #

        loss += margin 

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  # We're doing stochastic gradient so /n after adding gradients of all images
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]


  # X (500, 3073)
  # W (3073,10)
  # Scores (500, 10)
  # Margins (500, 10)
   
  scores = np.matmul(X, W)
  correct_scores = scores[np.arange(len(scores)), y]
  margins = scores - correct_scores.reshape(-1,1) + 1

  # update margins for correct scores to 0
  margins[np.arange(len(scores)), y] = 0
  margins = np.maximum(margins, 0)

  loss = np.sum(margins) / num_train + (reg * np.sum(W * W)) 

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

  # dW (3073, 10)
    
  # Creates a 500,10 matrix of 0s
  weight_updates = np.zeros_like(scores) 
  
  # places that have mistakes... become 1
  weight_updates[margins > 0] = 1  

  # We are going to set each of the correct images locs per row
  # to be the sum of the column for that image... which is equal to the number
  # of times that image was incorrectly misclassified
  weight_updates[np.arange(num_train), y] -= np.sum(weight_updates, axis=1)  

  # X.T 3k , 500 * 500, 10
  # TODO Take a look at the matrix mul
  dW = np.matmul(X.T, weight_updates) / num_train
  dW += 2 * reg * W 
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return loss, dW

