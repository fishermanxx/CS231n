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
  dW = np.zeros_like(W)   #(D,C)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]    #N
  num_classes = W.shape[1]   #C
  for i in range(num_train):
      scores = X[i].dot(W)   #(1,C)  
      scores -= np.max(scores)
      prob = np.exp(scores[y[i]])/np.sum( np.exp(scores),axis = 0 )
      loss -= np.log(prob) 
      for j in range(num_classes):
          dW[:,j] += (np.exp(scores[j])/np.sum( np.exp(scores) ))*X[i].T 
      dW[:,y[i]] -= X[i,:].T
      
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  dW /= num_train
  loss /= num_train
  loss +=  0.5*reg*np.sum(W ** 2)
  dW += reg * W
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)   #(D,C)   X(N,D),W(D,C)
  num_train = X.shape[0]  #N

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)   #(N,C)
#  scores -= np.max(scores,axis = 0)
  
  exp_scores = np.exp(scores)   #(N,C)
  exp_scores_sum = np.sum(exp_scores,axis = 1)   #(N,1)
  prob = (exp_scores.T/exp_scores_sum.T).T    #(N,C)
  dW += (X.T).dot(prob)
  
  select_correct = np.zeros(prob.shape)   #(N,C)
  select_correct[range(num_train),y] = 1   #(N,C)
  dW -= (X.T).dot(select_correct)
  
  select_correct_prob = prob[range(num_train),y]   #(N,1)
  loss -= np.sum(np.log(select_correct_prob))
  
  dW /= num_train
  loss /= num_train
  loss +=  0.5*reg*np.sum(W ** 2)
  dW += reg * W  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

