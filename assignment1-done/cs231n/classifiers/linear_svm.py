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
  for i in xrange(num_train):
    scores = X[i].dot(W)  #1*10
    correct_class_score = scores[y[i]]    #1*1
    indicator = (scores - correct_class_score + 1)>0  #change  1*10
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:,j] += -np.sum(np.delete(indicator,j))*X[i].T   #change
        continue
      dW[:,j] += indicator[j]*X[i].T    #change
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
#        dW[:,j] += ((X[i].T))

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
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
  #W 3073*10 X 500*3073 y 500*1
  W = W.T   #10*3073
  X = X.T   #3073*500
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero   #10*3073
#  num_classes = W.shape[1]  #10
  num_train = X.shape[1]    #500
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = W.dot(X)   #10*500
  correct_class_score = scores[y,range(num_train)]  #1*500
  margins = np.maximum(scores - correct_class_score + 1,0)  #10*500
  margins[y,range(num_train)] = 0
          
  loss = np.sum(margins)/num_train
  loss += 0.5 * reg * np.sum(W ** 2)              
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
  select_wrong = np.zeros(margins.shape)    #10*500
  select_wrong[margins > 0] = 1  #10*500
  
  select_correct = np.zeros(margins.shape)    #10*500
  select_correct[y,range(num_train)] = np.sum(select_wrong,axis = 0)   

  dW = select_wrong.dot(X.T)    #10*3073
  dW -= select_correct.dot(X.T)
  dW /= num_train

  dW += reg*W  

  dW = dW.T            
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
