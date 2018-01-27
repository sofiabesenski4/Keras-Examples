Amazon-Review-Classifier
Author: Thomas Besenski
Date Created: Jan 24th

The purpose of these experiments is to create a generate a predicted rating for an amazon food product,
given the review's text. 

I am exploring multiple different architectures to try and solve the same problem
The first:

One-Hot Encoding, with a Feed Forward Neural Network: Jan26
loss : mean squared error
evaluation metric : mean absolute error
Analysis: Through trying out a few different configurations of 2 hidden Dense layers,
          the trial with using a single 0.5 dropout layer in between the 2 hidden layers 
          yields a minimum mean average error around 0.105



          
         
