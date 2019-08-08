
# THIS FILE CONTAINS AN ALGORITHM FOR FINDING THE APPROXIMATE MLE (Q)
# OF A MATRIX OF MULTINOMIAL PROBABILITIES (PI), SUBJECT TO CONSTRAINTS ON
# THE ROWS AND COLUMNS OF Q.

# IT ALSO CONTAINS CODE FOR TESTING THE ALGORITHM.

import numpy as np
np.random.seed(1234)

############################

# Global parameters set by user

R = 3
C = 5
tolerance = .001
numtests = 50
verbose=False

#############################

# auxiliary functions: row and column sum
def colsum(array):
   return np.sum(array, axis=0)

def rowsum(array):
   return np.sum(array, axis=1)

# main function: returns [Q,i], where i is the number of iterations required
#!/usr/bin/python
# -*- coding: utf-8 -*-

# main function: returns [Q,i], where i is the number of iterations required

SHRINK_FAC = .99

def optimize_Q_internal(
    R,
    C,
    pi,
    v,
    d,
    tolerance,
    ):

   # Some auxiliary matrices
      # M is the matrix of linear constraints on Q (not counting the inequalities)

    M_top = np.eye(C)
    if R > 2:
        M_bottom = np.vstack((np.ones((1, C)), np.zeros((R - 2, C))))
    else:

          # if R==2

        M_bottom = np.ones((1, C))
    for r in range(1, R):
        M_top = np.hstack((M_top, np.eye(C)))
        bottom_new = np.zeros((R - 1, C))
        if r < R - 1:
            bottom_new[r] = np.ones((1, C))
        M_bottom = np.hstack((M_bottom, bottom_new))
    M = np.vstack((M_top, M_bottom))

   # Matrix D = M^T*(M*M^T)^{-1}*M
      # D is the matrx of projection to orthogonal complement of ker M;
      # it helps us obtain the nearest Q that actually satisfies the linear constraints

    D = np.linalg.inv(np.matmul(M, M.T))
    D = np.matmul(M.T, D)
    D = np.matmul(D, M)

   # Other stuff we'll need

    v_d = np.hstack((v, d)).reshape(R + C, 1)
    ind = np.matmul(d.reshape(R, 1), v.reshape(1, C))

  # OK, let's get started!
  # Start with a bad guess for beta and a very high error...

    beta = np.ones((1, C))
    errorQ = np.ones((R, C))

   # Iterate while the error (i.e. distance from Q to constraint space) is above the tolerance level

    i = 0
    while np.any(errorQ > tolerance) or np.any(errorQ < -tolerance):
        i = i + 1

      # adjust alpha based on current beta

        pi_beta = np.matmul(pi, np.diagflat(beta))
        M_alpha = np.vstack((pi_beta.T, np.diag(rowsum(pi_beta))))
        alpha = np.linalg.solve(np.matmul(M_alpha.T, M_alpha),
                                np.matmul(M_alpha.T, v_d))

      # adjust beta based on current alpha

        alpha_pi = np.matmul(np.diagflat(alpha), pi)
        M_beta = np.vstack((np.diag(colsum(alpha_pi)), alpha_pi))
        beta = np.linalg.solve(np.matmul(M_beta.T, M_beta),
                               np.matmul(M_beta.T, v_d))

      # figure out error

        Q = np.matmul(np.diagflat(alpha), np.matmul(pi,
                      np.diagflat(beta)))
        errorQ = np.matmul(D, (Q - ind).flatten()).reshape(R, C)

      # print(f"Error in Q:\n{errorQ}")

    Q = Q - errorQ
    while torch.any(Q <= 0):
        min_index = Q.view(-1).argmax(0)
        indymin = ind.view(-1)[min_index]
        fac = SHRINK_FAC * indymin / (indymin - Q.view(-1)[min_index])
        Q = (Q - ind) * fac.detach() + ind

    return [Q, i]



def optimize_Q(R,C,pi,v,d,tolerance):
    Q, i = optimize_Q_internal(R,C,pi,v,d,tolerance)
    while torch.any(Q <= 0):
        print("Q has elements less than zero")
        Q = 1. + (Q - 1.) * shrink_fac
        i = i + 10

    while torch.any(Q > 1):
        print("Q has elements greater than 1")
        Q = Q * shrink_fac
        i = i + 10

    return [Q, i]



##################################################################
# TESTING
print(f"Testing optimize_Q ({numtests} tests): \nR={R}, C={C}, tolerance={tolerance}")
print("==================================================")

# Here's where we'll record the number of iterations we need in each test
# and the worst error in Q (componentwise) that we come across
num_iter = np.zeros(100)
worst_error=0

# Do a bunch of tests!
for n in range(numtests):

# Create the problem together with its correct solution
   # Create pi (randomly)
   pi = np.random.rand(R,C)
   rsum = rowsum(pi)
   for r in range(R):
      pi[r] = pi[r]/rsum[r]

   # Create alpha and beta (randomly) and trueQ
   true_alpha = np.random.rand(R,1)
   true_beta = np.random.rand(1,C)
   trueQ = np.matmul(np.diagflat(true_alpha), np.matmul(pi,np.diagflat(true_beta)))
   true_alpha = true_alpha/np.sum(trueQ)
   trueQ = trueQ/np.sum(trueQ)
   #print(f"True Q:\n {trueQ}")

   # create "data" d and v
   v = colsum(trueQ)
   d = rowsum(trueQ)

# Solve the problem
   [Q,i] = optimize_Q(R,C,pi,v,d,tolerance)
   if np.amin(Q)<0:
      print(f"Oh no! In test {n+1}, Q has some negative entries:")
      for r in range(R):
         for c in range(C):
            if Q[r][c]<0:
               print(f"\t trueQ[{r}][{c}]={trueQ[r][c]}, \n\t     Q[{r}][{c}]={Q[r][c]}")

   num_iter[i]+=1
   worst_error_in_test=np.amax(np.fabs(trueQ-Q))
   if worst_error_in_test > worst_error:
      worst_error = worst_error_in_test
   if verbose:
      print(f"Test {n+1}: {i} iterations, max error {worst_error_in_test}")

print(f"\nCumulative results for the {numtests} tests \n(R={R}, C={C}, tolerance={tolerance}):")
print("-------------------------------------------")
print(f"Worst error in entry of Q: {worst_error}")
print("\nTo get within tolerance, it took us:")
for i in range(100):
   if num_iter[i]>0:
      print(f"{i} iterations: {num_iter[i]} times")
