
# THIS FILE CONTAINS AN ALGORITHM FOR FINDING THE APPROXIMATE MLE (Q)
# OF A MATRIX OF MULTINOMIAL PROBABILITIES (PI), SUBJECT TO CONSTRAINTS ON
# THE ROWS AND COLUMNS OF Q.

# IT ALSO CONTAINS CODE FOR TESTING THE ALGORITHM.

import numpy as np
np.random.seed(1234)
import torch

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
   return torch.sum(array, 0)

def rowsum(array):
   return torch.sum(array, 1)

# main function: returns [Q,i], where i is the number of iterations required
def optimize_Q(R,C,pi,v,d,tolerance=tolerance,maxiters=30):

   # Some auxiliary matrices
      # M is the matrix of linear constraints on Q (not counting the inequalities)
   M_top = torch.eye(C)
   M_bottom = torch.cat((torch.ones(1,C),torch.zeros(R-2,C)),0)
   for r in range(1,R):
      M_top = torch.cat((M_top,torch.eye(C)),1)
      bottom_new = torch.zeros(R-1,C)
      if r<R-1:
         bottom_new[r]=torch.ones(1,C)
      M_bottom = torch.cat((M_bottom, bottom_new),1)
   M = torch.vstack((M_top,M_bottom),0)

   # Matrix D = M^T*(M*M^T)^{-1}*M
      # D is the matrx of projection to orthogonal complement of ker M;
      # it helps us obtain the nearest Q that actually satisfies the linear constraints
   D = torch.inverse(torch.mm(M,M.T))
   D=torch.mm(M.T,D)
   D=torch.mm(D,M)

   # Other stuff we'll need
   v_d = torch.cat((v,d),1).reshape(R+C,1)
   ind = torch.mm(d.reshape(R,1),v.reshape(1,C))

  # OK, let's get started!
  # Start with a bad guess for beta and a very high error...
   beta = torch.ones(1,C)
   errorQ=torch.ones(R,C)

   # Iterate while the error (i.e. distance from Q to constraint space) is above the tolerance level
   i=0
   while (i<maxiters and torch.any(errorQ > tolerance) or torch.any(errorQ < -tolerance)):
      i=i+1

      #adjust alpha based on current beta
      pi_beta =  torch.mm(pi,torch.diag(beta.view(-1)))
      M_alpha = torch.cat((pi_beta.T,torch.diag(rowsum(pi_beta))),0)
      alpha, lu_ = torch.solve(torch.mm(M_alpha.T,v_d), torch.mm(M_alpha.T,M_alpha))

      #adjust beta based on current alpha
      alpha_pi = torch.mm(torch.diag(alpha.view(-1)),pi)
      M_beta = torch.cat((torch.diag(colsum(alpha_pi)),alpha_pi),0)
      beta, lu_ = torch.solve( torch.mm(M_beta.T,v_d), torch.mm(M_beta.T,M_beta))

      # figure out error
      Q = torch.mm(torch.diag(alpha.view(-1)), torch.mm(pi,torch.diag(beta.view(-1))))
      errorQ = torch.mm(D,(Q-ind).view(-1)).reshape(R,C)
      #print(f"Error in Q:\n{errorQ}")

   return [Q-errorQ,i]

def test_solver(numtests=numtests):
    ##################################################################
    # TESTING
    print(f"Testing optimize_Q ({numtests} tests): \nR={R}, C={C}, tolerance={tolerance}")
    print("==================================================")

    # Here's where we'll record the number of iterations we need in each test
    # and the worst error in Q (componentwise) that we come across
    num_iter = torch.zeros(100)
    worst_error=0

    # Do a bunch of tests!
    for n in range(numtests):

    # Create the problem together with its correct solution
       # Create pi (randomly)
       pi = torch.rand(R,C)
       rsum = rowsum(pi)
       for r in range(R):
          pi[r] = pi[r]/rsum[r]

       # Create alpha and beta (randomly) and trueQ
       true_alpha = torch.rand(R,1)
       true_beta = torch.rand(1,C)
       trueQ = torch.mm(torch.diag(true_alpha.view(-1)), torch.mm(pi,torch.diag(true_beta.view(-1))))
       true_alpha = true_alpha/torch.sum(trueQ)
       trueQ = trueQ/torch.sum(trueQ)
       #print(f"True Q:\n {trueQ}")

       # create "data" d and v
       v = colsum(trueQ)
       d = rowsum(trueQ)

    # Solve the problem
       [Q,i] = optimize_Q(R,C,pi,v,d,tolerance)
       if torch.min(Q)<0:
          print(f"Oh no! In test {n+1}, Q has some negative entries:")
          for r in range(R):
             for c in range(C):
                if Q[r][c]<0:
                   print(f"\t trueQ[{r}][{c}]={trueQ[r][c]}, \n\t     Q[{r}][{c}]={Q[r][c]}")

       num_iter[i]+=1
       worst_error_in_test=torch.max(torch.abs(trueQ-Q))
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
