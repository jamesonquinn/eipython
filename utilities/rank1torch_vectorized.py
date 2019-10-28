
from utilities.debugGizmos import *
# THIS FILE CONTAINS A BATCH ALGORITHM FOR FINDING THE APPROXIMATE MLE (Q)
# OF A MATRIX OF MULTINOMIAL PROBABILITIES (PI), SUBJECT TO CONSTRAINTS ON
# THE ROWS AND COLUMNS OF Q.

# IT ALSO CONTAINS CODE FOR TESTING THE ALGORITHM.

import numpy as np
np.random.seed(1234)
import torch

#print("torch version",torch.__version__)
#print("torch.solve",torch.solve)

############################

# Global parameters set by user

R = 3
C = 2
U = 7

shrinkage = torch.tensor(.99)
tolerance = .001  #default
numtests = 50     #default
maxiters = 3      #default

#############################

# auxiliary functions for batched matrices:
# row sum, column sum, transpose, and make_diagonal-matrix-out-of-vector
def colsum(batch_of_matrices):
    return torch.sum(batch_of_matrices, -2)

def rowsum(batch_of_matrices):
    return torch.sum(batch_of_matrices, -1)

def transpose(batch_of_matrices):
    return torch.transpose(batch_of_matrices,-1,-2)

def make_diag(batch_of_vectors):
    return torch.squeeze(torch.diag_embed(batch_of_vectors),-3)



# main function: returns [Q,i], where i is the number of iterations required
# Input dimensions:
#   U, R, C, maxiters, tolerance are non-tensor scalars
#   pi is U-by-R-by-C; in [0,1], rows add to 1
#   v is U-by-C; adds up to 1 along C
#   d is U-by-R; adds up to 1 along R
def optimize_Q(U,R,C,pi,v,d,
            shrinkage = shrinkage,tolerance=tolerance,maxiters=maxiters,
            M=None, D=None, v_d=None, ind=None, init_beta_errorQ=None):

    if M is not None:
        # Some auxiliary matrices
            # M is the matrix of linear constraints on Q (not counting the inequalities)
            # M is (R+C-1)-by-RC
        M_top = torch.eye(C)
        if R>2:
          M_bottom = torch.cat((torch.ones(1,C),torch.zeros(R-2,C)),0)
        else: #i.e. if R=2
          M_bottom = torch.ones(1,C)
        for r in range(1,R):
            M_top = torch.cat((M_top,torch.eye(C)),1)
            bottom_new = torch.zeros(R-1,C)
            if r<R-1:
                bottom_new[r]=torch.ones(1,C)
            M_bottom = torch.cat((M_bottom, bottom_new),1)
        M = torch.cat((M_top,M_bottom),0)

    if D is not None:
        # Matrix D = M^T*(M*M^T)^{-1}*M
            # D is the matrx of projection to orthogonal complement of ker M;
            # it helps us obtain the nearest Q that actually satisfies the linear constraints
        D = torch.inverse(torch.matmul(M,transpose(M)))
        D=torch.matmul(transpose(M),torch.matmul(D,M))


    # Other stuff we'll need
    if (v_d is not None) or (ind is not None):
        v_d = torch.unsqueeze(torch.cat((v,d),-1),-1)
        ind = torch.matmul(torch.unsqueeze(d,-1),torch.unsqueeze(v,-2))

    # OK, let's get started!
    # Start with a bad guess for beta and a very high error...
    beta,errorQ = (torch.ones(U,C), torch.ones(U,R,C)) if init_beta_errorQ is None else init_beta_errorQ

    # Iterate while the error (i.e. distance from Q to constraint space) is above the tolerance level
    i=0
#   while (i<maxiters and (torch.any(errorQ > tolerance) or torch.any(errorQ < -tolerance))):
    while (i<maxiters):
        i=i+1

        #adjust alpha based on current beta
        pi_beta =  torch.matmul(pi,make_diag(beta))
        M_beta = torch.cat((transpose(pi_beta),make_diag(rowsum(pi_beta))),-2)
        dp("types?",[a.dtype for a in [M_beta,v_d]])
        alpha, junk = torch.solve(torch.matmul(transpose(M_beta),v_d), torch.matmul(transpose(M_beta),M_beta))
        alpha = torch.squeeze(alpha,-1)

        #adjust beta based on current alpha
        alpha_pi = torch.matmul(make_diag(alpha),pi)
        M_alpha = torch.cat((make_diag(colsum(alpha_pi)),alpha_pi),-2)
        beta, junk = torch.solve(torch.matmul(transpose(M_alpha),v_d), torch.matmul(transpose(M_alpha),M_alpha))
        beta = torch.squeeze(beta,-1)

        # figure out error (how far Q lies from ker M, the hyperplane we want it to lie on)
        Q = torch.matmul(make_diag(alpha), torch.matmul(pi,make_diag(beta)))
        errorQ = torch.matmul(D,(Q-ind).view(U,R*C,1)).view(U,R,C)
        #print(f"Error in Q:\n{errorQ}")

    # Check for nan's
    if torch.any(torch.isnan(Q-errorQ)):
        print("optimize_Q error: nan")
        print(pi_beta,M_beta,alpha,alpha_pi)
        print("2optimize_Q error: nan")
        print(M_alpha,beta,Q,errorQ)
        raise Exception("optimize_Q error: nan")


    Q = Q - errorQ

    # Deal with possible negative entries in Q:
    Qrc = Q.view(U,R*C)
    # vector of length U whose u-th entry is 1 is Q[u] has all positive entries, 0 otherwise
    Q_ok = (Qrc>0).all(1).type_as(Q)
    if torch.equal(Q_ok,torch.ones(U)) ==False:
        # print(Qrc)
        # print(Q_ok)
        # convert ind to U-by-RC tensor
        ind_rc = ind.view(U,R*C)
        # Qmin and Qmin_ind are U-tensors, u-th entry is min of Qrc[u] and its index
        Qrc_min, Qrc_minind = Qrc.min(1)
        # ind_min is a U-tensor, u-th entry is entry of ind_rc[u] at index of min(Qrc[u])
        ind_min = torch.gather(ind_rc,1,Qrc_minind.unsqueeze(1)).squeeze(1)
        # fac is shrink factor (1 is Q[u] is all positive)
        fac = torch.max(Q_ok,shrinkage) * ind_min / (ind_min - torch.min(Qrc_min,torch.zeros(U)))
        # print(fac)
        fac = fac.unsqueeze(1).unsqueeze(2)
        Q = (Q - ind) * fac.detach() + ind

    return [Q, i]

def optimize_Q_objectly(pi,data,**kwargs):
    return optimize_Q(data.U,data.R,data.C,pi,data.nvs,data.nns,
                M=data.M, D=data.D, v_d=data.v_d, ind=data.nind, init_beta_errorQ=data.init_beta_errorQ,
                **kwargs)

def test_solver(numtests=numtests, maxiters = maxiters, verbose=False):
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
        pi = torch.rand(U,R,C)
        rsum = torch.unsqueeze(rowsum(pi),-1)
        pi = pi/rsum

        # Create alpha and beta (randomly) and trueQ
        true_alpha = torch.rand(U,R)
        true_beta = torch.rand(U,C)
        trueQ = torch.matmul(make_diag(true_alpha), torch.matmul(pi,make_diag(true_beta)))
        sumQ = torch.sum(trueQ,dim=(-1,-2), keepdim = True)
        true_alpha = true_alpha/torch.squeeze(sumQ,-1)
        trueQ = trueQ/sumQ
        #print(f"True Q:\n {trueQ}")

        # create "data" d and v
        v = colsum(trueQ)
        d = rowsum(trueQ)

    # Solve the problem
        [Q,i] = optimize_Q(U,R,C,pi,v,d,shrinkage,tolerance,maxiters)
        if torch.min(Q)<0:
            print(f"Oh no! In test {n+1}, Q has some negative entries:")
            for u in range(U):
                for r in range(R):
                    for c in range(C):
                        if Q[u][r][c]<0:
                            print(f"\t trueQ[{u}][{r}][{c}]={trueQ[u][r][c]}, \n\t     Q[{u}][{r}][{c}]={Q[u][r][c]}")

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
            print(f"{i:03} iterations: {'*'*int(num_iter[i])} {num_iter[i]} times")
