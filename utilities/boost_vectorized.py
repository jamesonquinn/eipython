from __future__ import print_function
import torch
import math
#from .debugGizmos import *


S = torch.tensor([[2.,5.,0.],[5.,4.,1.],[0.,1.,3.]], requires_grad=True)
C = torch.tensor([[2.,1.,0.],[1.,4.,2.],[0.,2.,3.]], requires_grad=True)
M = torch.stack((S,C),0)

def colsum(batch_of_matrices):
    return torch.sum(batch_of_matrices, -2)

def rowsum(batch_of_matrices):
    return torch.sum(batch_of_matrices, -1)

def transpose(batch_of_matrices):
    return torch.transpose(batch_of_matrices,-1,-2)

def make_diag(batch_of_vectors):
    return torch.squeeze(torch.diag_embed(batch_of_vectors),-3)

def get_diag(batch_of_matrices):
    return torch.diagonal(batch_of_matrices,dim1 = -2, dim2 = -1)

def zeros_and_one(U,k):
   v = torch.zeros(U,k+1)
   v[:,k]=1
   return v

# the GMW81 algorthm, as described in Fang & O'Leary 2006
# inputs:
#    U is an integer
#    M is a U-by-n-by-n tensor (the program figures out n by itself)
#    psi is an n tensor
def _boost(U,M, psi):
   n=M.size()[1]
   psi1 = psi+torch.ones(n)

   M_abs = torch.abs(M)
   d_abs = get_diag(M_abs)
   # eta = max diagonal element
   eta = torch.max(d_abs, dim=1)[0]
   # zeta = max off-diagonal element
   zeta = torch.max((M_abs - make_diag(d_abs)).view(U,-1),1)[0]
   beta2 = torch.max(torch.stack((eta, zeta**2/math.sqrt(n^2-1)),1),1)[0]
   LT=[]
   D=[]
   A=M
   for k in range(n):
      if k==n-1:
         d=torch.max(torch.stack((torch.ones(U)*psi[k], torch.abs(A[:,0,0])*psi1[k]),1),1)[0]
         D.append(d)
         LT.append(zeros_and_one(U,k))
      else:
         c = A[:,0,1:n-k]
         cmax = torch.max(torch.abs(c),1)[0]
         d = torch.max(torch.stack((torch.ones(U)*psi[k], torch.abs(A[:,0,0])*psi1[k], cmax**2/beta2),1),1)[0]
         D.append(d)
         LT.append(torch.cat((zeros_and_one(U,k), c/d.unsqueeze(1)),1))
         A = A[:,1:n-k,1:n-k]-torch.matmul(c.unsqueeze(2),c.unsqueeze(1))/d.unsqueeze(1).unsqueeze(2)


   L = torch.stack(LT,2)
   Dvecs = torch.cat([d.unsqueeze(1) for d in D],1)
   D = make_diag(Dvecs)
   LDLT = torch.matmul(L,torch.matmul(D,transpose(L)))

   return L,Dvecs, LDLT
