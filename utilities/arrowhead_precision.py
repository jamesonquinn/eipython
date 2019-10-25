import torch
import math
import json
from .boost_vectorized import _boost, make_diag, get_diag, transpose
from .boost_sdd import rescaledSDD
from .debugGizmos import *

def boost_to_chol(Ms,psi,include_sym=False): #Ms is a tensor of matrices
    U = Ms.size()[0]
    L,Dvecs, LDLT = _boost(U,Ms,psi)
    Lchol = torch.matmul(L, make_diag(torch.sqrt(Dvecs)))
    dp("b_t_c", U,sizes(Ms,psi,Lchol))
    if include_sym:
        return (Lchol,LDLT)
    return Lchol


class ArrowheadPrecision:
    """

    Note: This class is valid in itself, but leaves the responsibility
    for dealing with duplicate (double-sampled) units with the code above.

    Currently, the multisite model cannot have double-sampling because it
    samples without replacement; and the EI model has no problems with
    double-sampling because it amortizes, so everything naturally works out
    correctly. However, if we ever had something that was
    subsampled-but-unamortized and had probability weighting, the current
    multisite code would likely break, and possibly do so invisibly.


    To use:

    >>> arrow = ArrowheadPrecision(3,2,torch.eye(3))
    >>> arrow.add_one_l(torch.ones(3,2) * .2,torch.eye(2))
    >>> arrow.add_one_l(torch.zeros(3,2),torch.eye(2)) #zeros on the off-diagonal; no impact on head
    >>> arrow.add_one_l(torch.zeros(3,2),torch.eye(2),100) #as above, but big weight — will SDD funny
    >>> arrow.setpsis(torch.ones(3) * .01,torch.ones(2)*.01) #must be roughly this order or smaller here so that sdd has no impact
    >>> arrow.marginal_gg()
    tensor([[ 0.9200, -0.0800, -0.0800],
            [-0.0800,  0.9200, -0.0800],
            [-0.0800, -0.0800,  0.9200]])
    >>> mu, sig = arrow.conditional_ll_mcov(1,torch.ones(3),torch.ones(2)*1000)
    >>> mu
    tensor([1000., 1000.])
    >>> sig
    tensor([[1., -0.],
            [0., 1.]])
    >>> mu, sig = arrow.conditional_ll_mcov(0,torch.ones(3),torch.ones(2)*1000)
    >>> mu
    tensor([999.4000, 999.4000])
    >>> sig
    tensor([[1., -0.],
            [0., 1.]])
    """
    #@autoassign
    def __init__(self, G, L, gg):
        assert list(gg.size()) == [G, G]
        self.G = G
        self.L = L
        self.gg = gg
        self.gls = []
        self.raw_lls = []
        self.weights = []
        self.lls = None
        self._mgg = None

    #@autoassign
    def setpsis(self, psig, psil):
        self.psig = psig
        self.psil = psil
        assert self._mgg == None #should not change after calculating this

    def add_one_l(self, gl, ll, weight = 1.):
        assert list(gl.size()) == [self.G,self.L]
        assert list(ll.size()) == [self.L,self.L]
        self.add_ls([gl],[ll],[weight])

    def add_ls(self, gls, lls, weights):
        assert self._mgg == None #should not add any more after calculating this
        assert len(gls) == len(lls)
        self.gls.extend(gls)
        self.raw_lls.extend(lls)
        self.weights.extend(weights)

    def calc_lls(self):
        if self.lls is not None:
            return
        dp("calc_lls",sizes(self.gg,self.gls[0],self.raw_lls[0],self.psig,self.psil))
        self.U = len(self.weights)
        self.vecweights = torch.stack([torch.tensor(w) for w in self.weights]).view(self.U,1,1)
        self.vecraw_lls = torch.stack(self.raw_lls)
        chol_lls,self.lls = boost_to_chol(self.vecraw_lls/self.vecweights,self.psil, include_sym=True)
        self.chol_lls = self.vecweights*chol_lls
        #print("example:")
        #print(self.raw_lls[0][:4,:4])
        #print(self.lls[0][:4,:4])

        #dp("chol_lls",self.chol_lls.size())
        self.llinvs = torch.inverse(self.lls)

    def marginal_gg_cov(self):
        if hasattr(self,"gg_cov"):
            return self.gg_cov
        self.calc_lls()
        if self._mgg is not None:
            return self._mgg
        self.vecgls = torch.stack(self.gls)
        inner = torch.matmul(self.vecgls,self.llinvs)
        transp = transpose(self.vecgls)
        dp("inner si",sizes(self.gg,inner,self.vecgls,transp))
        mgg = self.gg - torch.sum(torch.matmul(inner,transp)
                                            ,0)
        #print("marginal")
        #print(self.gg[:4,:4])
        #print(mgg[:4,:4])
        dp("btc 2", mgg.unsqueeze(0).size(), self.psig.size())
        self._mgg_chol, self._mgg = boost_to_chol(mgg.unsqueeze(0),self.psig, include_sym=True)
        #self.gg_cov = torch.cholesky_inverse(self._mgg_chol)
        self.gg_cov = torch.inverse(self._mgg)
        return self.gg_cov

    def conditional_ll_mcov(self,i,g_delta,l_mean):
        assert self._mgg_chol is not None
        assert self.chol_lls is not None
        gl,llinv = (self.gls[i],self.llinvs[i])
        #combined_gg = self._mgg + torch.mm(torch.mm(gl,llinv),gl.t())

        new_mean = l_mean - torch.mv(torch.mm(llinv,gl.t()),
                                            g_delta)
        return (new_mean,llinv)

    def to_jsonable(self):
        result = dict(
            G = self.G,
            L = self.L,
            gg_raw = self.gg,
            raw_lls = self.raw_lls,
            gg_cov = self.gg_cov,
            chol_lls = self.chol_lls,
            llinvs = self.llinvs,
            gls = self.gls,
            weights = self.weights
        )
        return result

class FittedGuideEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ArrowheadPrecision):
            return obj.to_jsonable()
        if torch.is_tensor(obj):
            return obj.tolist()
        return json.JSONEncoder.default(self,obj)
