import torch
import math
import json
#from .autoassign import autoassign
ts = torch.tensor

SAFETY_MULT = 1.001
BASE_PSI = .01
LOG_BASE_PSI = math.log(BASE_PSI)



def infoToM(Info,psi=None,ignore_head=0,debug=False):
    tlen = len(Info)
    #print("lens", tlen, len(psi))


    if psi is None:
        psi = torch.ones(tlen) * BASE_PSI
    try:
        assert len(Info)==tlen
    except:
        print(Info.size(),tlen)
        raise
    assert torch.all(psi>0)
    M = []
    for i in range(ignore_head,tlen):
        ii = i-ignore_head
        lseterms = torch.stack([ts(0.),
                            -Info[i,i] + psi[ii],
                            -Info[i,i] + psi[ii] + #was: -abs(Info[i,i]) + psi[i] +
                                torch.sum(torch.abs(Info[i,:])) +
                                - torch.abs(Info[i,i])
                        ])
        if debug is True:
            print("infoToM",i,torch.logsumexp(lseterms / psi[ii],0))
            print(lseterms)
            print(Info[i,])
        #print("m",lseterms)
        m = psi[ii] * torch.logsumexp(lseterms / psi[ii],0)
        if torch.isnan(m):
            #print("infoToM nan", lseterms)
            print([j
                for j in range(tlen) if torch.isnan(Info[i,j])])
            print(psi[ii])
        M.append(m)
    return torch.diag(SAFETY_MULT * torch.stack(M))

def nan_to_num(t,mynan=0.):
    if torch.all(torch.isfinite(t)):
        return t
    if len(t.size()) == 0:
        return torch.tensor(mynan)
    return torch.cat([nan_to_num(l).unsqueeze(0) for l in t],0)

def rescaledSDDD(Info,*args,**kwargs):
    #print("rescaledSDDD",Info.size())
    d1 = torch.diag(Info)
    if torch.any(d1 <= 0):#set to 1, without ruining grads... hack
        adjustment1 = torch.zeros_like(d1)
        adjustment2 = torch.zeros_like(d1)
        adjustment1[d1 <= 0] = d1[d1 <= 0]
        adjustment2[d1 <= 0] = 1.
        #print("d11",d1)
        d1 = d1 - adjustment1
        #print("d12",d1)
        d1 = d1 + adjustment2
        #print("d13",d1)

    d2 = torch.sqrt(d1)

    d3 = nan_to_num(d2,1.)
    #print("d3",d3,d2,d1,)
    rescaler = torch.mm(d3.unsqueeze(1),d3.unsqueeze(0))
    if torch.any(rescaler <= 0):
        print("Rescaler fail",rescaler)
        print(torch.diag(rescaler))
        for i in range(len(rescaler)):
            if rescaler[i,i]==0:
                print("row",i)
                print(Info[i,:])
                print(d1[i])
                print(d2[i])
                print(d3[i])
                print("badrow")
                print(torch.sqrt(torch.diag(Info))[i])
                print(adjustment1[i], adjustment2[i])
                break
        raise Exception("that's bad.")
    rescaled = Info / rescaler
    rescaled = nan_to_num(rescaled)
    delta = infoToM(rescaled,*args,**kwargs)
    #print("ddd",rescaled, delta, rescaled+delta)
    fixed = rescaled + delta
    if "debug" in kwargs and kwargs["debug"] is 0:
        print("rescaledSDDD",Info.size())
        print(Info[:4,:4])
        print((fixed * rescaler)[:4,:4])
    return(fixed * rescaler, delta*rescaler) #elementwise multiplication


def rescaledSDD(Info,*args,**kwargs):
    result, delta = rescaledSDDD(Info,*args,**kwargs)
    return(result)

def conditional_normal(full_mean, full_precision, n, first_draw, full_cov=None):
    if full_cov is None:
        full_cov = torch.inverse(full_precision)

    new_precision = full_precision[n:,n:]
    new_mean = full_mean[n:] + torch.mv(torch.mm(full_cov[n:,:n],
                                                torch.inverse(full_cov[:n,:n])),
                                        first_draw - full_mean[:n])
    return(new_mean,new_precision)


def getMarginalPrecision(fullprec,nhead,nsub,latentpsi,weight):
    result = fullprec[:nhead,:nhead]
    I = (len(fullprec) - nhead) // nsub
    lowerblocks = [None] * I
    for i in range(I):
        rawlower = fullprec[nhead+i*nsub:nhead+(i+1)*nsub,nhead+i*nsub:nhead+(i+1)*nsub] / weight
        lowerblocks[i] = lower = (weight * rescaledSDD(rawlower,latentpsi))
        result = result - torch.mm(torch.mm(fullprec[:nhead,nhead+i*nsub:nhead+(i+1)*nsub],
                        torch.inverse(lower)),
                        fullprec[nhead+i*nsub:nhead+(i+1)*nsub,:nhead])

    return(result,lowerblocks)

def getMpD(fullprec,nhead,nsub,globalpsi,latentpsi,weight):
    result1, lowerblocks = getMarginalPrecision(fullprec,nhead,nsub,latentpsi,weight)
    result, delta = rescaledSDDD(result1,globalpsi)
    return(result,delta,lowerblocks)

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
        print("calc_lls")
        self.lls = [weight*rescaledSDD(raw_ll/weight,self.psil)#,debug=i)
                    for (i,(raw_ll,weight)) in enumerate(zip(self.raw_lls, self.weights))]
        #print("example:")
        #print(self.raw_lls[0][:4,:4])
        #print(self.lls[0][:4,:4])
        self.llinvs = [torch.inverse(ll) for ll in self.lls]

    def marginal_gg(self):
        self.calc_lls()
        if self._mgg is not None:
            return self._mgg
        mgg = self.gg - torch.sum(torch.stack([torch.mm(torch.mm(gl,llinv),gl.t())
                                            for (gl,llinv,weight) in zip(self.gls,self.llinvs,self.weights)
                                            ]),0)
        #print("marginal")
        #print(self.gg[:4,:4])
        #print(mgg[:4,:4])
        self._mgg = rescaledSDD(mgg,self.psig,debug=0)
        return self._mgg

    def conditional_ll_mcov(self,i,g_delta,l_mean):
        assert self._mgg is not None
        assert self.lls is not None
        gl,llinv = (self.gls[i],self.llinvs[i])
        combined_gg = self._mgg - torch.mm(torch.mm(gl,llinv),gl.t())

        new_mean = l_mean - torch.mv(torch.mm(llinv,gl.t()),
                                            g_delta)
        return (new_mean,llinv)

    def to_jsonable(self):
        result = dict(
            G = self.G,
            L = self.L,
            mgg = self._mgg,
            lls = self.lls,
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
