import torch
import math
ts = torch.tensor

SAFETY_MULT = 1.001
BASE_PSI = .01
LOG_BASE_PSI = math.log(BASE_PSI)


def infoToM(Info,psi=None,ignore_head=0,debug=False):
    tlen = len(Info)


    if psi is None:
        psi = torch.ones(tlen) * BASE_PSI
    try:
        assert len(Info)==tlen
    except:
        print(Info.size(),tlen)
        raise
    M = []
    for i in range(ignore_head,tlen):
        ii = i-ignore_head
        lseterms = torch.stack([ts(0.),
                            -Info[i,i] + psi[ii],
                            -Info[i,i] + psi[ii] + #was: -abs(Info[i,i]) + psi[i] +
                                torch.sum(torch.abs(Info[i,:])) +
                                - torch.abs(Info[i,i])
                        ])
        if debug:
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
