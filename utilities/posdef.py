import torch
import math
ts = torch.tensor

SAFETY_MULT = 1.001
BASE_PSI = .01
LOG_BASE_PSI = math.log(BASE_PSI)


def infoToM(Info,psi=None,debug=False,strong=True):
    tlen = len(Info)
    if psi is None:
        psi = torch.ones(tlen) * BASE_PSI
    try:
        assert len(Info)==tlen
    except:
        print(Info.size(),tlen)
        raise
    M = []
    for i in range(tlen):
        if strong:
            lseterms = torch.stack([ts(0.),
                                -Info[i,i] + psi[i],
                                -Info[i,i] + psi[i] + #was: -abs(Info[i,i]) + psi[i] +
                                    torch.sum(torch.stack([abs(Info[i,j])
                                        for j in range(tlen) if j != i]))])
        else:
            lseterms = torch.stack([ts(0.),
                                -Info[i,i] + psi[i],
                                -abs(Info[i,i]) + psi[i] +
                                    torch.sum(torch.stack([abs(Info[i,j])
                                        for j in range(tlen) if j != i]))])
        if debug:
            print("infoToM",i,torch.logsumexp(lseterms / psi[i],0))
            print(lseterms)
            print(Info[i,])
        m = psi[i] * torch.logsumexp(lseterms / psi[i],0)
        if torch.isnan(m):
            print("infoToM nan", lseterms)
            print([j
                for j in range(tlen) if torch.isnan(Info[i,j])])
            print(psi[i])
        M.append(m)
    return torch.diag(SAFETY_MULT * torch.stack(M))

def nan_to_num(t,mynan=0.):
    if torch.all(torch.isfinite(t)):
        return t
    if len(t.size()) == 0:
        return torch.tensor(mynan)
    return torch.cat([nan_to_num(l).unsqueeze(0) for l in t],0)

def rescaledSDD(Info,*args,**kwargs):
    d1 = torch.diag(Info)
    if torch.any(d1 <= 0):#set to 1, without ruining grads... hack
        adjustment = torch.zeros_like(d1)
        adjustment[d1 <= 0] = d1[d1 <= 0] -1.
        d1 = d1 - adjustment

    d2 = torch.sqrt(d1)

    d3 = nan_to_num(d2,1.)
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
                print(adjustment[i])
                break
        raise Exception("that's bad.")
    rescaled = Info / rescaler
    rescaled = nan_to_num(rescaled)
    fixed = rescaled + infoToM(rescaled,*args,**kwargs)
    return(fixed * rescaler)


def conditional_normal(full_mean, full_precision, n, first_draw, full_cov=None):
    if full_cov is None:
        full_cov = torch.inverse(full_precision)

    new_precision = full_precision[n:,n:]
    new_mean = full_mean[n:] + torch.mv(torch.mm(full_cov[n:,:n],
                                                torch.inverse(full_cov[:n,:n])),
                                        first_draw - full_mean[:n])
    return(new_mean,new_precision)
