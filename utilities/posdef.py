import torch
import math
ts = torch.tensor

SAFETY_MULT = 1.001
BASE_PSI = .01
LOG_BASE_PSI = math.log(BASE_PSI)


def infoToM(Info,psi=None,debug=False,strong=False):
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
        M.append( psi[i] * torch.logsumexp(lseterms / psi[i],0))
    return torch.diag(SAFETY_MULT * torch.stack(M))

def rescaledSDD(Info,*args,**kwargs):
    d = torch.sqrt(torch.diag(Info))
    if torch.any(d < 0):#set to 1, without ruining grads... hack
        adjustment = torch.zeros_like(d)
        adjustment[d < 0] = d[d < 0] -1.
        d = d - adjustment
    rescaler = torch.mm(d.unsqueeze(1),d.unsqueeze(0))
    rescaled = Info / rescaler
    fixed = rescaled + infoToM(rescaled,*args,**kwargs)
    return(fixed * rescaler)
