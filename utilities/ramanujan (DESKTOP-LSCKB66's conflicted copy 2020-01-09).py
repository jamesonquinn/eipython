import torch

ONE_THIRTIETH = torch.tensor(1/30)
def lfac_ramanujan(t): #like lgamma(t+1), ignoring constant .5 ln pi
    return t*torch.log(t)-t+torch.log(8*t**3+4*t**2+t+ONE_THIRTIETH)/6
def lfac_stirling(t): #like lgamma(t+1), ignoring constant .5 ln pi
    return t*torch.log(t)

def lgamma_difble(t):
    return lfac_ramanujan(t-1)
