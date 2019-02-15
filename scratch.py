print(mt(2,2,2,2)[1,1])
def approx_eq(a,b):
    print(torch.lt(torch.abs(torch.add(a, -b)), 1e-8))
    print("So:",torch.all(torch.lt(torch.abs(torch.add(a, -b)), 1e-8)))
    print("So2:",torch.all(torch.lt(torch.abs(torch.add(a, -b)), 1e-8))==
            torch.all(torch.lt(zs(1),1)))
    return(torch.all(torch.lt(torch.abs(torch.add(a, -b)), 1e-8)))

def get_directions(R, C, rnums, cnums): #note, not-voting is a candidate
    assert len(rnums)==R
    assert len(cnums)==C
    tot = sum(rnums)
    print("sums",tot,sum(cnums))
    assert approx_eq(tot,sum(cnums))
    indep = ts([[(rnum * cnum / tot) for cnum in cnums] for rnum in rnums])
    basis = mt(R,C,R,C) #the first R,C tells "which basis vector"; the second R,C hold the elements of that "basis vector"
    for rbas in range(R):
        for cbas in range(C):
            cor = basis[rbas,cbas,rbas,cbas] = indep[rbas,cbas]
            remainder = tot - cor
            fac = cor / remainder
            for r in range(R):
                if r==rbas:
                    continue
                basis[rbas,cbas,r,cbas] = -fac * indep[r,cbas]
            for c in range(C):
                if c==cbas:
                    continue
                basis[rbas,cbas,rbas,c] = -fac * indep[rbas,c]
                for r in range(R):
                    if r==rbas:
                        continue
                    basis[rbas,cbas,r,c] = basis[rbas,cbas,r,cbas] * basis[rbas,cbas,rbas,c] / remainder
    return(indep,basis)

def inbasis(R, C, raw, basis):
    result = zs(R, C)
    for r in range(R):
        for c in range(C):
            result.add_(raw[r,c] * basis[r,c]) #scalar times matrix
    return result

def polytopize(R, C, raw, basis, start):
    if 0==torch.max(torch.abs(raw)):
        return(start)
    step1 = inbasis(R, C, raw, basis)
    ratio = torch.div(step1, start)
    closest = torch.min(ratio)
    return((step1 / -closest) * (1 - exp(closest)))

def test_funs(R, C, innerReps=2, outerReps=2):
    for i in range(outerReps):
        rnums = pyro.distributions.Exponential(1.).sample(torch.Size([R]))
        cnums = pyro.distributions.Exponential(1.).sample(torch.Size([C]))
        cnums = cnums / sum(cnums) * sum(rnums)
        indep, basis = get_directions(R,C,rnums,cnums)
        for j in range(innerReps):
            loc = pyro.distributions.Normal(0.,4.).sample(torch.Size([R,C]))
            polytopedLoc = polytopize(R,C,loc,basis,indep)
            assert rnums == sum(polytopedLoc, dim=1).view(R)
            assert cnums == sum(polytopedLoc, dim=0).view(C)

test_funs(2,2)
test_funs(5,2)
test_funs(2,4)
test_funs(6,8)
