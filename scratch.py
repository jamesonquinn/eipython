print(mt(2,2,2,2)[2,2])
def get_directions(R, C, rnums, cnums): #note, not-voting is a candidate
    assert len(rnums)==R
    assert len(cnums)==C
    tot = sum(rnums)
    assert tot==sum(cnums)
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
            result.add_(raw[R,C] * basis[R,C]) #scalar times matrix
    return result

def inbasis_pos(R, C, raw, basis, start):
    if 0==torch.max(torch.abs(raw)):
        return(start)
    step1 = inbasis(R, C, raw, basis)
    ratio = torch.div(step1, start)
    closest = torch.min(ratio)
    return((step1 / -closest) * (1 - exp(closest)))
    
