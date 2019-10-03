# pylint: disable=E1101, C0103, C0111
'''
Computes the Hessian
'''
import torch
from hessian.gradient import gradient
from .posdef import *

def mygradient(output, inputs, allow_unused=False):
    '''
    Compute the gradient of `output` with respect to `inputs`
    '''
    assert output.ndimension() == 0

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)


    fullgrad = []
    for i, inp in enumerate(inputs):
        #print("myhessian",i,inp,output)
        [grad] = torch.autograd.grad(output, inp, create_graph=True, retain_graph=True, allow_unused=allow_unused)
        grad = torch.zeros_like(inp) if grad is None else grad

        fullgrad.append(grad.contiguous().view(-1))

    return torch.cat(fullgrad,0)


def hessian(output, inputs, out=None, allow_unused=False, create_graph=False, return_grad=False):
    '''
    Compute the Hessian of `output` with respect to `inputs`

    hessian((x * y).sum(), [x, y])
    '''
    assert output.ndimension() == 0

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    n = sum(p.numel() for p in inputs)
    if out is None:
        out = output.new_zeros(n, n)
    #out.requires_grad = True

    ai = 0
    if return_grad:
        fullgrad = []
    for i, inp in enumerate(inputs):
        #print("myhessian",i,inp,output)
        [grad] = torch.autograd.grad(output, inp, create_graph=True, retain_graph=True, allow_unused=allow_unused)
        grad = torch.zeros_like(inp) if grad is None else grad
        grad = grad.contiguous().view(-1)

        for j in range(inp.numel()):
            if grad[j].requires_grad:
                row = gradient(grad[j], inputs[i:], create_graph=True, retain_graph=True)[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            out[ai, ai:].add_(row.type_as(out))  # ai's row
            if ai + 1 < n:
                out[ai + 1:, ai].add_(row[1:].type_as(out))  # ai's column
            del row
            ai += 1
        if return_grad:
            fullgrad.append(grad.view(-1))
        else:
            del grad

    if return_grad:
        return (out, torch.cat(fullgrad,0))
    else:
        return out


def arrowhead_hessian(output, inputs, headsize, blocksize, out=None, allow_unused=False,
    create_graph=False, return_grad=False):
    '''
    Compute the Hessian of `output` with respect to `inputs`,
    assuming block arrowhead form. That is, for the call:

    hessian((x * y).sum(), [x1, x2, x3, y1, z1, y2, z2], 3, 2)

    ... the full rows for all elements of x1, x2, and x3 are calculated,
    but [yn,ym], [yn,zm], and [zm,zm] elements are not.

    '''
    assert output.ndimension() == 0

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    n = sum(p.numel() for p in inputs)
    if out is None:
        out = output.new_zeros(n, n)

    ai = 0
    if return_grad:
        fullgrad = []
    I = len(inputs)
    for i, inp in enumerate(inputs):
        #print("graddy",inp.size())
        try:
            [grad] = torch.autograd.grad(output, inp, retain_graph=True, create_graph=True, allow_unused=allow_unused)
        except:
            print("arrowhead_hessian failing",i,inp,output)
            raise
        grad = torch.zeros_like(inp) if grad is None else grad
        grad = grad.contiguous().view(-1)

        if i < headsize:
            maxi = I
        else:
            maxi = headsize + (1 + (i - headsize) // blocksize) * blocksize
        for j in range(inp.numel()):
            if grad[j].requires_grad:
                #print(f"arrow {i},{maxi},{headsize},{blocksize}")
                row = gradient(grad[j], inputs[i:maxi], retain_graph=True, create_graph=True)[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:maxi]) - j)

            rowlen = len(row)
            out[ai, ai:ai+rowlen].add_(row.type_as(out))  # ai's row
            if rowlen > 1:
                out[ai + 1:ai+rowlen, ai].add_(row[1:].type_as(out))  # ai's column
            del row
            ai += 1
        if return_grad:
            fullgrad.append(grad.view(-1))
        else:
            del grad

    if return_grad:
        return (out, torch.cat(fullgrad,0))
    else:
        return out


















def arrowhead_hessian_precision(output_raw, inputs, headsize, blocksize, allow_unused=False,
    create_graph=False, return_grad=False):
    '''
    Compute the Hessian of `output` with respect to `inputs`,
    assuming block arrowhead form. That is, for the call:

    hessian((x * y).sum(), [x1, x2, x3, y1, z1, y2, z2], 3, 2)

    ... the full rows for all elements of x1, x2, and x3 are calculated,
    but [yn,ym], [yn,zm], and [zm,zm] elements are not.

    This returns an ArrowheadPrecision object, as defined in posdef.py
    WARNING: This object does NOT have correct weights, psil, or psig!!!!!
    '''
    assert output_raw.ndimension() == 0

    output = -output_raw #precision is negative of Hessian; easiest to do it this way, not rewrite below

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    G = sum(p.numel() for p in inputs[:headsize])
    L = sum(p.numel() for p in inputs[headsize:headsize+blocksize])

    gg = output.new_zeros(G, G)

    ai = 0
    if return_grad:
        fullgrad = []
    I = len(inputs)
    global_grads = []
    for i, inp in enumerate(inputs[:headsize]): #Globals only
        #print("graddy",inp.size())
        try:
            [grad] = torch.autograd.grad(output, inp, retain_graph=True, create_graph=True, allow_unused=allow_unused)
        except:
            print("arrowhead_hessian_precision failing",i,inp,output)
            raise
        grad = torch.zeros_like(inp) if grad is None else grad
        grad = grad.contiguous().view(-1)

        maxi = headsize
        for j in range(inp.numel()):
            if grad[j].requires_grad:
                #print(f"arrow {i},{maxi},{headsize},{blocksize}")
                row = gradient(grad[j], inputs[i:maxi], retain_graph=True, create_graph=True)[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            rowlen = len(row)
            gg[ai, ai:ai+rowlen].add_(row.type_as(gg))  # ai's row
            if rowlen > 1:
                gg[ai + 1:ai+rowlen, ai].add_(row[1:].type_as(gg))  # ai's column
            del row
            ai += 1
        global_grads.append(grad.view(-1))

    if return_grad:
        fullgrad.extend(global_grads)

    ggrads = torch.cat([g.view(-1) for g in global_grads])

    arrow = ArrowheadPrecision(G, L, gg)



    for l in range((len(inputs) - headsize) // blocksize): #locals
        ll = output.new_zeros(L, L)
        ail = 0 #dim index within unit — counting
        for i, inp in enumerate(inputs[headsize+blocksize*l:headsize+blocksize*(l+1)]): #locals
            ii = headsize + blocksize*l + i #tensor index globally
            maxi = headsize + blocksize*(l+1) #max tensor index globally for unit
            #print("graddy",inp.size())
            try:
                [grad] = torch.autograd.grad(output, inp, retain_graph=True, create_graph=True, allow_unused=allow_unused)
            except:
                print("arrowhead_hessian failing",i,inp,output)
                raise
            grad = torch.zeros_like(inp) if grad is None else grad
            grad = grad.contiguous().view(-1)

            for j in range(inp.numel()):
                # ll first
                if grad[j].requires_grad:
                    #print(f"arrow {i},{maxi},{headsize},{blocksize}")
                    row = gradient(grad[j], inputs[ii:maxi], retain_graph=True, create_graph=True)[j:]
                else:
                    row = grad[j].new_zeros(sum(x.numel() for x in inputs[ii:maxi]) - j)

                rowlen = len(row)
                ll[ail, ail:ail+rowlen].add_(row.type_as(ll))  # ai's row
                if rowlen > 1:
                    ll[ail + 1:ail+rowlen, ail].add_(row[1:].type_as(ll))  # ai's column
                del row
                ail += 1

            if return_grad:
                fullgrad.append(grad.view(-1))
            else:
                del grad


        gl = output.new_zeros(G, L)
        for (k, elem) in enumerate(ggrads):
            #now gl
            unitTensors = inputs[headsize+blocksize*l:headsize+blocksize*(l+1)]
            if elem.requires_grad:
                #print(f"arrow {i},{maxi},{headsize},{blocksize}")
                row = gradient(elem, unitTensors, retain_graph=True, create_graph=True)
            else:
                row = elem.new_zeros(sum(x.numel() for x in unitTensors))

            rowlen = len(row)
            #print("indices:",l,i,j,k,ii,ail)
            #print("sizes:",headsize,blocksize,G,L,gl.size(),ggrads.size())
            #print("ut sizes",[t.size() for t in unitTensors])
            #print("rowlen:",rowlen)
            gl[k, :].add_(row.type_as(ll))  # ai's row
            del row

        arrow.add_one_l(gl,ll)

    if return_grad:
        return (arrow, torch.cat([g.view(-1) for g in fullgrad],0))
    else:
        return arrow

def grts(v):
    return torch.tensor(v,requires_grad=True)

def test_arrowhead_hessian_precision():
    g = [grts([1.,1.,1.]), grts([2.])]
    l1 = [grts([3.,3.]), grts([4.])]
    l2 = [grts([5.,5.]),grts([6.])]
    all = g + l1 + l2
    result = torch.prod(torch.cat([t.view(-1) for t in all])) * all[0][0]
    return arrowhead_hessian_precision(result, all, 2, 2)
