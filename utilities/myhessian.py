# pylint: disable=E1101, C0103, C0111
'''
Computes the Hessian
'''
import torch
from hessian.gradient import gradient

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
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

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
