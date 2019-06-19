from __future__ import absolute_import, division, print_function

import numbers

import torch
from torch.distributions import constraints

from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import sum_rightmost


class VecDelta(TorchDistribution):
    """
    Degenerate discrete distribution (a single point).

    Discrete distribution that assigns probability one to the single element in
    its support. VecDelta distribution parameterized by a random choice should not
    be used with MCMC based inference, as doing so produces incorrect results.

    :param torch.Tensor v: The single support element.
    :param torch.Tensor log_density: An optional density for this VecDelta. This
        is useful to keep the class of :class:`VecDelta` distributions closed
        under differentiable transformation.
    :param int event_dim: Optional event dimension, defaults to zero.
    """
    has_rsample = True
    arg_constraints = {'v': constraints.real, 'log_density': constraints.real}
    support = constraints.real

    def __init__(self, v, log_density=0.0, validate_args=None):
        batch_dim = v.dim() - 1
        batch_shape = v.shape[:batch_dim]
        if isinstance(log_density, numbers.Number):
            log_density = torch.full(batch_shape, log_density, dtype=v.dtype, device=v.device)
        elif validate_args and log_density.shape != batch_shape:
            raise ValueError('Expected log_density.shape = {}, actual {}'.format(
                log_density.shape, batch_shape))
        self.v = v
        event_shape = v.shape[-1:]
        self.log_density = log_density
        super(VecDelta, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(VecDelta, _instance)
        batch_shape = torch.Size(batch_shape)
        param_shape = batch_shape + self.event_shape
        new.v = self.v.expand(param_shape)
        new.log_density = self.log_density.expand(batch_shape)
        event_shape = self.event_shape
        super(VecDelta, new).__init__(batch_shape, event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = sample_shape + self.v.shape
        print("rsample",shape,sample_shape)
        return self.v.expand(shape)

    def xxxrsample(self, sample_shape=torch.Size()):
        sample_shape = self._extended_shape(sample_shape)
        param_shape = sample_shape + self.event_shape
        v = self.v.expand(param_shape)
        print("xxxrsample",param_shape,sample_shape,v.shape,v.contiguous().view(sample_shape).shape)
        return v.contiguous().view(sample_shape)

    def xxxsample(self, sample_shape=torch.Size()):
        sample_shape = self._extended_shape(sample_shape)
        param_shape = sample_shape + self.event_shape
        v = self.v.expand(param_shape)
        print("sample",param_shape,sample_shape,v.shape,v.contiguous().view(sample_shape).shape)
        return v.contiguous().view(sample_shape)

    def log_prob(self, x):
        v = self.v.expand(self.shape())
        log_prob = (x == v).type(x.dtype).log()
        print("log_prob:",log_prob)
        log_prob = sum_lefttmost(log_prob, 1)
        print("log_prob2:",log_prob)
        log_prob = sum_rightmost(log_prob, self.event_dim)
        print("log_prob3:",log_prob)
        return log_prob + self.log_density

    @property
    def mean(self):
        return self.v

    @property
    def variance(self):
        return torch.zeros_like(self.v)
