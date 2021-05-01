# ----------------------------------------------------------------------
# Copyright 2018 Marco Inacio <pythonpackages@marcoinacio.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------

from __future__ import division

import collections

import numpy as np
import torch
from scipy.special import logsumexp
from torch.distributions.normal import Normal
from torch.nn import functional as F
from torch.utils.data.dataloader import default_collate


# Based on Pytorch default_collate
class Collate:
    def __init__(self, float_type):
        if float_type == "half":
            self.float_type = torch.float16
        elif float_type == "float":
            self.float_type = torch.float32
        elif float_type == "double":
            self.float_type = torch.float64
        else:
            raise ValueError("Invalid float_type.")

    def __call__(self, batch):
        batch = default_collate(batch)
        return self.post_default_collate(batch)

    def post_default_collate(self, batch):
        if isinstance(batch, torch.Tensor):
            if batch.dtype.is_floating_point:
                return torch.as_tensor(batch, dtype=self.float_type)
            return torch.as_tensor(batch, dtype=torch.int64)
        elif isinstance(batch, collections.Mapping):
            return {key: self.post_default_collate(batch[key])
                    for key in batch}
        elif isinstance(batch, collections.Sequence):
            return [self.post_default_collate(elem) for elem in batch]
        else:
            return batch


def loss_function(output, inputv, distribution):
    if distribution == "gaussian":
        mu_dec, logvar_dec, mu_enc, logvar_enc = output
        dist = Normal(mu_dec, (0.5 * logvar_dec).exp(), True)
        BCE = - dist.log_prob(inputv)
    elif distribution == "bernoulli":
        theta_dec, mu_enc, logvar_enc = output
        BCE = F.binary_cross_entropy(theta_dec, inputv,
                                     reduction='sum')

    KLD = torch.sum(1 + logvar_enc - mu_enc.pow(2) - logvar_enc.exp())
    KLD = -0.5 * KLD

    return (BCE + KLD).mean()


def _pre_kld_gaussians(mua, mub, logvara, logvarb):
    distance = logvarb.sum() - logvara.sum()
    distance -= len(logvara)
    distance += np.exp(logsumexp(logvara - logvarb))

    distance += ((mub - mua) ** 2 * np.exp(logvara)).sum()
    return distance * 0.5


def kld_gaussians(mua, mub, logvara, logvarb):
    d1 = _pre_kld_gaussians(mua, mub, logvara, logvarb)
    d2 = _pre_kld_gaussians(mub, mua, logvarb, logvara)
    return (d1 + d2) * 0.5 / len(mua)


def kld_bernoullis(thetaa, thetab):
    distancep1 = thetaa - thetab
    distancep2 = np.log(thetab) - np.log(thetaa)
    distancep2 += np.log(1 - thetaa) - np.log(1 - thetab)

    distance = distancep1 * distancep2 * 0.5
    return - distance.sum() / len(thetaa)
