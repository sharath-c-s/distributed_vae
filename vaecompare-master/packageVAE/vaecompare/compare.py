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

import numpy as np

from .estimate import VAE
from .utils import kld_gaussians, kld_bernoullis


class Compare():
    def __init__(self,
                 *args,
                 distribution="gaussian",
                 **kwargs,
                 ):

        self.vae0 = VAE(*args, distribution=distribution, **kwargs)
        self.vae1 = VAE(*args, distribution=distribution, **kwargs)
        self.samples = np.empty(0)
        self.fitted = False
        self.distribution = distribution

    def fit(self, y_train0, y_train1, nsamples=10000):
        if len(y_train0.shape) == 1:
            y_train0 = y_train0[:, None]
        if len(y_train1.shape) == 1:
            y_train1 = y_train1[:, None]
        self.vae0.fit(y_train0)
        self.vae1.fit(y_train1)
        self.fitted = True
        self.improve_comparison(nsamples)

        return self

    def improve_comparison(self, nsamples=10000):
        if not self.fitted:
            raise Exception("Call method fit first")

        s0 = self.vae0.sample_parameters(nsamples)
        s1 = self.vae1.sample_parameters(nsamples)

        samples = np.empty(nsamples)
        for i in range(nsamples):
            if self.distribution == "gaussian":
                kld = kld_gaussians(s0[0][i], s1[0][i], s0[1][i],
                                    s1[1][i])

            elif self.distribution == "bernoulli":
                kld = kld_bernoullis(s0[i], s1[i])

            samples[i] = kld

        self.samples = np.hstack([self.samples, samples])

        return self
