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
from scipy import stats

from .compare import Compare


class HTest():
    def __init__(self,
                 *args,
                 averaging,
                 distribution="gaussian",
                 **kwargs,
                 ):
        self.args = args
        self.kwargs = kwargs
        self.distribution = distribution
        self.averaging = averaging

    def fit(self, y_train0, y_train1, nsamples=10000, ncomparisons=100,
            nrefits=1):
        len0 = len(y_train0)
        if len(y_train0.shape) == 1:
            y_train0 = y_train0[:, None]
        if len(y_train1.shape) == 1:
            y_train1 = y_train1[:, None]

        y_train01 = np.vstack((y_train0, y_train1))

        averaging = self.averaging
        if averaging == "median":
            averaging = np.median
        elif averaging == "mean":
            averaging = np.mean

        divergences = []
        for i in range(ncomparisons):
            cfsamples = np.arange(0)
            for j in range(nrefits):
                tsamples = Compare(*self.args,
                                   distribution=self.distribution, **self.kwargs)
                tsamples = tsamples.fit(y_train0, y_train1, nsamples)
                tsamples = tsamples.samples
                cfsamples = np.hstack([cfsamples, tsamples])
            divergences.append(cfsamples)

            y_train01 = np.random.permutation(y_train01)
            y_train0 = y_train01[:len0]
            y_train1 = y_train01[len0:]
            print("Made comparison", (i + 1) * nrefits, "out of",
                  (ncomparisons) * nrefits)

        if ncomparisons > 2:
            divergences = [averaging(x) for x in divergences]
            self.divergence_unpermuted = divergences[0]
            self.divergence_permuted = np.array(divergences[1:])

            n1 = (self.divergence_unpermuted <=
                  self.divergence_permuted).sum()
            n1 = (n1 + 1) / ncomparisons
            n2 = (self.divergence_unpermuted <
                  self.divergence_permuted).sum() / (ncomparisons + 1)
            n2 = (n2 + 1) / ncomparisons

            self.pvalue = (n1 + n2) / 2
        elif ncomparison == 2:
            self.divergence_unpermuted = divergences[0]
            self.divergence_permuted = divergences[1]

            # H0: population1.mean() >= population2.mean()
            def one_tailed_test(sample1, sample2):
                pvalue = stats.ttest_rel(sample1, sample2).pvalue
                if sample1.mean() <= sample2.mean():
                    pvalue /= 2
                else:
                    pvalue = 1 - pvalue / 2
                return pvalue

            self.pvalue = one_tailed_test(self.divergence_unpermuted,
                                          self.divergence_permuted)
            # self.pvalue = stats.ks_2samp(self.divergence_unpermuted,
            #     self.divergence_permuted).pvalue
            self.pvalue
        else:
            raise ValueError("Invalid number of comparison")

        return self
