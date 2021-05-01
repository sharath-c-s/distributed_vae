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

import time

import numpy as np
from htest_db_structure import ResultVAEHTest, db
from scipy import stats
from sstudy import do_simulation_study
from vaecompare import HTest

to_sample = dict(
    distribution=range(1),
    no_instances=[10_000],
    dissimilarity=[0, 0.01, 0.1, 0.2],
    ncomparisons=[2, 100],
    averaging=["median", "mean"],
)


def sample_filter(distribution,
                  no_instances,
                  dissimilarity,
                  ncomparisons,
                  averaging):
    if ncomparisons == 1 and averaging == "median":
        return False
    return True


def func(distribution,
         no_instances,
         dissimilarity,
         ncomparisons,
         averaging):
    def data_gen(size, dim, mu):
        res = np.linspace(0.2, 0.9, dim)
        res = stats.lognorm.rvs(res, scale=2, size=(size, dim))
        res -= stats.lognorm.rvs(0.5, scale=2, size=(size, 1))
        res += stats.norm.rvs(loc=mu, scale=2, size=(size, 1))
        return res

    start_time = time.time()
    y_train0 = data_gen(no_instances, 10, 0)
    y_train1 = data_gen(no_instances, 10, dissimilarity)
    htest = HTest(dataloader_workers=0, verbose=1, averaging=averaging)
    htest.fit(y_train0, y_train1, 10000, ncomparisons=ncomparisons)
    elapsed_time = time.time() - start_time

    return dict(
        pvalue=htest.pvalue,
        elapsed_time=elapsed_time,
    )


do_simulation_study(to_sample, func, db, ResultVAEHTest,
                    max_count=200, sample_filter=sample_filter)
