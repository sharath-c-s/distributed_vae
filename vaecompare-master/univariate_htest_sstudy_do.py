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
from scipy import stats
from sstudy import do_simulation_study
from univariate_htest_db_structure import ResultUVAEHTest, db
from vaecompare import HTest

to_sample = dict(
    distribution=[0],
    no_instances=[1000],
    dissimilarity=[
        # 0.0,
        0.1,
    ],
    method=[
        'vaecompare_median',
        # 'vaecompare_mean',
        'mannwhitneyu',
        'ks',
        'ttest',
    ],
    nrefits=[10],
    num_layers=[5],
)


def sample_filter(distribution,
                  no_instances,
                  dissimilarity,
                  method,
                  nrefits,
                  num_layers):
    if method[:10] == 'vaecompare':
        return 500

    return True


def func(distribution,
         no_instances,
         dissimilarity,
         method,
         nrefits,
         num_layers):
    def data_gen(distribution, size, dissimilarity):
        probs = np.random.random(size)
        res = np.empty(size)
        if distribution == 0:
            for i in range(size):
                if probs[i] < 1 / 3:
                    mu = -2. + dissimilarity
                elif probs[i] > 2 / 3:
                    mu = 0. + dissimilarity
                else:
                    mu = 2. + dissimilarity
                res[i] = stats.norm.rvs(mu)

        return res

    start_time = time.time()
    y_train0 = data_gen(distribution, no_instances, 0)
    y_train1 = data_gen(distribution, no_instances, dissimilarity)

    if method == 'mannwhitneyu':
        htest = stats.mannwhitneyu(y_train0, y_train1,
                                   alternative='two-sided')

    if method == 'ks':
        htest = stats.ks_2samp(y_train0, y_train1)

    if method == 'ttest':
        htest = stats.ttest_ind(y_train0, y_train1,
                                equal_var=False)

    if method[:10] == 'vaecompare':
        averaging = method[11:]

        htest = HTest(dataloader_workers=0, verbose=1,
                      averaging=averaging,
                      num_layers_decoder=num_layers,
                      num_layers_encoder=num_layers,
                      )
        htest.fit(y_train0, y_train1, 10000,
                  nrefits=nrefits,
                  )

    elapsed_time = time.time() - start_time

    return dict(
        pvalue=htest.pvalue,
        elapsed_time=elapsed_time,
    )


do_simulation_study(to_sample, func, db, ResultUVAEHTest,
                    max_count=10000, sample_filter=sample_filter)
