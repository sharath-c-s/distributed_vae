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
from cifar_htest_db_structure import ResultVAECIFARHTest, db
from sstudy import do_simulation_study
from utils import get_categories
from vaecompare import HTest

to_sample = dict(
    category1=range(10),
    category2=range(10),
    averaging=["median", "mean"],
    nrefits=[1, 5],
)


def sample_filter(category1, category2, averaging, nrefits):
    if category2 > category1:
        return False
    return True


def func(category1, category2, averaging, nrefits):
    start_time = time.time()
    np.random.seed(10 * category1 + category2)
    y_train1, y_train2 = get_categories(category1, category2)
    np.random.seed()
    htest = HTest(dataloader_workers=0, verbose=1,
                  distribution="bernoulli", averaging=averaging)
    htest.fit(y_train1, y_train2, nrefits=nrefits)
    elapsed_time = time.time() - start_time

    return dict(
        pvalue=htest.pvalue,
        elapsed_time=elapsed_time,
    )


do_simulation_study(to_sample, func, db, ResultVAECIFARHTest,
                    sample_filter=sample_filter)
