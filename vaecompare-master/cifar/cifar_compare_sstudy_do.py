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

import pickle
import time

from cifar_compare_db_structure import ResultVAECIFARCompare, db
from sstudy import do_simulation_study
from utils import get_categories
from vaecompare import Compare

to_sample = dict(
    category1=range(10),
    category2=range(10),
)


def sample_filter(category1, category2):
    if category2 > category1:
        return False
    return True


def func(category1, category2):
    start_time = time.time()
    y_train1, y_train2 = get_categories(category1, category2)
    compare = Compare(dataloader_workers=0, verbose=2,
                      distribution="bernoulli")
    compare.fit(y_train1, y_train2, 10000)
    elapsed_time = time.time() - start_time

    return dict(
        samples=pickle.dumps(compare.samples),
        elapsed_time=elapsed_time,
    )


do_simulation_study(to_sample, func, db, ResultVAECIFARCompare,
                    sample_filter=sample_filter, max_count=90)
