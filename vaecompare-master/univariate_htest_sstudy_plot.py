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

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from univariate_htest_db_structure import ResultUVAEHTest

df = pd.DataFrame(list(ResultUVAEHTest.select()
                       .where(ResultUVAEHTest.dissimilarity == 0.1)
                       # .order_by(fn.Random()).limit(1000)
                       .dicts()))

del df["id"]
assert all(df["distribution"] == 0)
del df["distribution"]
assert all(df["no_instances"] == 1000)
del df["no_instances"]
assert all(df["nrefits"] == 10)
del df["nrefits"]
assert all(df["num_layers"] == 5)
del df["num_layers"]
assert all(df["dissimilarity"] == 0.1)
del df["dissimilarity"]
del df["elapsed_time"]

method = df.method.copy()
method[method == "vaecompare_median"] = "vaecompare"
method[method == "mannwhitneyu"] = "M-Whitney"
method[method == "ks"] = "K-Smirnov"
method[method == "ttest"] = "Welch"
df['method'] = method

to_group = ['method']

alpha_grid = np.linspace(0, 0.20, 10100)


def mpse_all(data):
    res = np.empty((len(alpha_grid), 2)) + np.nan
    for i, alpha in enumerate(alpha_grid):
        rejections = data <= alpha
        mean = rejections.mean()
        std_error = rejections.std() / np.sqrt(len(rejections))
        res[i] = [mean, std_error]

    # smoothfier
    if smoothfy:
        previous = np.nan
        change_point = []
        for i, alpha in enumerate(alpha_grid):
            if res[i][0] != previous:
                previous = res[i][0]
                change_point.append(i)

        for i in range(len(change_point) - 1):
            for j in range(change_point[i] + 1, change_point[i + 1]):
                w_1 = change_point[i + 1] - j
                w_2 = j - change_point[i]

                mean_c1 = res[change_point[i], 0]
                mean_c2 = res[change_point[i + 1], 0]
                mean = mean_c1 * w_1 + mean_c2 * w_2
                mean /= (w_1 + w_2)

                std_error_c1 = res[change_point[i], 1]
                std_error_c2 = res[change_point[i + 1], 1]
                std_error = std_error_c1 * w_1 + std_error_c2 * w_2
                std_error /= (w_1 + w_2)

                res[j] = [mean, std_error]

    return pickle.dumps(res)


try:
    os.mkdir('plots')
except FileExistsError:
    pass

for smoothfy in [True, False]:
    dfplot = df.groupby(to_group).agg(mpse_all)
    fig, ax = plt.subplots()
    for i in range(len(dfplot)):
        method = dfplot.iloc[i].name
        mean, stderror = zip(*pickle.loads(dfplot.iloc[i, 0]))
        mean = np.array(mean)
        stderror = np.array(stderror)
        ax.fill_between(alpha_grid, mean + stderror * 2,
                        mean - stderror * 2, alpha=.5)
        ax.plot(alpha_grid, mean, label=method)
        fname = 'univ_smoothfy_{0}.pdf'
        fname = fname.format(smoothfy)
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("Power")
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              ncol=2, mode="expand", borderaxespad=0.)
    fig.savefig(os.path.join('plots', fname))
    # plt.show()
