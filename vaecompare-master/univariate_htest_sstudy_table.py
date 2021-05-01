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

import numpy as np
import pandas as pd

from univariate_htest_db_structure import ResultUVAEHTest

df = pd.DataFrame(list(ResultUVAEHTest.select()
                       # .order_by(fn.Random()).limit(10000)
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
del df["elapsed_time"]

method = df.method.copy()
method[method == "vaecompare_median"] = "vaecompare"
method[method == "mannwhitneyu"] = "M-Whitney"
method[method == "ks"] = "K-Smirnov"
method[method == "ttest"] = "Welch"
df['method'] = method

to_group = ['dissimilarity', 'method']

df["e1_power_1"] = df['pvalue'] <= 0.01
df["e1_power_5"] = df['pvalue'] <= 0.05
df["e1_power_10"] = df['pvalue'] <= 0.10
count = df.groupby(to_group).count().iloc[:, 0]


def mpse(data):
    mean = data.mean()
    std_error = np.std(data) / np.sqrt(len(data))
    return "{0:.3f} ({1:.3f})".format(mean, std_error)


grouped = df.groupby(to_group).agg(mpse)
grouped["count"] = count

grouped.rename(columns={"count": "n sim"}, inplace=True)
grouped.rename(columns={"pvalue": "avg pvalue"}, inplace=True)

grouped1 = grouped[[x[0] == 0.0 for x in grouped.index]]
grouped2 = grouped[[x[0] == 0.1 for x in grouped.index]]

grouped1.rename(columns={"e1_power_1": "Error (α=1%)"}, inplace=True)
grouped1.rename(columns={"e1_power_5": "Error (α=5%)"}, inplace=True)
grouped1.rename(columns={"e1_power_10": "Error (α=10%)"}, inplace=True)
grouped2.rename(columns={"e1_power_1": "Power (α=1%)"}, inplace=True)
grouped2.rename(columns={"e1_power_5": "Power (α=5%)"}, inplace=True)
grouped2.rename(columns={"e1_power_10": "Power (α=10%)"}, inplace=True)

print(grouped1)
print(grouped2)
print(grouped1.to_latex(index=True))
print(grouped2.to_latex(index=True))
