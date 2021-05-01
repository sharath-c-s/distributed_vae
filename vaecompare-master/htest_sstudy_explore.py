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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from htest_db_structure import ResultVAEHTest
from matplotlib.backends.backend_pdf import PdfPages


def ecdf_plot(x, ax, *args, **kwargs):
    xc = np.concatenate(([0], np.sort(x), [1]))
    y = np.linspace(0, 1, len(x) + 1)
    yc = np.concatenate(([0], y))
    ax.step(xc, yc, *args, **kwargs)


df = pd.DataFrame(list(ResultVAEHTest.select().dicts()))
del df["id"]

cls = [":", "-", "-.", "--", "-", "-."]
clw = [2.2, 2.2, 2.2, 2.2, 1.0, 1.0]
for ncomparisons in [1, 100]:
    for averaging in ["median", "mean"]:
        if ncomparisons == 1 and averaging == "median":
            continue
        fig, ax = plt.subplots()
        for i, dissimilarity in enumerate([0, 0.01, 0.1, 0.2]):
            vals = df
            vals = vals[vals['dissimilarity'] == dissimilarity]
            vals = vals[vals['ncomparisons'] == ncomparisons]
            vals = vals[vals['averaging'] == averaging]
            vals = vals.pvalue
            print(len(vals))
            name = "dissimilarity " + str(dissimilarity)
            ecdf_plot(vals, ax, label=name,
                      linestyle=cls[i], lw=clw[i])

        legend = ax.legend(shadow=True, frameon=True, loc='best',
                           fancybox=True, borderaxespad=.5)

        ax.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000),
                c='black')

        filename = "plots/gen_data_compare"
        filename += "_ncomparisons_" + str(ncomparisons)
        if averaging == 'median':
            filename += "_averaging_" + str(averaging)
        filename += ".pdf"
        with PdfPages(filename) as ps:
            ps.savefig(fig, bbox_inches='tight')
        plt.close(fig)
