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
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cifar_compare_db_structure import ResultVAECIFARCompare
from matplotlib.backends.backend_pdf import PdfPages
from vaecompare.utils import kld_bernoullis

df = pd.DataFrame(list(ResultVAECIFARCompare.select().dicts()))
df = df.sort_values(['category1', 'category2'])
df["samples"] = [pickle.loads(x) for x in df["samples"]]
df["mean_kl_divergence"] = [x.mean() for x in df["samples"]]
df["std_kl_divergence"] = [x.std() for x in df["samples"]]
del df["id"]

assert (df["mean_kl_divergence"] > 0).all()

# Dot and box plots prepare data

for mtype1 in range(10):
    names = list()
    machine_names = np.empty(0)
    dotvals = np.empty((0, 2))
    boxvals = list()
    metricobjlist = list()
    for i, (mtype2) in enumerate(range(10)):
        rawvals = df.loc[df["category1"] == min(mtype1, mtype2)]
        rawvals = rawvals.loc[df["category2"] == max(mtype1, mtype2)]
        rawvals = rawvals["samples"].iloc[:90]
        # rawvals = rawvals.iloc[0]
        rawvals = reduce(lambda x, y: np.hstack([x, y]), rawvals)

        names.append("{} vs {}".format(mtype1, mtype2))

        machine_name = i + 1
        machine_names = np.hstack((machine_names, machine_name))
        repmn = np.repeat(machine_name, rawvals.size)
        dotval = np.column_stack((repmn, rawvals))
        dotvals = np.vstack((dotvals, dotval))

        boxvals.append(rawvals)

    fig, axes = plt.subplots(2)
    fig.set_size_inches(7.3, 8.9)

    # Dot plot
    # ax.scatter(dotvals[:, 0], dotvals[:, 1], marker='o', s=15.0,
    #                      color="green")
    # ax.set_xlabel("Models compared")
    # ax.set_ylabel("Probability")
    # ax.set_ylim(0.9, 1.05)

    # Box plot
    for ax in axes:
        ax.violinplot(boxvals, showextrema=False)
        bplot = ax.boxplot(boxvals,
                           whis=[.02, .98],
                           showfliers=False,
                           showmeans=True,
                           meanline=True, notch=True)

        for mean in bplot['means']:
            mean.set(color='#FF6600')

        for median in bplot['medians']:
            median.set(color='blue')

        for i, v in enumerate([.35, .4, .45, .49]):
            label = "$\\mathbb{D}(p="
            label += "{0:.2f}".format(v)
            label += "; q="
            label += "{0:.2f}".format(1 - v)
            label += ")$"
            ax.axhline(
                y=kld_bernoullis(np.repeat(v, 3072), np.repeat(1 - v, 3072)),
                xmin=0.0, xmax=1.0,
                color=["red", "blue", "green", "yellow"][i],
                label=label)

        ax.set_ylabel("Divergence")

    axes[1].set_xlabel("Models compared")
    axes[0].tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False
    )
    axes[1].set_xticklabels(names)

    # if mtype1 in [0, 1, 6, 7]:
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                   ncol=2, fancybox=True, shadow=True)
    fig.subplots_adjust(hspace=0.19)

    axes[1].set_ylim(0, 0.01)

    ps = PdfPages("plots/dotplot_" + str(mtype1) + ".pdf")
    ps.savefig(fig, bbox_inches='tight')
    ps.close()
    plt.close(fig)
