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

from cifar_htest_db_structure import ResultVAECIFARHTest

text_pvalues_begin = """
\\begin{{table}}[htbp]
 \\centering
 \\caption{{P-values for hypothesis testing for each category \\textbf{{{0}}} refits and averaging using the \\textbf{{{1}}}.}}
"""

text_pvalues_end = """
 \\label{{tab:cifar_htest_pvalues_{0}_refits_and_{1}_averaging}}
\\end{{table}}
"""

text_errors_begin = """
\\begin{{table}}[htbp]
 \\centering
 \\caption{{Results of the hypothesis testing when applying a critical rate of 5\\% \\textbf{{{0}}} refits and averaging using the \\textbf{{{1}}}. Here G stands for ``good'' and E2 for type 2 error.}}
"""

text_errors_end = """
 \label{{tab:cifar_htest_errors_{0}_refits_and_{1}_averaging}}
\end{{table}}
"""

text_summary_begin = """
\\begin{{table}}[htbp]
 \\centering
 \\caption{{Summary of the results of the hypothesis testing when applying a critical rate of 5\\%.}}
"""

text_summary_end = """
 \label{{tab:cifar_htest_summary}}
\end{{table}}
"""

df = pd.DataFrame(list(ResultVAECIFARHTest.select().dicts()))
del df["id"]

df_summary = pd.DataFrame(columns=[
    'Averaging', 'Refits',
    'Number Type I errors', 'Number Type II errors'
])

out_pvals = out_errors = ""

for averaging in ["median", "mean"]:
    for nrefits in [1, 5]:
        datap = dict()
        datae = dict()
        ce1 = ce2 = 0
        for cat1 in range(10):
            valsp = []
            valse = []
            for cat2 in range(10):
                try:
                    valp = df
                    valp = valp[valp.averaging == averaging]
                    valp = valp[valp.nrefits == nrefits]
                    valp = valp[valp['category1'] == max(cat1, cat2)]
                    valp = valp[valp['category2'] == min(cat1, cat2)]
                    valp = valp.pvalue.item()
                except ValueError:
                    valp = np.nan
                if cat1 == cat2:
                    if valp > .05:
                        vale = "G"
                    else:
                        vale = "E1"
                        ce1 += 1
                else:
                    if valp <= .05:
                        vale = "G"
                    else:
                        vale = "E2"
                        ce2 += 1

                if cat1 >= cat2:
                    valsp.append("{0:.2f}".format(valp))
                    valse.append(vale)
                else:
                    valsp.append("-")
                    valse.append("-")

            datap['c' + str(cat1)] = valsp
            datae['c' + str(cat1)] = valse

        dfp = pd.DataFrame.from_dict(datap)
        dfe = pd.DataFrame.from_dict(datae)

        dfp.index = ['c' + str(cat) for cat in range(10)]
        dfe.index = dfp.index

        # print(dfp)
        # print(dfe)
        p1 = "with" if nrefits == 5 else "without"

        out_pvals += text_pvalues_begin.format(p1, averaging)
        out_pvals += dfp.to_latex()
        out_pvals += text_pvalues_end.format(p1, averaging)

        print("\n")

        out_errors += text_errors_begin.format(p1, averaging)
        out_errors += dfe.to_latex()
        out_errors += text_errors_end.format(p1, averaging)

        print("\n\n\n")

        df_summary.loc[len(df_summary)] = (
            averaging, p1, ce1, ce2
        )

print(out_pvals)
print(out_errors)

df_summary = df_summary.sort_values(list(df_summary.columns))
print(text_summary_begin.format())
print(df_summary.to_latex(index=False))
print(text_summary_end.format())
