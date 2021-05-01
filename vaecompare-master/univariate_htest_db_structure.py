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

from peewee import *

try:
    pgdb = os.environ['pgdb']
    pguser = os.environ['pguser']
    pgpass = os.environ['pgpass']
    pghost = os.environ['pghost']
    pgport = os.environ['pgport']

    db = PostgresqlDatabase(pgdb, user=pguser, password=pgpass,
                            host=pghost, port=pgport)
except KeyError:
    db = SqliteDatabase('results_univariate_htest.sqlite3')


class ResultUVAEHTest(Model):
    # Data settings
    distribution = IntegerField()
    no_instances = IntegerField()
    dissimilarity = DoubleField()
    method = TextField()
    nrefits = IntegerField()
    num_layers = IntegerField()

    # Estimation settings
    pvalue = DoubleField()
    elapsed_time = DoubleField()

    class Meta:
        database = db


ResultUVAEHTest.create_table()
