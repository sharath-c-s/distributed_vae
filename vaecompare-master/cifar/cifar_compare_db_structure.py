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
    db = SqliteDatabase('results.sqlite3')


class ResultVAECIFARCompare(Model):
    # Data settings
    category1 = IntegerField()
    category2 = IntegerField()

    # Estimation settings
    samples = BlobField()
    elapsed_time = DoubleField()

    class Meta:
        database = db
        # indexes = (
        #    (('category1', 'category2'), True),
        # )


ResultVAECIFARCompare.create_table()
