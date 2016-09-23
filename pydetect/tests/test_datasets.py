#!/usr/bin/env python

import pydetect

import pandas as pd
import pandas.util.testing as tm


class TestDatasets(tm.TestCase):

    def test_nile(self):
        nile = pydetect.datasets.get_nile()
        self.assertIsInstance(nile, pd.Series)

        # do not overwrite
        new = pydetect.datasets.get_nile()
        self.assertIsNot(nile, new)
