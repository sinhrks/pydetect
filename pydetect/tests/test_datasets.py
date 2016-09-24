#!/usr/bin/env python

import pydetect

import pandas as pd
import pandas.util.testing as tm


class TestDatasets(tm.TestCase):

    def test_nile(self):
        nile = pydetect.datasets.get_nile()
        self.assertIsInstance(nile, pd.Series)
        self.assertEqual(nile.name, 'Nile')

        # do not overwrite
        new = pydetect.datasets.get_nile()
        self.assertIsNot(nile, new)

    def test_airpassengers(self):
        ap = pydetect.datasets.get_airpassengers()
        self.assertIsInstance(ap, pd.Series)
        self.assertEqual(ap.name, 'Air Passengers')

        # do not overwrite
        new = pydetect.datasets.get_airpassengers()
        self.assertIsNot(ap, new)
