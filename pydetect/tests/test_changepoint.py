#!/usr/bin/env python

import pydetect
import pandas.util.testing as tm


class TestMeanDetector(tm.TestCase):

    def test_nile(self):
        nile = pydetect.datasets.get_nile()

        d = pydetect.MeanDetector()
        i, n, v = d.get_statistics(nile)

        self.assertEqual(27, i)
        self.assertEqual(2835156.75, n)
        self.assertEqual(1597457.1944444478, v)


class TestVarianceDetector(tm.TestCase):

    def test_nile(self):
        nile = pydetect.datasets.get_nile()

        d = pydetect.VarianceDetector()
        i, n, v = d.get_statistics(nile)

        self.assertEqual(46, i)
        self.assertEqual(1025.2437598632698, n)
        self.assertEqual(1013.6146385197355, v)


class TestMeanVarianceDetector(tm.TestCase):

    def test_nile(self):
        nile = pydetect.datasets.get_nile()

        d = pydetect.MeanVarianceDetector()
        i, n, v = d.get_statistics(nile)

        self.assertEqual(27, i)
        self.assertEqual(1025.2437598632698, n)
        self.assertEqual(967.68788456436801, v)
