#!/usr/bin/env python

import pydetect

import numpy as np
import pandas as pd

import unittest
import pandas.util.testing as tm


class TestGeneralizedESD(unittest.TestCase):

    def test_sample(self):
        """
        the test is derived from
        http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm
        """

        d = pydetect.GESDDetector(alpha=0.05)
        self.assertEqual(d._get_lambda(54, 0), 3.1587939408872967)
        self.assertEqual(d._get_lambda(54, 1), 3.1514300233157844)
        self.assertEqual(d._get_lambda(54, 2), 3.1438896850317302)
        self.assertEqual(d._get_lambda(54, 3), 3.1361649560574856)
        self.assertEqual(d._get_lambda(54, 4), 3.1282473343306387)
        self.assertEqual(d._get_lambda(54, 5), 3.1201277383143937)
        self.assertEqual(d._get_lambda(54, 6), 3.1117964542894945)
        self.assertEqual(d._get_lambda(54, 7), 3.1032430776016966)
        self.assertEqual(d._get_lambda(54, 8), 3.0944564470227012)
        self.assertEqual(d._get_lambda(54, 9), 3.0854245712422848)

    def test_detect(self):
        data = [-0.25, 0.68, 0.94, 1.15, 1.20, 1.26, 1.26,
                1.34, 1.38, 1.43, 1.49, 1.49, 1.55, 1.56,
                1.58, 1.65, 1.69, 1.70, 1.76, 1.77, 1.81,
                1.91, 1.94, 1.96, 1.99, 2.06, 2.09, 2.10,
                2.14, 2.15, 2.23, 2.24, 2.26, 2.35, 2.37,
                2.40, 2.47, 2.54, 2.62, 2.64, 2.90, 2.92,
                2.92, 2.93, 3.21, 3.26, 3.30, 3.59, 3.68,
                4.30, 4.64, 5.34, 5.42, 6.01]
        exp = np.array([False] * 51 + [True] * 3)

        d = pydetect.GESDDetector(alpha=0.05)
        res = d.detect(data)
        print(res)
        tm.assert_numpy_array_equal(res, exp)

    def test_decompose(self):
        ap = pydetect.datasets.get_airpassengers()
        ap[50] += 100

        d = pydetect.GESDDetector(alpha=0.05, decompose=True)
        res = d.detect(ap)

        exp = np.zeros(144)
        exp[50] = 1.0
        exp = pd.Series(exp, index=ap.index)
        tm.assert_series_equal(res, exp)
