#!/usr/bin/env python

import pydetect

import numpy as np
import pandas as pd

import unittest
import pandas.util.testing as tm


class TestMeanDetector(unittest.TestCase):

    def test_nile_statistics(self):
        nile = pydetect.datasets.get_nile()

        d = pydetect.MeanDetector()
        i, n, v = d.get_statistics(nile)

        self.assertEqual(27, i)
        self.assertEqual(2835156.75, n)
        self.assertEqual(1597457.1944444478, v)

    def test_input_class(self):
        nile = pydetect.datasets.get_nile()
        d = pydetect.MeanDetector()

        exp = np.array([0.] * 100)
        exp[27] = 1.
        exp_index = pd.date_range('1871-01-01', freq='AS', periods=100)
        # name will be reset
        exp = pd.Series(exp, exp_index)

        res = d.detect(nile)
        tm.assert_series_equal(res, exp)

        res = d.detect(nile.to_frame())
        tm.assert_frame_equal(res, exp.to_frame())

        res = d.detect(np.array(nile))
        tm.assert_numpy_array_equal(res, exp.values)

        res = d.detect(nile.tolist())
        tm.assert_numpy_array_equal(res, exp.values)


class TestVarianceDetector(unittest.TestCase):

    def test_nile(self):
        nile = pydetect.datasets.get_nile()

        d = pydetect.VarianceDetector()
        i, n, v = d.get_statistics(nile)

        self.assertEqual(46, i)
        self.assertEqual(1025.2437598632698, n)
        self.assertEqual(1013.6146385197355, v)

    def test_input_class(self):
        nile = pydetect.datasets.get_nile()
        d = pydetect.VarianceDetector()

        exp = np.array([0.] * 100)
        exp[46] = 1.
        exp_index = pd.date_range('1871-01-01', freq='AS', periods=100)
        # name will be reset
        exp = pd.Series(exp, exp_index)

        res = d.detect(nile)
        tm.assert_series_equal(res, exp)

        res = d.detect(nile.to_frame())
        tm.assert_frame_equal(res, exp.to_frame())

        res = d.detect(np.array(nile))
        tm.assert_numpy_array_equal(res, exp.values)

        res = d.detect(nile.tolist())
        tm.assert_numpy_array_equal(res, exp.values)


class TestMeanVarianceDetector(unittest.TestCase):

    def test_nile(self):
        nile = pydetect.datasets.get_nile()

        d = pydetect.MeanVarianceDetector()
        i, n, v = d.get_statistics(nile)

        self.assertEqual(27, i)
        self.assertEqual(1025.2437598632698, n)
        self.assertEqual(967.68788456436801, v)

    def test_input_class(self):
        nile = pydetect.datasets.get_nile()
        d = pydetect.MeanVarianceDetector()

        exp = np.array([0.] * 100)
        exp[27] = 1.
        exp_index = pd.date_range('1871-01-01', freq='AS', periods=100)
        # name will be reset
        exp = pd.Series(exp, exp_index)

        res = d.detect(nile)
        tm.assert_series_equal(res, exp)

        res = d.detect(nile.to_frame())
        tm.assert_frame_equal(res, exp.to_frame())

        res = d.detect(np.array(nile))
        tm.assert_numpy_array_equal(res, exp.values)

        res = d.detect(nile.tolist())
        tm.assert_numpy_array_equal(res, exp.values)
