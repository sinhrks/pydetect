#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import numpy as np
from pydetect.base import SingleChangePointDetector


class MeanDetector(SingleChangePointDetector):

    def get_statistics(self, data):

        n = len(data)
        x = np.cumsum(data)
        x2 = np.cumsum(data ** 2)

        null = x2[-1] - (x[-1] ** 2 / n)

        denom = np.arange(1, n + 1)
        values = (x2 - x ** 2 / denom + (x2[-1] - x2) -
                  ((x[-1] - x) ** 2) / (n - denom))

        i = np.nanargmin(values)
        return i, null, values[i]


class VarianceDetector(SingleChangePointDetector):

    def get_statistics(self, data):

        mu = np.mean(data)

        n = len(data)
        x = np.cumsum((data - mu) ** 2)
        null = n * np.log(x[-1] / n)

        denom = np.arange(1, n + 1)
        denom_rev = n - denom

        sigma1 = x / denom
        sigma1[sigma1 <= 0] = 1e-10

        sigman = (x[-1] - x) / denom_rev
        sigman[sigman <= 0] = 1e-10

        values = denom * np.log(sigma1) + denom_rev * np.log(sigman)

        i = np.nanargmin(values)
        return i, null, values[i]


class MeanVarianceDetector(SingleChangePointDetector):

    def get_statistics(self, data):
        n = len(data)
        y = np.cumsum(data)
        y2 = np.cumsum(data ** 2)

        null = n * np.log((y2[-1] - (y[-1] ** 2 / n)) / n)

        denom = np.arange(1, n + 1)
        denom_rev = n - denom

        sigma1 = (y2 - (y ** 2 / denom)) / denom
        sigma1[sigma1 <= 0] = 1e-10

        sigman = ((y2[-1] - y2) - (y[-1] - y) ** 2 / denom_rev) / denom_rev
        sigman[sigman <= 0] = 1e-10

        values = denom * np.log(sigma1) + denom_rev * np.log(sigman)

        i = np.nanargmin(values)
        return i, null, values[i]
