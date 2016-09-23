#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.stats import t, zscore
import statsmodels.api as sm

from pydetect.base import OutlierDetector


class GESDDetector(OutlierDetector):

    """
    http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm
    """

    def __init__(self, alpha=0.05, max_outliers=None):
        self.alpha = 0.05
        self.max_outliers = None

    def _get_lambda(self, n, i):
        # because of the loop condition, original formula's i
        # is replaced by i + 1
        df = n - i - 2

        p = 1 - self.alpha / (2 * (n - i))
        # Critical value
        critical = t.ppf(p, df=df)
        lambda_i = (((n - i - 1) * critical) /
                   (np.sqrt((df + critical ** 2) * (n - i))))
        return lambda_i

    def get_statistics(self, data):

        n = len(data)
        if self.max_outliers is None:
            max_outliers = int(np.floor(n // 2))
        else:
            max_outliers = self.max_outliers

        remainings = data.copy()
        outliers = np.zeros(len(data))

        # number of outliers
        n_outliers = 0

        r = []
        l = []

        for i in range(max_outliers):
            # s must be sample standard deviation
            z_score = np.abs(zscore(remainings, ddof=1))
            max_pos = np.argmax(z_score)
            r_i = z_score[max_pos]

            lambda_i = self._get_lambda(n, i)

            # ToDo: ?
            r.append(r_i)
            l.append(lambda_i)

            if np.isnan(r_i) or np.isnan(lambda_i):
                break;
            elif r_i > lambda_i:
                n_outliers = i + 1

            # convert to bool indexer
            max_indexer = np.arange(len(remainings)) == max_pos

            flag = np.zeros(len(max_indexer))
            flag[max_indexer] = i + 1

            outliers[outliers == 0] = flag

            # update remainings
            remainings = remainings[~max_indexer]

        outliers = (outliers != 0) & (outliers <= n_outliers)

        return outliers, np.array(r), np.array(l)



class TimeSeriesGESDDetector(GESDDetector):

    def get_statistics(self, data):
        res = sm.tsa.seasonal_decompose(data)
        resid = res.resid
        return super(TimeSeriesGESDDetector, self).get_statistics(resid)
