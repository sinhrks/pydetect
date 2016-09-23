#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.stats import t, zscore
import statsmodels.api as sm

from pydetect.base import AnomalyDetector


class GESDDetector(AnomalyDetector):

    def __init__(self, alpha=0.05, r=None):
        self.alpha = 0.05
        self.r = None

    def get_statistics(self, data):

        n = len(data)
        r = int(np.floor(n // 2)) if self.r is None else self.r

        remainings = data.copy()
        outliers = np.zeros(len(data))

        for i in range(r):
            z_score = np.abs(zscore(remainings, ddof=1))
            max_pos = np.argmax(z_score)
            max_z_score = z_score[max_pos]

            # this statistics is quite desceptive
            # needs to compare with other impl
            p = 1 - self.alpha / (2 * (n - i))
            # Critical value
            critical = t.ppf(p, df=n - 2)
            l = (((n - i - 1) * critical) /
                 (np.sqrt((n - i + critical ** 2) * (n - i))))

            if np.isnan(max_z_score) or np.isnan(l) or max_z_score < l:
                break;

            # convert to bool indexer
            max_indexer = np.arange(len(remainings)) == max_pos
            outliers[outliers == 0] = max_indexer

            # update remainings
            remainings = remainings[~max_indexer]

        return outliers



class TimeSeriesGESDDetector(GESDDetector):

    def get_statistics(self, data):
        res = sm.tsa.seasonal_decompose(data)
        resid = res.resid
        return super(TimeSeriesGESDDetector, self).get_statistics(resid)
