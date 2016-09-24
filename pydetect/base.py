#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd


class BaseDetector(object):

    def __init__(self, decompose=False):
        self.decompose = decompose

    def detect(self, data):
        raise NotImplementedError

    def get_statistics(self, data):
        raise NotImplementedError

    def _validate(self, data):
        """ validate data """
        self._original = data

        if self.decompose:
            data = self._decompose(data)

        data = np.asarray(data)
        if data.ndim > 2:
            raise ValueError('Input must be less than 2 dimentions')
        elif data.ndim == 2 and data.shape[1] > 1:
            raise ValueError('Input must be univariate')

        return data

    def _decompose(self, data):
        try:
            import statsmodels.api as sm
        except ImportError:
            msg = ('statsmodels >= 0.6.0 is required to perform '
                   'seasonal decomposition')
            raise ImportError(msg)

        # if not isinstance(data, pd.Series):
        #     raise ValueError('Input must be pd.Series')

        # ToDo:
        # - check DatetimeIndex and freq

        decomposed = sm.tsa.seasonal_decompose(data)
        resid = decomposed.resid

        self._decompose_indexer = pd.notnull(resid.values)
        resid = resid[self._decompose_indexer]
        return resid

    def _wrap_result(self, data, result):
        """ wrap result to be compat with data """

        if self.decompose:
            # pad NaN
            pad = np.zeros(len(self._original))
            pad[self._decompose_indexer] = result
            result = pad

        if isinstance(self._original, (pd.Series, pd.DataFrame)):
            index = self._original.index
            result = self._original._constructor(result, index=index)
        return result


class OutlierDetector(BaseDetector):
    pass


class ChangePointDetector(BaseDetector):
    pass


class SingleChangePointDetector(ChangePointDetector):

    def detect(self, data):
        data = self._validate(data)

        indexer, null, value = self.get_statistics(data)
        result = np.zeros(len(data))
        result[indexer] = 1
        result = self._wrap_result(data, result)
        return result
