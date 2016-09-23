#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd


# ToDo: maybe needs generic detector and time series detector?


class BaseDetector(object):

    def detect(self, data):
        raise NotImplementedError

    def get_statistics(self, data):
        raise NotImplementedError

    def _validate(self, data):
        """ validate data """
        if not isinstance(data, pd.Series):
            raise ValueError('Input must be pd.Series')

        # ToDo:
        # - check DatetimeIndex and freq

        return data

    def _wrap_result(self, data, result):
        """ wrap result to be compat with data """

        result = data._constructor(result, index=data.index)
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
