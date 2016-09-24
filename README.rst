pydetect
========

.. image:: https://readthedocs.org/projects/pydetect/badge/?version=latest
    :target: http://pydetect.readthedocs.org/en/latest/
    :alt: Latest Docs
.. image:: https://travis-ci.org/sinhrks/pydetect.svg?branch=master
    :target: https://travis-ci.org/sinhrks/pydetect
.. image:: https://coveralls.io/repos/sinhrks/pydetect/badge.svg?branch=master&service=github
    :target: https://coveralls.io/github/sinhrks/pydetect?branch=master

Change point and anomaly detections for time-series.
See `notebook <https://github.com/sinhrks/pydetect/tree/master/notebook>`_ to check basic usage.

Change point detection
----------------------

Mean or / and variance shift (at most one change)
"""""""""""""""""""""""""""""""""""""""""""""""""

- CUMSUM statistics

  - ``MeanDetector``
  - ``VarianceDetector``
  - ``MeanVarianceDetector``

Anomaly detection
-----------------

Generalized ESD Test
""""""""""""""""""""

- ``GESDDetector``

  See http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm
