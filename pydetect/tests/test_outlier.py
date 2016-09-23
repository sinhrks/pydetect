#!/usr/bin/env python

import pydetect
import pandas.util.testing as tm


class TestGeneralizedESD(tm.TestCase):

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
