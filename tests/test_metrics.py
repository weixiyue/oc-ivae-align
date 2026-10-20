"""Public HI RMSE checks; no network training or experiment outputs."""
import json
import unittest

import numpy as np
import pandas as pd

from utils.metrics import evaluate_all, rmse


class HIMetrics(unittest.TestCase):
    def setUp(self):
        # Four exact predictions and eight errors of 3: pooled RMSE = sqrt(6).
        self.frame = pd.DataFrame(dict(
            unit=[1] * 4 + [2] * 8,
            cycle=list(range(1, 5)) + list(range(1, 9)),
            hi=[4., 3., 2., 1.] + list(np.arange(11., 3., -1.)),
            hi_gt=[4., 3., 2., 1.] + list(np.arange(8., 0., -1.)),
            rul=[3., 2., 1., 0.] + list(np.arange(7., -1., -1.)),
        ))

    def test_rmse_pools_cycles_and_serializes(self):
        self.assertAlmostEqual(rmse(np.array([0., 2.]), np.zeros(2)), np.sqrt(2.))
        result = evaluate_all(self.frame)
        self.assertEqual(result['per_unit']['1']['rmse'], 0.)
        self.assertEqual(result['per_unit']['2']['rmse'], 3.)
        self.assertAlmostEqual(result['rmse'], np.sqrt(6.))
        self.assertNotAlmostEqual(result['rmse'], 1.5)  # equal-unit mean is wrong
        saved = json.loads(json.dumps(result, allow_nan=False))
        self.assertEqual(saved['rmse'], result['rmse'])
        self.assertEqual(saved['per_unit']['2']['rmse'], 3.)
        self.assertEqual(result['mae'], 2.)
        self.assertEqual(result['monotonicity'], 1.)
        self.assertEqual(result['trendability'], 1.)

    def test_missing_reference_is_null_with_reason(self):
        result = evaluate_all(self.frame.assign(hi_gt=np.nan))
        for values in [result, *result['per_unit'].values()]:
            for name in ('mae', 'rmse', 'mape'):
                self.assertIsNone(values[name])
                self.assertEqual(values['issues'][name], 'reference HI unavailable')
        json.dumps(result, allow_nan=False)

    def test_partial_reference_still_fails(self):
        frame = self.frame.copy()
        frame.loc[0, 'hi_gt'] = np.nan
        with self.assertRaisesRegex(ValueError, 'partial or nonfinite reference HI'):
            evaluate_all(frame)


if __name__ == '__main__':
    unittest.main()
