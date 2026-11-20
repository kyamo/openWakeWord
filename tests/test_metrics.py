import importlib.util
import unittest
from pathlib import Path


def _get_false_positives():
    path = Path(__file__).resolve().parents[1] / "openwakeword" / "metrics.py"
    spec = importlib.util.spec_from_file_location("oww_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_false_positives


get_false_positives = _get_false_positives()


class FalsePositiveTest(unittest.TestCase):
    def test_a_run_counts_once_inside_the_window(self):
        self.assertEqual(get_false_positives([0, 1, 1, 1, 1], 0.5), 1)
        self.assertEqual(get_false_positives([0, 1], 0.5), 1)
        self.assertEqual(get_false_positives([1, 1, 1], 0.5), 1)
        self.assertEqual(get_false_positives([0, 1, 0, 1], 0.5, grouping_window=50), 1)

    def test_a_later_spike_counts_again(self):
        scores = [1] + [0] * 50 + [1]
        self.assertEqual(get_false_positives(scores, 0.5, grouping_window=50), 2)
        self.assertEqual(get_false_positives([0, 0, 0], 0.5), 0)


if __name__ == "__main__":
    unittest.main()


class RocCurveTest(unittest.TestCase):
    def test_roc_curves_accept_lists(self):
        from openwakeword.metrics import generate_roc_curve_fprs, generate_roc_curve_tprs
        tprs = generate_roc_curve_tprs([0.0, 0.5, 1.0], n_points=3)
        self.assertEqual(tprs, [2/3, 2/3, 1/3])
        fprs = generate_roc_curve_fprs([0.0]*45000 + [1.0], n_points=3)  # 1 hour of 80 ms frames
        self.assertEqual(len(fprs), 3)
        self.assertAlmostEqual(fprs[0], 1.0, places=2)
