"""Latency fit (tok_exam.latency); the GPU part is not tested here."""
import unittest

from tok_exam.latency import fit_rate, predicted_seconds


class TestLatency(unittest.TestCase):
    def test_fit_recovers_line(self):
        rows = [{"generated": n, "seconds": 0.2 + 0.03 * n} for n in (10, 100, 500, 2000)]
        a, b = fit_rate(rows)
        self.assertAlmostEqual(a, 0.2, places=6)
        self.assertAlmostEqual(b, 0.03, places=8)

    def test_predicted_seconds(self):
        self.assertAlmostEqual(predicted_seconds(20, 1000, 0.2, 0.03), 20 * 0.2 + 30)


if __name__ == "__main__":
    unittest.main()
