import unittest

from compare_localization_runs import paired_bootstrap


class PairedBootstrapTest(unittest.TestCase):
    def test_constant_delta_has_exact_interval(self):
        baseline = {
            str(i): {"image_id": str(i), "score": str(float(i))}
            for i in range(5)
        }
        candidate = {
            str(i): {"image_id": str(i), "score": str(float(i) + 0.25)}
            for i in range(5)
        }
        result = paired_bootstrap(
            baseline, candidate, ["score"], samples=200, seed=0
        )[0]
        self.assertAlmostEqual(result["delta"], 0.25)
        self.assertAlmostEqual(result["paired_ci95_low"], 0.25)
        self.assertAlmostEqual(result["paired_ci95_high"], 0.25)

    def test_requires_identical_image_sets(self):
        baseline = {"1": {"image_id": "1", "score": "0"}}
        candidate = {"2": {"image_id": "2", "score": "1"}}
        with self.assertRaises(ValueError):
            paired_bootstrap(baseline, candidate, ["score"], samples=10, seed=0)


if __name__ == "__main__":
    unittest.main()
