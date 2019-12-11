import pandas as pd
import unittest as ut

from workflows.bayesian_interpretation_confusion_matrix import calculate_prior


class TestPriorDetermination(ut.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestPriorDetermination, self).__init__(*args, **kwargs)

    def test_caelen_example(self):
        """Caelen2017 (p. 438) proposed how to determine prior from metrics, confirm that the results are identical"""
        caelen_prior = pd.Series({'TP': 0.3319149, 'TN': 0.2680851, 'FP': 0.2212766, 'FN': 0.1787234})
        prior_from_func = calculate_prior('ACC', 0.6, 'PPV', 0.6, 'TPR', 0.65, 1)

        for cm_entry in caelen_prior.index:
            self.assertAlmostEqual(caelen_prior[cm_entry], prior_from_func[cm_entry], delta=1e-7)


if __name__ == '__main__':
    ut.main(verbosity=2)
