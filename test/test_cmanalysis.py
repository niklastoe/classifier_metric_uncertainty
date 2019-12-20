import pandas as pd
import unittest as ut

from workflows.bayesian_interpretation_confusion_matrix import ConfusionMatrixAnalyser, haldane_prior


class TestConfusionMatrixAnalyser(ut.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestConfusionMatrixAnalyser, self).__init__(*args, **kwargs)
        input_cm = pd.Series([9, 1, 3, 2], index=['TP', 'FN', 'TN', 'FP'])
        # use improper prior to avoid bias / simplifies calculation
        self.analyser = ConfusionMatrixAnalyser(input_cm, prior=haldane_prior)
        self.N = self.analyser.confusion_matrix.values.sum()

        sel_n = 100000
        inf_n_pp = self.analyser.posterior_predict_confusion_matrices(N=sel_n)
        inf_n_pp /= float(sel_n)
        self.inf_n_pp = inf_n_pp

    def test_theta_and_x_sampling(self):
        """confirm that sampled expected value/variance for theta and X are close to the analytical solution
        see https://en.wikipedia.org/wiki/Dirichlet_distribution
        and https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution"""

        alpha = self.analyser.prior + self.analyser.confusion_matrix
        alpha_0 = float(sum(alpha))
        dirichlet_mean = alpha / alpha_0
        dcm_mean = self.N * dirichlet_mean
        dirichlet_var = dirichlet_mean * (1 - dirichlet_mean) / (alpha_0 + 1)
        dcm_var = self.N * (self.N + alpha_0) * dirichlet_var

        for i in self.analyser.theta_samples:
            self.assertAlmostEqual(dirichlet_mean[i],
                                   self.analyser.theta_samples[i].mean(),
                                   delta=1e-2)
            self.assertAlmostEqual(dcm_mean[i],
                                   self.analyser.pp_samples[i].mean(),
                                   delta=5e-2)
            self.assertAlmostEqual(dirichlet_var[i],
                                   self.analyser.theta_samples[i].var(),
                                   delta=1e-3)
            self.assertAlmostEqual(dcm_var[i],
                                   self.analyser.pp_samples[i].var(),
                                   delta=2e-1)

    def test_expected_value(self):
        """confirm that expected value is equal to the metric for the original confusion matrix
        (within 1 percentage point)"""
        for metric in self.analyser.cm_metrics.index:
            self.assertAlmostEqual(self.analyser.cm_metrics[metric],
                                   self.analyser.theta_metrics.mean()[metric],
                                   delta=1e-2)

    def test_variance_convergence(self):
        """test that the variance of the posterior predictions of V_i/N converge towards the variance of theta_i"""

        theta_var = self.analyser.theta_samples.var()
        inf_n_pp_var = self.inf_n_pp.var()

        for i in theta_var.index:
            self.assertAlmostEqual(theta_var[i], inf_n_pp_var[i], delta=1e-5)

    def test_expected_value_pp_theta(self):
        """test that the expected value from the posterior prediction and theta are identical
        this only works for very large N"""

        for i in self.analyser.theta_samples.columns:
            self.assertAlmostEqual(self.analyser.theta_samples.mean()[i],
                                   self.inf_n_pp.mean()[i],
                                   delta=1e-4)

    def test_selected_metrics(self):
        """test if metrics are properly calculated, this is only done for a handful"""

        self.assertEqual(self.analyser.cm_metrics['ACC'], 12./15.)
        self.assertEqual(self.analyser.cm_metrics['PREVALENCE'], 10./15.)
        self.assertEqual(self.analyser.cm_metrics['TPR'], 9./10.)


if __name__ == '__main__':
    ut.main(verbosity=2)
