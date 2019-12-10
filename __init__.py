import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns
import ipywidgets


class ConfusionMatrixAnalyser(object):

    def __init__(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix

        self.theta_samples = self.sample_theta()
        self.pp_samples = self.posterior_predict_confusion_matrices()

        self.cm_metrics = self.calc_metrics(self.confusion_matrix.astype(float))
        self.theta_metrics = self.calc_metrics(self.theta_samples)
        self.pp_metrics = self.calc_metrics(self.pp_samples)

        self.metrics = [x for x in self.theta_metrics.columns
                        if x not in self.confusion_matrix.index]

    def sample_theta(self):

        dirichlet_alpha = 1 + self.confusion_matrix.values
        dirichlet_alpha = dirichlet_alpha.astype(int)

        distribution_samples = int(2e4)
        dirichlet_samples = np.random.dirichlet(dirichlet_alpha, size=distribution_samples)

        if not self.gelman_rubin_test_on_samples(dirichlet_samples):
            raise ValueError('Model did not converge according to Gelman-Rubin diagnostics!!')

        return pd.DataFrame(dirichlet_samples, columns=self.confusion_matrix.index)

    def posterior_predict_confusion_matrices(self, N=None):

        if N is None:
            N = self.confusion_matrix.values.sum()

        posterior_prediction = np.array([np.random.multinomial(N, x) for x in self.theta_samples.values])

        if not self.gelman_rubin_test_on_samples(posterior_prediction):
            raise ValueError('Not enough posterior predictive samples according to Gelman-Rubin diagnostics!!')

        return pd.DataFrame(posterior_prediction, columns=self.confusion_matrix.index)

    @staticmethod
    def gelman_rubin_test_on_samples(samples):
        no_samples = len(samples)
        split_samples = np.stack([samples[:int(no_samples / 2)],
                                  samples[int(no_samples / 2):]])
        passed_gelman_rubin = (pm.diagnostics.gelman_rubin(split_samples) < 1.01).all()
        return passed_gelman_rubin

    @staticmethod
    def calc_metrics(input_df):
        df = copy.deepcopy(input_df)

        n = df['TP'] + df['FN'] + df['TN'] + df['FP']
        if type(input_df) == pd.Series:
            n = float(n)

        df['prevalence'] = (df['TP'] + df['FN']) / n
        df['accuracy'] = (df['TP'] + df['TN']) / n

        df['sensitivity'] = df['TP'] / (df['TP'] + df['FN'])
        df['specificity'] = df['TN'] / (df['TN'] + df['FP'])
        df['precision'] = df['TP'] / (df['TP'] + df['FP'])
        df['NPV'] = df['TN'] / (df['TN'] + df['FN'])

        # for convenience, also add the other metrics
        df['FNR'] = 1 - df['sensitivity']
        df['FPR'] = 1 - df['specificity']
        df['1-specificity'] = df['FPR']
        df['FDR'] = 1 - df['precision']
        df['FOR'] = 1 - df['NPV']

        MCC_upper = (df['TP'] * df['TN'] - df['FP'] * df['FN'])
        MCC_lower = (df['TP'] + df['FP']) * (df['TP'] + df['FN']) * (df['TN'] + df['FP']) * (df['TN'] + df['FN'])
        df['MCC'] = MCC_upper / np.sqrt(MCC_lower)

        df['F1'] = 2 * (df['precision'] * df['sensitivity']) / (df['precision'] + df['sensitivity'])
        df['informedness'] = df['sensitivity'] + df['specificity'] - 1
        df['markedness'] = df['precision'] + df['NPV'] - 1

        return df

    def chance_to_be_random_process(self):
        return 1 - (self.theta_metrics['MCC'] > 0).sum() / float(len(self.theta_metrics))

    def chance_to_appear_random_process(self):
        return 1 - (self.pp_metrics['MCC'] > 0).sum() / float(len(self.pp_metrics))

    @staticmethod
    def calc_hpd(dataseries, alpha=0.05):
        return pm.stats.hpd(dataseries, alpha=alpha)

    def plot_metric(self,
                    metric,
                    show_theta_metric=True,
                    show_pp_metric=False,
                    show_sample_metric=True):
        if show_theta_metric:
            sns.distplot(self.theta_metrics[metric], label=r'from $\theta$', kde=False, bins=100)
        if show_pp_metric:
            sns.distplot(self.pp_metrics[metric].dropna(), label='pp', kde=False, bins=100)
        if show_sample_metric:
            plt.axvline(self.calc_metrics(self.confusion_matrix.astype(float))[metric], c='k', label='sample')
        plt.ylabel('Probability density')
        plt.yticks([])
        plt.legend(loc='upper left', bbox_to_anchor=(1., 1.))

        # ensure that x- and y-lim are always appropriate
        if metric in ['MCC', 'markdness', 'informedness']:
            plt.xlim(-1.05, 1.05)
        else:
            plt.xlim(-0.05, 1.05)

    def interactive_metric_plot(self):
        metric_slider = ipywidgets.Dropdown(options=self.metrics, description='metric', value='MCC')
        ipywidgets.interact(self.plot_metric, metric=metric_slider)

    def integrate_metric(self, metric, lower_boundary, upper_boundary):
        integral = ((self.theta_metrics[metric] > lower_boundary) & (self.theta_metrics[metric] < upper_boundary)).sum()
        integral = integral / float(len(self.theta_metrics))

        return integral
