import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns
import sympy
import ipywidgets


# sympy symbol definition for confusion matrix (CM) entries
symbol_order = 'TP FN TN FP'.split()
tp, fn, tn, fp = cm_elements = sympy.symbols(symbol_order)
n = sum(cm_elements)


class ConfusionMatrixAnalyser(object):

    def __init__(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix
        self.metrics = get_metric_dictionary()

        self.theta_samples = self.sample_theta()
        self.pp_samples = self.posterior_predict_confusion_matrices()

        self.cm_metrics = self.calc_metrics(self.confusion_matrix.astype(float))
        self.theta_metrics = self.calc_metrics(self.theta_samples)
        self.pp_metrics = self.calc_metrics(self.pp_samples)

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

    def calc_metrics(self, samples):
        metrics_numpy_functions = self.metrics['numpy']

        # pass samples to lambdified functions of metrics
        # important to keep the order of samples consistent with definition: TP, FN, TN, FP
        metrics_dict = {x: metrics_numpy_functions[x](*samples[symbol_order].values.T)
                        for x in metrics_numpy_functions.index}

        # store in pandas for usability
        if type(samples) == pd.DataFrame:
            metrics = pd.DataFrame(metrics_dict)
        else:
            metrics = pd.Series(metrics_dict)

        return metrics

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
        metric_slider = ipywidgets.Dropdown(options=self.metrics.index, description='metric', value='MCC')
        ipywidgets.interact(self.plot_metric, metric=metric_slider)

    def integrate_metric(self, metric, lower_boundary, upper_boundary):
        integral = ((self.theta_metrics[metric] > lower_boundary) & (self.theta_metrics[metric] < upper_boundary)).sum()
        integral = integral / float(len(self.theta_metrics))

        return integral


def get_metric_dictionary():
    metrics = {}

    metrics['PREVALENCE'] = (tp + fn) / n

    metrics['TPR'] = tpr = tp / (tp + fn)
    metrics['TNR'] = tnr = tn / (tn + fp)
    metrics['PPV'] = ppv = tp / (tp + fp)
    metrics['NPV'] = npv = tn / (tn + fn)
    metrics['FNR'] = 1 - tpr
    metrics['FPR'] = 1 - tnr
    metrics['FDR'] = 1 - ppv
    metrics['FOR'] = 1 - npv

    metrics['ACC'] = (tp + tn) / n

    MCC_upper = (tp * tn - fp * fn)
    MCC_lower = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    metrics['MCC'] = MCC_upper / sympy.sqrt(MCC_lower)

    metrics['F1'] = 2 * (ppv * tpr) / (ppv + tpr)
    metrics['BM'] = tpr + tnr - 1
    metrics['MK'] = ppv + npv - 1

    numpy_metrics = {x: sympy.lambdify(cm_elements, metrics[x], "numpy") for x in metrics}

    metrics_df = pd.DataFrame({'symbolic': metrics, 'numpy': numpy_metrics})

    return metrics_df
