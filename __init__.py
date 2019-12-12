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

# priors
max_entropy_prior = pd.Series([1] * len(symbol_order), index=symbol_order)
improper_prior = pd.Series([0] * len(symbol_order), index=symbol_order)


class ConfusionMatrixAnalyser(object):

    def __init__(self, confusion_matrix, prior=max_entropy_prior):
        self.confusion_matrix = confusion_matrix
        self.prior = prior
        self.metrics = get_metric_dictionary()

        self.theta_samples = self.sample_theta()
        self.pp_samples = self.posterior_predict_confusion_matrices()

        self.cm_metrics = self.calc_metrics(self.confusion_matrix.astype(float))
        self.theta_metrics = self.calc_metrics(self.theta_samples)
        self.pp_metrics = self.calc_metrics(self.pp_samples)

    def sample_theta(self):

        dirichlet_alpha = (self.prior + self.confusion_matrix)[self.confusion_matrix.index].values.astype(float)

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
                    show_sample_metric=True,
                    sel_ax=None):

        sel_ax = sel_ax or plt.gca()

        if show_theta_metric:
            sns.distplot(self.theta_metrics[metric], label=r'from $\theta$', kde=False, bins=100, ax=sel_ax)
        if show_pp_metric:
            sns.distplot(self.pp_metrics[metric].dropna(), label='pp', kde=False, bins=100, ax=sel_ax)
        if show_sample_metric:
            sel_ax.axvline(self.calc_metrics(self.confusion_matrix.astype(float))[metric], c='k', label='sample')
        sel_ax.set_ylabel('Probability density')
        sel_ax.set_yticks([])
        plt.legend(loc='upper left', bbox_to_anchor=(1., 1.))

        # ensure that x- and y-lim are always appropriate
        if metric in ['MCC', 'MK', 'BM']:
            sel_ax.set_xlim(-1.05, 1.05)
        else:
            sel_ax.set_xlim(-0.05, 1.05)

    def interactive_metric_plot(self):
        metric_slider = ipywidgets.Dropdown(options=self.metrics.index, description='metric', value='MCC')

        # interact will try to make sel_ax a slider, that's not possible: fix it
        ipywidgets.interact(self.plot_metric, metric=metric_slider, sel_ax=ipywidgets.fixed(None))

    def plot_all_metrics(self, show_theta_metric=True, show_pp_metric=False, show_sample_metric=True):
        fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(10, 10))

        for idx, metric in enumerate(self.metrics.index):
            curr_row = idx / 5
            curr_col = idx % 5
            curr_ax = axes[curr_col, curr_row]
            self.plot_metric(metric,
                             show_theta_metric=show_theta_metric,
                             show_pp_metric=show_pp_metric,
                             show_sample_metric=show_sample_metric,
                             sel_ax=curr_ax)
            curr_ax.set_yticks([])

        axes[2, 0].set_ylabel('Probability density')
        plt.subplots_adjust(hspace=0.5)

    def integrate_metric(self, metric, lower_boundary, upper_boundary):
        integral = ((self.theta_metrics[metric] > lower_boundary) & (self.theta_metrics[metric] < upper_boundary)).sum()
        integral = integral / float(len(self.theta_metrics))

        return integral


def get_metric_dictionary():
    metrics = {}

    metrics['PREVALENCE'] = (tp + fn) / n
    metrics['PREDICTIONPREVALENCE'] = (tp + fp) / n

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


def calculate_prior(metric1, val1, metric2, val2, metric3, val3, weight):
    sym_met = get_metric_dictionary()['symbolic']
    equation_system = [n - weight, sym_met[metric1] - val1, sym_met[metric2] - val2, sym_met[metric3] - val3]

    # use sympy to solve for cm_entries (which corresponds to the prior)
    cm_entries_values = sympy.nonlinsolve(equation_system, cm_elements)

    # indeces must be clear, properly format as pd.Series
    prior = pd.Series(cm_entries_values.args[0], index=symbol_order)
    return prior


class Prior(object):

    def __init__(self):
        self.metrics = get_metric_dictionary()

    @staticmethod
    def visualize_prior(metric1, val1, metric2, val2, metric3, val3, weight):
        curr_prior = calculate_prior(metric1, val1, metric2, val2, metric3, val3, weight)

        analyser = ConfusionMatrixAnalyser(curr_prior, prior=improper_prior)

        print(curr_prior)
        fig, axes = plt.subplots(ncols=3, sharey=True)

        for idx, metric in enumerate([metric1, metric2, metric3]):
            analyser.plot_metric(metric, show_sample_metric=False, sel_ax=axes[idx])
            if idx > 0:
                axes[idx].set_ylabel('')


    def interactive_prior_visualization(self):
        metric_slider1 = ipywidgets.Dropdown(options=self.metrics.index, description='metric1', value='ACC')
        metric_slider2 = ipywidgets.Dropdown(options=self.metrics.index, description='metric2', value='TPR')
        metric_slider3 = ipywidgets.Dropdown(options=self.metrics.index, description='metric3', value='PPV')

        val_slider1 = ipywidgets.FloatSlider(value=0.5, min=-1., max=1., step=0.1)
        val_slider2 = ipywidgets.FloatSlider(value=0.5, min=-1., max=1., step=0.1)
        val_slider3 = ipywidgets.FloatSlider(value=0.5, min=-1., max=1., step=0.1)

        weight_slider = ipywidgets.FloatSlider(value=10, min=1., max=100., step=1)

        ipywidgets.interact(self.visualize_prior,
                            metric1=metric_slider1,
                            metric2=metric_slider2,
                            metric3=metric_slider3,
                            val1=val_slider1,
                            val2=val_slider2,
                            val3=val_slider3,
                            weight=weight_slider)
