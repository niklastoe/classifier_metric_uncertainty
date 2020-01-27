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

distribution_samples = int(2e4)
default_rope = 0.05


# priors (naming conventions from Alvares2018)
def objective_prior(val):
    return pd.Series([val] * len(symbol_order), index=symbol_order)


bayes_laplace_prior = objective_prior(1)
haldane_prior = objective_prior(0)
jeffreys_prior = objective_prior(0.5)
rda_prior = objective_prior(1. / len(symbol_order))
ha_prior = objective_prior(np.sqrt(2) / len(symbol_order))

dcm_priors = {'Bayes-Laplace': bayes_laplace_prior,
              'Haldane': haldane_prior,
              'Jeffreys': jeffreys_prior,
              'RDA': rda_prior,
              'HA': ha_prior}

triplebeta_priors = {'Haldane': {'PREVALENCE': [0, 0], 'TPR': [0, 0], 'TNR': [0, 0]},
                     'Bayes-Laplace': {'PREVALENCE': [1, 1], 'TPR': [1, 1], 'TNR': [1, 1]},
                     'Jeffreys': {'PREVALENCE': [0.5, 0.5], 'TPR': [0.5, 0.5], 'TNR': [0.5, 0.5]}}


class BetaBinomialDist(object):

    def __init__(self, k, j, prior=[0, 0]):
        self.k = k
        self.j = j
        self.n = k + j
        self.prior = prior
        self.theta_samples = self.sample_theta()
        self.pp_samples = self.posterior_predict_metric()

    def sample_theta(self):
        alpha = self.k + self.prior[0]
        beta = self.n - self.k + self.prior[1]
        return pd.Series(np.random.beta(alpha, beta, size=distribution_samples))

    def posterior_predict_metric(self):
        predicted_k = np.array([np.random.binomial(self.n, x) for x in self.theta_samples.values])
        predicted_j = self.n - predicted_k
        return pd.DataFrame({'k': predicted_k, 'j': predicted_j})[['k', 'j']]


class ConfusionMatrixAnalyser(object):

    def __init__(self,
                 confusion_matrix,
                 priors=triplebeta_priors['Haldane'],
                 fixed_prevalence=False,
                 posterior_predictions=True):
        self.confusion_matrix = confusion_matrix
        self.priors = priors
        self.n = float(self.confusion_matrix.values.sum())
        self.metrics = get_metric_dictionary()
        self.cm_metrics = self.calc_metrics(self.confusion_matrix.astype(float))

        if fixed_prevalence:
            self.prevalence = self.confusion_matrix[['TP', 'FN']].sum() / self.n
        else:
            self.prevalence = BetaBinomialDist(self.confusion_matrix[['TP', 'FN']].sum(),
                                               self.confusion_matrix[['TN', 'FP']].sum(),
                                               prior=self.priors['PREVALENCE']).theta_samples
        self.tpr = BetaBinomialDist(self.confusion_matrix['TP'],
                                    self.confusion_matrix['FN'],
                                    prior=self.priors['TPR']).theta_samples
        self.tnr = BetaBinomialDist(self.confusion_matrix['TN'],
                                    self.confusion_matrix['FP'],
                                    prior=self.priors['TNR']).theta_samples

        self.theta_samples = self.sample_theta()
        self.theta_metrics = self.calc_metrics(self.theta_samples)

        # if we just want to look at the priors, we don't want posterior predictions because self.n=0
        if posterior_predictions:
            self.pp_samples = self.posterior_predict_confusion_matrices()
            self.pp_metrics = self.calc_metrics(self.pp_samples)

    def sample_theta(self):

        tp_samples = self.prevalence * self.tpr
        fn_samples = self.prevalence * (1 - self.tpr)
        tn_samples = (1 - self.prevalence) * self.tnr
        fp_samples = (1 - self.prevalence) * (1 - self.tnr)

        theta_samples = pd.DataFrame({'TP': tp_samples,
                                      'FN': fn_samples,
                                      'TN': tn_samples,
                                      'FP': fp_samples})

        if not self.gelman_rubin_test_on_samples(theta_samples.values):
            raise ValueError('Model did not converge according to Gelman-Rubin diagnostics!!')

        return theta_samples

    def posterior_predict_confusion_matrices(self, pp_n=None):

        if pp_n is None:
            pp_n = self.n

        posterior_prediction = np.array([np.random.multinomial(pp_n, x) for x in self.theta_samples.values])

        if not self.gelman_rubin_test_on_samples(posterior_prediction):
            raise ValueError('Not enough posterior predictive samples according to Gelman-Rubin diagnostics!!')

        return pd.DataFrame(posterior_prediction, columns=self.theta_samples.columns)

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

    @staticmethod
    def chance_to_be_in_interval(metric_samples, low=-np.inf, high=np.inf):
        count = (metric_samples > low).astype(int) + (metric_samples < high).astype(int) == 2
        return count.sum() / float(len(metric_samples))

    def chance_to_be_random_process(self, rope=default_rope):
        return self.chance_to_be_in_interval(self.theta_metrics['MCC'], low=-rope, high=rope)

    def chance_to_be_harmful(self, rope=default_rope):
        return self.chance_to_be_in_interval(self.theta_metrics['MCC'], high=-rope)

    def chance_to_appear_random_process(self, rope=default_rope):
        return self.chance_to_be_in_interval(self.pp_metrics['MCC'], low=-rope, high=rope)

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


class NewPrevalence(ConfusionMatrixAnalyser):
    """analyse the impact of prevalence on different metrics.
    TPR and TNR are independent of prevalence and can be copied from instance of ConfusionMatrixAnalyser.
    Use different prevalence distribution and recalculate all other metrics."""

    def __init__(self,
                 cma,
                 prevalence=BetaBinomialDist(0, 0,
                                             prior=triplebeta_priors['Bayes-Laplace']['PREVALENCE'])):

        self.prevalence = prevalence.theta_samples
        self.tpr = cma.tpr
        self.tnr = cma.tnr

        self.metrics = get_metric_dictionary()

        # use metrics from original ConfusionMatrixAnalyser, otherwise there will be errors that they are missing
        # keep in mind that the original confusion matrix has a different prevalence
        self.confusion_matrix = cma.confusion_matrix
        self.cm_metrics = self.calc_metrics(self.confusion_matrix.astype(float))

        self.theta_samples = self.sample_theta()
        self.theta_metrics = self.calc_metrics(self.theta_samples)


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

    mcc_upper = (tp * tn - fp * fn)
    mcc_lower = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    metrics['MCC'] = mcc_upper / sympy.sqrt(mcc_lower)

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

        analyser = ConfusionMatrixAnalyser(curr_prior, priors=triplebeta_priors['Haldane'])

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
