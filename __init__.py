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

        print('Starting to evaluate model!')
        self.evaluate_model()
        print('Starting posterior prediction!')
        self.posterior_predictions()

        self.trace_metrics = self.calc_metrics(self.trace_samples)
        self.pp_metrics = self.calc_metrics(self.pp_samples)

        self.metrics = [x for x in self.trace_metrics.columns
                        if x not in self.confusion_matrix.index]

    def evaluate_model(self):

        y = self.confusion_matrix.values
        k = len(y)
        n = sum(y)

        with pm.Model() as multinom_test:
            a = pm.Dirichlet('a', a=np.ones(k))
            data_pred = pm.Multinomial('data_pred', n=n, p=a, observed=y)
            # use NUTS sampler
            trace = pm.sample(5000)

        passed_gelman_rub = (pm.diagnostics.gelman_rubin(trace)['a'] < 1.01).all()
        if not passed_gelman_rub:
            raise ValueError('Model did not converge according to Gelman-Rubin diagnostics!!')

        self.model = multinom_test
        self.data_pred = data_pred
        self.trace = trace
        self.trace_samples = pd.DataFrame(trace.get_values('a'), columns=self.confusion_matrix.index)

    def posterior_predictions(self):
        no_pp_samples = int(2e4)
        posterior_prediction = pm.sample_ppc(self.trace, samples=no_pp_samples, model=self.model)['data_pred']
        split_pp_samples = np.stack([posterior_prediction[:(no_pp_samples / 2)],
                                     posterior_prediction[(no_pp_samples / 2):]])

        passed_gelman_rubin = (pm.diagnostics.gelman_rubin(split_pp_samples) < 1.01).all()

        if not passed_gelman_rubin:
            raise ValueError('Not enough posterior predictive samples according to Gelman-Rubin diagnostics!!')

        self.pp_samples = pd.DataFrame(posterior_prediction, columns=self.confusion_matrix.index)

    def calc_metrics(self, input_df):
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

        df['ROC_approved'] = df['sensitivity'] > df['1-specificity']

        return df

    def chance_to_be_random_process(self):
        return 1 - (self.trace_metrics['MCC'] > 0).sum() / float(len(self.trace_metrics))

    def chance_to_appear_random_process(self):
        return 1 - (self.pp_metrics['MCC'] > 0).sum() / float(len(self.pp_metrics))

    @staticmethod
    def calc_hpd(dataseries, alpha=0.05):
        return pm.stats.hpd(dataseries, alpha=alpha)

    def plot_metric(self, metric):
        sns.distplot(self.trace_metrics[metric], label='trace')
        sns.distplot(self.pp_metrics[metric].dropna(), label='pp')
        plt.axvline(self.calc_metrics(self.confusion_matrix.astype(float))[metric], c='k', label='sample')
        plt.ylabel('Probability density')
        plt.legend()

    def interactive_metric_plot(self):
        metric_slider = ipywidgets.Dropdown(options=self.metrics, description='metric', value='MCC')
        ipywidgets.interact(self.plot_metric, metric=metric_slider)
