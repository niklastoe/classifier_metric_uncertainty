import numpy as np
import pandas as pd
import tqdm


def classifier_outperformance(a_metric_samples, b_metric_samples, margin=0.):
    """calculate the chance that a outperforms b by a given margin.
    Input: samples from the metrics for classifiers a and b"""
    greater = (a_metric_samples - margin) > b_metric_samples
    return greater.sum() / float(len(greater))


def monte_carlo_rank_classifiers(cma_list, metric, mc_sampling=10000):
    """create many rankings for classifiers to see how likely every rank is"""

    samples = []

    for i in cma_list:
        samples.append(i.theta_metrics[metric].sample(mc_sampling).values)

    no_models = len(cma_list)

    positions = []

    for idx, i in tqdm.tqdm_notebook(pd.DataFrame(samples).T.iterrows(), total=mc_sampling):
        curr_pos = pd.Series(np.arange(1, no_models + 1), index=i.sort_values()[::-1].index)
        positions.append(curr_pos)

    positions_df = pd.DataFrame(positions)

    position_probability_df = pd.DataFrame({x: positions_df[x].value_counts() / mc_sampling
                                            for x in positions_df.columns})

    if hasattr(cma_list, 'index'):
        position_probability_df.columns = cma_list.index

    return position_probability_df
