def classifier_outperformance(a_metric_samples, b_metric_samples, margin=0.):
    """calculate the chance that a outperforms b by a given margin.
    Input: samples from the metrics for classifiers a and b"""
    greater = (a_metric_samples - margin) > b_metric_samples
    return greater.sum() / float(len(greater))
