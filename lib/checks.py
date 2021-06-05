import numpy as np
from tabulate import tabulate

def poisson_predictive_checks(pred, obs):
    """
    pred: [S,N]
    obs: [N]
    """
    obs_mean, obs_std = obs.mean(), obs.std()
    # [S], [S]
    pred_mean, pred_std = pred.mean(1), pred.std(1)
    pred_median, pred_mode = np.median(pred, 1), np.max(pred, 1)
    pred_skewness = pred_mean ** (-0.5)
    pred_kurtois = pred_mean ** (-1)
    headers = ['check', 'obs', 'pred (mean)', 'pred (std)']
    rows = [
        ['S', 1, pred.shape[0], None],
        ['size', obs.size, pred.size, None],
        ['mean', obs_mean, pred_mean.mean(), pred_mean.std()], 
        ['std', obs_std, pred_std.mean(), pred_std.std()],
        ['median', np.median(obs), pred_median.mean(), pred_median.std()],
        ['mode', np.max(obs), pred_mode.mean(), pred_mode.std()],
        ['skewness', obs_mean ** (-0.5), pred_skewness.mean(), pred_skewness.std()],
        ['kurtois', obs_mean ** (-1), pred_kurtois.mean(), pred_kurtois.std()]
    ]    
    for C in [0.25, 0.5, 0.75, 1., 2.]:
        mean_check = (np.abs(pred_mean - obs_mean) < C * pred_std)
        rows.append([f"mean within {C:.2f}*std", None, mean_check.mean(), mean_check.std()])
    return tabulate(rows, headers=headers)    
