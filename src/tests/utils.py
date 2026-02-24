import torch
from scipy import special
from scipy.stats import poisson
import numpy as np

def info_rate(truth_count, expectation_value):
    '''
    Input:
        truth_count: observed count data, 1-dim (d)
        expectation_value: list(tensor) estimated value of count data computed by models, n tensors with length d

    Output:
        info_rate: models' information rate
    '''
    poisson_pmf = []
    for ev in expectation_value:
        assert len(ev) == len(truth_count)
        pp = poisson.pmf(truth_count, ev)
        poisson_pmf.append(np.copy(pp))
    mean_over_sample = np.log(np.mean(poisson_pmf, axis = 0) + 1e-32) # 1-dim array with length n (the number of held out counts)
    info_rate = - np.mean(mean_over_sample)
    return info_rate

def mae(truth, pred):
    return np.abs(truth-pred).mean()

def mre(truth, pred):
    return (np.abs(truth - pred) / (truth + 1)).mean()

def init_missing_data(masked_data):
    T, V = masked_data.shape
    Y_TV = masked_data.astype(np.int32).filled(fill_value=-1)
    lam = masked_data.mean()  # mean of the observed counts

    for v in range(V):
        if not masked_data.mask[:, v].any():
            continue

        if masked_data.mask[:, v].all():
            Y_TV[:, v] = int(np.round(lam))

        else:
            # time indices at which data for feature v is missing
            time_indices = list(np.where(masked_data.mask[:, v])[0])

            for t in time_indices:
                next_obs_y = None
                for s in range(t+1, T):  # look ahead for an observed val
                    if not masked_data.mask[s, v]:
                        next_obs_y = Y_TV[s, v]
                        break

                prev_obs_y = None
                for s in range(t-1, -1, -1):  # look behind for an observed val
                    if not masked_data.mask[s, v]:
                        prev_obs_y = Y_TV[s, v]
                        break

                if prev_obs_y is None:
                    if next_obs_y is None:
                        # this should only happen when there is only one missing val
                        assert (~masked_data.mask[:, v]).sum() == 1
                        lam_tv = lam
                    else:
                        lam_tv = next_obs_y
                else:
                    if next_obs_y is None:
                        lam_tv = prev_obs_y
                    else:
                        lam_tv = (next_obs_y + prev_obs_y) / 2.

                Y_TV[t, v] = int(np.round(lam_tv))
    assert (Y_TV >= 0).all()
    return Y_TV
