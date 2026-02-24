import torch
import numpy as np
import numpy.random as rn
import scipy.stats as st
from sklearn.metrics import precision_score, recall_score

from BPTF import BPTF
from  utils_bptf import parafac
from utils import info_rate, init_missing_data, mae, mre
import sys
import os
from path import Path
from scipy.stats import poisson
current_directory = os.getcwd()
root_directory = Path(current_directory).parent.parent.parent
sys.path.append(root_directory)

# load data
def load_data(data_dir, type = 'matrix', seed = None):
    if seed != None:
        np.random.seed(seed)
    held_out_forecast_steps = 2
    held_out_smooth_percent = 0.1
    #org_data = np.load(data_dir)
    #org_data = org_data['data'][240:]
    org_data = torch.load(data_dir)
    org_data = np.array(org_data)
    data_forecast = org_data[-held_out_forecast_steps:, :]
    data_smooth = org_data[:-held_out_forecast_steps, :]

    mask = (np.random.random(size = data_smooth.shape) < held_out_smooth_percent).astype(bool)
    if type == 'matrix':
        masked_train_data = np.ma.array(data_smooth, mask = mask)
        init_data = np.ascontiguousarray(init_missing_data(masked_train_data))
    else:
        init_data = data_smooth
    return init_data, data_forecast, data_smooth, mask # Note that the shapes of init_data and mask are same with data_smooth

def run_model(data, K, alpha, maxiter):
    bptf = BPTF(n_modes=data.ndim,
                n_components=K,
                max_iter = maxiter,
                tol = 1e-4,
                smoothness = 100,
                verbose = False,
                alpha = alpha,
                debug = False)
    bptf.fit(data)
    rc_data = bptf.reconstruct() # estimation value
    return rc_data

def DRPS(smooth_expectation, smooth_truth_count, y):
    drps_collection = []
    for est, obs in zip(smooth_expectation, smooth_truth_count):
        est_Fy = poisson.cdf(np.arange(y), est)
        obs_Fy = (np.arange(y) >= int(obs)).astype(int)

        drps = np.linalg.norm(est_Fy - obs_Fy) ** 2
        drps_collection.append(np.copy(drps))
    drps_mean = np.mean(drps_collection)
    return drps_mean

def main(data_dir, K, alpha, maxiter, type = 'matrix', seed = None) -> None:
    init_data, data_forecast, data_smooth, mask = load_data(data_dir, type = type, seed = seed)
    rc_data = run_model(init_data, K, alpha, maxiter = maxiter)
    smooth_truth_count = torch.tensor(data_smooth[mask].astype(np.float32))
    #smooth_expectation = np.stack(smooth_expectation) # stack along iteration
    smooth_expectation = rc_data[mask]
    #forecast_collection = np.stack(forecast_collection) # stack along iteration

    # compute DRPS of smooth
    smooth_drps = DRPS(smooth_expectation, smooth_truth_count, y = 100)
    print('DRPS of smoothing data:', smooth_drps)

    '''
    # 1. compute MAE
    smooth_mae = mae(smooth_truth_count, smooth_expectation)
    print('MAE of smoothing task based on average results over steady period:', smooth_mae )
    # 2. compute MRE
    smooth_mre = mre(smooth_truth_count, smooth_expectation)
    print('MRE of smoothing task based on average results over steady period:', smooth_mre )

    #bptf_smooth_ir = info_rate(smooth_truth_count, [rc_data[mask]])
    #print('Information rate of BPTF for smooth data:', bptf_smooth_ir)

    #data_forecast = data_forecast.flatten()
    #bptf_forecast = np.tile(rc_data[-1,:], (2,1))
    #bptf_forecast = bptf_forecast.flatten()
    #bptf_forecast_ir = info_rate(data_forecast, [bptf_forecast])
    #print('Information rate of BPTF for forecast data:', bptf_forecast_ir)
    '''
    return smooth_drps


if __name__ == '__main__':
    params = {
    'data_dir': 'data/email(dept1)-matrix-WST-115_1938.pt',
    'K': 100,
    'alpha': 100,
    'maxiter': 100,
    'type':'tensor',
    'seed': None
    }

    smooth_drps_coll = []
    for i in range(3):
        smooth_drps = main(**params)
        smooth_drps_coll.append(smooth_drps)

    print('Mean of smooth_drps:', np.mean(smooth_drps_coll))
    print('Standard deviation of smooth_drps:', np.std(smooth_drps_coll))