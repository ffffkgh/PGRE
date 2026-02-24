import torch
import numpy as np
from tqdm.auto import tqdm
from utils import info_rate, init_missing_data, mre, mae
from FS_PRGDS_Interval_Tensor import FS_PRGDS_tensor

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
    org_data = np.load(data_dir)
    org_data = org_data['data'][192:288]
    #org_data = torch.load(data_dir)
    data_forecast = org_data[-held_out_forecast_steps:]
    data_smooth = org_data[:-held_out_forecast_steps]

    if type == 'matrix':
        mask = (np.random.random(size = data_smooth.shape) < held_out_smooth_percent).astype(bool)
        masked_train_data = np.ma.array(data_smooth, mask = mask)
        init_data = np.ascontiguousarray(init_missing_data(masked_train_data))
    else:
        mask = (np.random.random(size = data_smooth.shape) < held_out_smooth_percent).astype(bool)
        masked_train_data = np.ma.array(data_smooth, mask = mask)
        init_data = data_smooth
    return init_data, data_forecast, data_smooth, mask # Note that the shapes of init_data and mask are same with data_smooth

# run model
def run_model(data, K, S, burnin=60, maxiter=20):
    burnin = burnin
    maxiter = maxiter 
    params = {
    'tau': 1.,
    'alpha0': 10.,
    'epsilon_the': 0.1,
    'epsilon_lam': 1.,
    'stationary': True,
    'data' : data,
    'K' : K, # latent components
    'S' : S, # sub-intervals
    'parallel' : False
    }

    model = FS_PRGDS_tensor(**params)
    full_process_expectation_collection = []
    expectation_collection = []
    forecast_collection = []
    transition_collection = []

    for iter in tqdm(range(burnin+maxiter)):
        model.sample_n()
        model.sample_h()
        model.sample_the()
        model.sample_lam()
        model.sample_delt()
        model.sample_g()
        model.sample_gam()
        model.sample_beta()
        model.sample_phi()
        model.sample_pi()
        model.sample_eta()
        model.sample_A()
        model.sample_dpgm()

        if model.deep == 1:
            # compute expectation of training data
            smooth_expectation = model.phi[0] @ torch.diag(model.lam) @ model.the.T @ torch.diag(model.delt)
            #smooth_expectation[smooth_expectation < 0.5] = 0
            full_process_expectation_collection.append(smooth_expectation.clone())
        elif model.deep == 2:
            # compute expectation of training data
            part1 = torch.einsum('ik, jk -> ijk', model.phi[0], model.phi[1])
            part2 = torch.einsum('k, tk, t -> kt', model.lam, model.the, model.delt)
            smooth_expectation = torch.einsum('ijk, kt -> ijt', part1, part2)
            ratio = (iter+1) / 65
            if ratio >=1 : ratio = 1
            mask = torch.rand_like(smooth_expectation) < ratio
            less_than_one = smooth_expectation < 1
            smooth_expectation[less_than_one & mask] = 0
            full_process_expectation_collection.append(smooth_expectation.clone())

        if iter+1 > burnin:
            #if (iter-burnin+1) % 10 == 0:
                if model.deep == 1:
                    expectation_collection.append(smooth_expectation.clone())
                    # compute expectation of forecasting next two steps
                    forecast_expectation = forecast(model)
                    forecast_expectation = torch.stack(forecast_expectation).flatten()
                    forecast_collection.append(forecast_expectation.clone())
                elif model.deep == 2:
                    # compute expectation of training data
                    part1 = torch.einsum('ik, jk -> ijk', model.phi[0], model.phi[1])
                    part2 = torch.einsum('k, tk, t -> kt', model.lam, model.the, model.delt)
                    smooth_expectation = torch.einsum('ijk, kt -> ijt', part1, part2)
                    expectation_collection.append(smooth_expectation.clone())
                    #compute expectation of forecasting next two steps
                    forecast_expectation = forecast(model, deep = 2)
                    forecast_expectation = torch.stack(forecast_expectation).flatten()
                    forecast_collection.append(forecast_expectation.clone())
                elif model.deep == 3:
                    # compute expectation of training data
                    part1 = torch.einsum('ik, jk, lk -> ijlk', model.phi[0], model.phi[1], model.phi[2])
                    part2 = torch.einsum('k, tk, t -> kt', model.lam, model.the, model.delt)
                    smooth_expectation = torch.einsum('ijlk, kt -> ijlt', part1, part2)
                    less_than_one = smooth_expectation < 1
                    smooth_expectation[less_than_one] = 0
                    expectation_collection.append(smooth_expectation.clone())
                    # compute expectation of forecasting next two steps
                    forecast_expectation = forecast(model, deep = 3)
                    forecast_expectation = torch.stack(forecast_expectation).flatten()
                    less_than_one = forecast_expectation < 1
                    forecast_expectation[less_than_one] = 0
                    forecast_collection.append(forecast_expectation.clone())
                else:
                    ...
                transition_collection.append(model.pi.clone())
    return model, full_process_expectation_collection, expectation_collection, forecast_collection, transition_collection

def forecast(model, deep = 1):
    theta = model.the
    pi = model.pi[-1]  # pi: S * K * K
    if deep == 1:
        phi = model.phi[0]
    else:
        phi = model.phi
    lam = model.lam
    epsi_the = model.epsi_the
    tau = model.tau
    delt = torch.mean(model.delt)
    forecast_expectation = []
    for s in range(2):
        if s == 0 :
            expectation_the = (epsi_the + (tau*(pi @ theta[-1,:].view(-1,1)))) / tau   
        else:
            expectation_the = (epsi_the + (tau*(pi @ texp))) / tau
        # 预测second step时需要用到first step的结果texp
        texp = expectation_the
        if deep == 1:
            expectation = delt * (phi @ torch.diag(lam) @ texp)
            forecast_expectation.append(expectation.clone())
        elif deep == 2:
            part1 = torch.einsum('ik, jk -> ijk', phi[0], phi[1])
            part2 = delt * torch.einsum('k, kt -> kt', lam, texp)
            expectation = torch.einsum('ijk, kt -> ijt', part1, part2)
            forecast_expectation.append(expectation.clone())
        elif deep == 3:
            part1 = torch.einsum('ik, jk, lk -> ijlk', phi[0], phi[1], phi[2])
            part2 = delt * torch.einsum('k, kt -> kt', lam, texp)
            expectation = torch.einsum('ijlk, kt -> ijlt', part1, part2)
            forecast_expectation.append(expectation.clone())
    return forecast_expectation

def DRPS(smooth_expectation, smooth_truth_count, y):
    drps_collection = []
    for est, obs in zip(smooth_expectation, smooth_truth_count):
        est_Fy = poisson.cdf(np.arange(y), est)
        #obs_Fy = (np.arange(y) >= int(obs)).astype(int)
        obs_Fy = poisson.cdf(np.arange(y), obs)

        drps = np.linalg.norm(est_Fy - obs_Fy) ** 2
        drps_collection.append(np.copy(drps))
    drps_mean = np.mean(drps_collection)
    return drps_mean

def KLE(smooth_expectation, smooth_truth_count):
    ratio = (smooth_truth_count) / (smooth_expectation)
    kle = smooth_truth_count * np.log(ratio) - smooth_truth_count + smooth_expectation 
    kle = np.array(kle)
    kle_sum = np.sum(kle) 
    kle_mean = np.mean(kle)
    return kle_mean

def main(data_dir, K, S, burin, maxiter, seed, type = 'matrix'):
    init_data, data_forecast, data_smooth, mask = load_data(data_dir, seed = seed, type = type)
    model, fp_expectation_collection, expectation_collection, forecast_collection, transition_collection = run_model(torch.tensor(init_data), K, S, burin, maxiter)
    mask = torch.tensor(mask)
    if type == 'matrix':
        smooth_expectation = [(sample.T)[mask] for sample in expectation_collection] # list(tensor)
    else:
        smooth_expectation = [torch.permute(sample, dims=(3,0,1,2))[mask] for sample in expectation_collection]

    data_smooth = np.array(data_smooth)
    smooth_truth_count = torch.tensor(data_smooth[mask].astype(np.float32)) # 1-dim tensor, length=d i.e. num of held out obs count data

    smooth_expectation = np.stack(smooth_expectation) # stack along iteration
    forecast_collection = np.stack(forecast_collection) # stack along iteration

    # compute KLE-S
    smooth_expectation = np.mean(smooth_expectation ,axis = 0)
    smooth_kle = KLE(smooth_expectation+0.01, smooth_truth_count+0.01)
    print('KLE of smoothing data:', smooth_kle)

    # compute KLE-F
    forecast_collection = np.mean(forecast_collection, axis = 0)
    forecast_kle = KLE(forecast_collection+0.01, data_forecast.flatten()+0.01)
    print('KLE of forecasting data:', forecast_kle)

    '''
    # compute DRPS of smooth
    smooth_expectation = np.mean(smooth_expectation, axis = 0)
    smooth_drps = DRPS(smooth_expectation, smooth_truth_count, y = 100)
    print('DRPS of smoothing data:', smooth_drps)

    # compute DRPS of forecast
    forecast_collection = np.mean(forecast_collection, axis = 0)
    forecast_drps = DRPS(forecast_collection, data_forecast.flatten(), y = 100)
    print('DRPS of forecasting data:', forecast_drps)

    # 1.compute MAE
    smooth_mae = mae(smooth_truth_count, np.mean(smooth_expectation, axis = 0))
    print('MAE of smoothing task based on everage results over steady period:', smooth_mae)
    # 2. compute MRE
    smooth_mre = mre(smooth_truth_count, np.mean(smooth_expectation, axis = 0))
    print('MRE of smoothing task based on average results over steady period:', smooth_mre)
    # 3. compute negative log likelihood of training data across full process
    nglogll_collection_train = []
    for exp in fp_expectation_collection:
        pmf = poisson.pmf(data_smooth, exp.T)
        nglogll = -np.log(np.mean(pmf.flatten(), axis = 0))
        nglogll_collection_train.append(np.copy(nglogll))
    # 4. compute negative log likelihood of held out data across steady process
    nglogll_collection_smooth = []
    for exp in smooth_expectation:
        pmf = poisson.pmf(smooth_truth_count, exp)
        nglogll = -np.log(np.mean(pmf, axis = 0))
        nglogll_collection_smooth.append(np.copy(nglogll))

    # 5. compute MAE for forecasting task
    forecast_mae = mae(data_forecast.flatten(), np.mean(forecast_collection, axis = 0) )
    print('MAE of forecasting task based on average results over steady period:', forecast_mae )
    # 6. compute MRE for forecasting task
    forecast_mre = mre(data_forecast.flatten(), np.mean(forecast_collection, axis = 0))
    print('MRE of forecasting task based on average results over steady period:', forecast_mre )
    # 7. compute negative log likelihood of forecasting data across steady process
    nglogll_collection_forecast = []
    for frc in forecast_collection:
        pmf = poisson.pmf(data_forecast.flatten(), frc) 
        nglogll = -np.log(np.mean(pmf, axis = 0))
        nglogll_collection_forecast.append(np.copy(nglogll))
    
    nglogll_collection = {'Training':nglogll_collection_train,
                          'Smooth':nglogll_collection_smooth,
                          'Forecast': nglogll_collection_forecast}
    '''
    return model, transition_collection, smooth_kle, forecast_kle#, nglogll_collection


if __name__ == '__main__':
    params = {
    'data_dir': 'data/icews_tensor_preprocessed_33610010013.npz',
    'K': 100,
    'S': 5,
    'burin': 40,
    'maxiter': 20,
    'seed': None,
    'type':'tensor'
    }

    smooth_kle_coll = []
    forecast_kle_coll = []
    for i in range(3):
        _,_, smooth_kle, forecast_kle = main(**params)
        smooth_kle_coll.append(smooth_kle)
        forecast_kle_coll.append(forecast_kle)
    print('Mean of smooth_kle:', np.mean(smooth_kle_coll))
    print('Mean of forecast_kle:', np.mean(forecast_kle_coll))
    print('Standard deviation of smooth_kle:', np.std(smooth_kle_coll))
    print('Standard deviation of forecast_kle:', np.std(forecast_kle_coll))


