import torch
import numpy as np
from tqdm.auto import tqdm
from utils import info_rate, init_missing_data, mae, mre
from GS_PGDS_Interval_Tensor import GS_PGDS_tensor

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
    if type == 'matrix':
        org_data = org_data['data'][:,1900:2000]
        data_forecast = org_data[-held_out_forecast_steps:]
        data_smooth = org_data[:-held_out_forecast_steps]

        mask = (np.random.random(size = data_smooth.shape) < held_out_smooth_percent).astype(bool)
        masked_train_data = np.ma.array(data_smooth, mask = mask)
        init_data = np.ascontiguousarray(init_missing_data(masked_train_data))
    else:
        org_data = org_data['data']
        data_forecast = org_data[-held_out_forecast_steps:]
        data_smooth = org_data[:-held_out_forecast_steps]

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
    'stationary': True,
    'data' : data,
    'K' : K, # latent components
    'S' : S, # sub-intervals
    'parallel' : True
    }

    model = GS_PGDS_tensor(**params)
    full_process_expectation_collection = []
    expectation_collection = []
    forecast_collection = []
    transition_collection = []

    for iter in tqdm(range(burnin+maxiter)):
        model.sample_n()
        model.sample_l()
        model.sample_the()
        model.sample_delt()
        model.sample_phi()
        model.sample_pi()
        model.sample_eta()
        model.sample_tilde_lskk()
        model.sample_Z()
        model.sample_D()
        model.sample_dpgm()

        if model.deep == 1:
            # compute expectation of training dadta
            smooth_expectation = model.phi[0] @ model.the.T @ torch.diag(model.delt)
            full_process_expectation_collection.append(smooth_expectation.clone())
        elif model.deep == 2:
            # compute expectation of training data
            part1 = torch.einsum('ik, jk -> ijk', model.phi[0], model.phi[1])
            part2 = torch.einsum('tk, t -> kt', model.the, model.delt)
            smooth_expectation = torch.einsum('ijk, kt -> ijt', part1, part2)
            ratio = (iter+1) / 30
            if ratio >=0.5 : ratio = 0.5
            mask = torch.rand_like(smooth_expectation) < ratio
            less_than_one = smooth_expectation < 1
            smooth_expectation[less_than_one & mask] = 0
            full_process_expectation_collection.append(smooth_expectation.clone())

        if iter+1 > burnin:
            if (iter-burnin+1) % 10 == 0:
                if model.deep == 1:
                    expectation_collection.append(smooth_expectation.clone())
                    # compute expectation of forecasting next two steps
                    forecast_expectation = forecast(model)
                    forecast_expectation = torch.stack(forecast_expectation).flatten()
                    forecast_collection.append(forecast_expectation.clone())
                elif model.deep == 2:
                    # compute expectation of training data
                    part1 = torch.einsum('ik, jk -> ijk', model.phi[0], model.phi[1])
                    part2 = torch.einsum('tk, t -> kt', model.the, model.delt)
                    smooth_expectation = torch.einsum('ijk, kt -> ijt', part1, part2)
                    expectation_collection.append(smooth_expectation.clone())
                    #compute expectation of forecasting next two steps
                    forecast_expectation = forecast(model, deep = 2)
                    forecast_expectation = torch.stack(forecast_expectation).flatten()
                    forecast_collection.append(forecast_expectation.clone())
                elif model.deep == 3:
                    # compute expectation of training data
                    part1 = torch.einsum('ik, jk, lk -> ijlk', model.phi[0], model.phi[1], model.phi[2])
                    part2 = torch.einsum('tk, t -> kt', model.the, model.delt)
                    smooth_expectation = torch.einsum('ijlk, kt -> ijlt', part1, part2)
                    expectation_collection.append(smooth_expectation.clone())
                    # compute expectation of forecasting next two steps
                    forecast_expectation = forecast(model, deep = 3)
                    forecast_expectation = torch.stack(forecast_expectation).flatten()
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
    delt = torch.mean(model.delt)
    forecast_expectation = []
    for s in range(2):
        if s == 0 :
            expectation_the = pi @ theta[-1,:].view(-1,1) 
        else:
            expectation_the = pi @ texp
        # 预测second step时需要用到first step的结果texp
        texp = expectation_the
        if deep == 1:
            expectation = delt * (phi @ texp)
            forecast_expectation.append(expectation.clone())
        elif deep == 2:
            part1 = torch.einsum('ik, jk -> ijk', phi[0], phi[1])
            expectation = torch.einsum('ijk, kt -> ijt', part1, texp)
            forecast_expectation.append(expectation.clone())
        elif deep == 3:
            part1 = torch.einsum('ik, jk, lk -> ijlk', phi[0], phi[1], phi[2])
            expectation = torch.einsum('ijlk, kt -> ijlt', part1, texp)
            forecast_expectation.append(expectation.clone())
    return forecast_expectation

def main(data_dir, K, S, burin, maxiter, seed, type = 'matrix') -> None:
    init_data, data_forecast, data_smooth, mask = load_data(data_dir, type = type, seed = seed)
    model, fp_expectation_collection, expectation_collection, forecast_collection, transition_collection = run_model(torch.tensor(init_data), K, S, burin, maxiter)
    mask = torch.tensor(mask)
    if type == 'matrix':
        smooth_expectation = [(sample.T)[mask] for sample in expectation_collection] # list(tensor)
    else:
        smooth_expectation = [torch.permute(sample, dims=(2,0,1))[mask] for sample in expectation_collection]
    smooth_truth_count = torch.tensor(data_smooth[mask].astype(np.float32)) # 1-dim tensor, length=d i.e. num of held out obs count data

    smooth_expectation = np.stack(smooth_expectation) # stack along iteration
    forecast_collection = np.stack(forecast_collection) # stack along iteration

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

    return model, transition_collection, nglogll_collection


if __name__ == '__main__':
    params = {
    'data_dir': 'data/icews_preprocessed.npz',
    'K': 100,
    'S': 1,
    'burin': 100,
    'maxiter': 200,
    'seed': 1234
    }

    main(**params)


