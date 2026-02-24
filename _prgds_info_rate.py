import torch
import numpy as np
from tqdm.auto import tqdm
# from utils import info_rate, init_missing_data, mae, mre
from PRGDS_Interval_Tensor import PRGDS_tensor

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_recall_curve
import sys
import os
from path import Path
from scipy.stats import poisson

current_directory = os.getcwd()
root_directory = Path(current_directory).parent.parent.parent
sys.path.append(root_directory)

import utils

def evaluate_forecast_from_collections(forecast_collection, data_forecast, threshold=0.5, remove_self_loop=True):
    """
    forecast_collection: np.array, shape (num_iters, L)
        每次迭代的两步预测强度(λ)展平后的一维向量（L = steps * R * N * N）
    data_forecast: np.array or torch.Tensor, shape (steps, R, N, N)
        未来两步的 0/1 标签（或计数，非0将被视为1）
    threshold: float
        概率阈值，用于 F1@threshold / Recall@threshold
    remove_self_loop: bool
        是否去掉对角线 (i==j)

    返回：与你的 evaluate_link_prediction 相同结构的 dict（micro 评估）
    """
    # 1) numpy 化
    fc = np.asarray(forecast_collection)               # (num_iters, L)
    df = np.asarray(data_forecast)                     # (steps, R, N, N)
    steps, R, N, _ = df.shape

    # 2) 把每次迭代的 λ → 概率，再对迭代取均值（E[1-e^{-Λ}]）
    fc_probs_each = 1.0 - np.exp(-np.clip(fc, 0, None))   # (num_iters, L)
    fc_probs_mean = fc_probs_each.mean(axis=0)            # (L,)

    # 3) 还原形状为 (steps, R, N, N)，并转成 evaluate_link_prediction 需要的 (R, T, N, N)
    data_pred_TRNN = fc_probs_mean.reshape((steps, R, N, N))
    data_pred_RTNN = np.transpose(data_pred_TRNN, (1, 0, 2, 3))  # (R, T=steps, N, N)

    # 4) 标签转 0/1，并转成 (R, T, N, N)
    data_test_TRNN = (df > 0).astype(np.int32)
    data_test_RTNN = np.transpose(data_test_TRNN, (1, 0, 2, 3))   # (R, T, N, N)

    # 5) micro 评估（与 evaluate_link_prediction 等价实现）
    y_true, y_score = [], []
    for r in range(R):
        for t in range(steps):
            labels = data_test_RTNN[r, t].reshape(-1)
            scores = data_pred_RTNN[r, t].reshape(-1)

            if remove_self_loop:
                mask = np.ones_like(labels, dtype=bool)
                for i in range(N):
                    mask[i * N + i] = False
                labels = labels[mask]
                scores = scores[mask]

            y_true.append(labels)
            y_score.append(scores)

    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)

    # AUC-ROC / AUC-PR（micro：全部位置拼一起）
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc  = average_precision_score(y_true, y_score)

    # F1/Recall@threshold
    y_pred = (y_score >= threshold).astype(int)
    f1_at  = f1_score(y_true, y_pred)
    rec_at = recall_score(y_true, y_pred)

    # Best-F1（从 PR 曲线找最优阈值）
    precision, recall_curve, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precision * recall_curve / (precision + recall_curve + 1e-12)
    best_idx = np.argmax(f1_scores)
    best_thresh    = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1        = f1_scores[best_idx]
    best_recall    = recall_curve[best_idx]
    best_precision = precision[best_idx]

    return {
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        f"F1@{threshold}": f1_at,
        f"Recall@{threshold}": rec_at,
        "Best-F1": best_f1,
        "Best-threshold": best_thresh,
        "Best-Precision": best_precision,
        "Best-Recall": best_recall
    }

def load_data_from_RTNN(adjacency_4d, type='tensor', seed=None,
                        held_out_forecast_steps=1, held_out_smooth_percent=0.1):
    """
    adjacency_4d: numpy array with shape (R, T, N, N) or (T, R, N, N)
    returns: init_data, data_forecast, data_smooth, mask
    - 直接使用输入数据的全部 T，不再需要 t_slice
    - 最后 held_out_forecast_steps 个时间步作为 forecast，其余作为 smooth
    - 在 smooth 段随机挖 held_out_smooth_percent 的位置作为评估用缺失(mask)
    """
    if seed is not None:
        np.random.seed(seed)

    arr = np.asarray(adjacency_4d)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array (R,T,N,N) or (T,R,N,N), got shape {arr.shape}")

    # 统一时间在第 0 维：得到 (T, R, N, N)
    # 若原数据是 (R, T, N, N)，则转置到 (T, R, N, N)；若已是 (T, R, N, N) 则不动
    # 这里用一个简单启发判断：当第 0 维远小于第 1 维时，多半是 R 在前，需要转置
    # if arr.shape[0] < arr.shape[1]:
    #     data_TRNN = np.transpose(arr, (1, 0, 2, 3))  # (R,T,N,N) -> (T,R,N,N)
    # else:
    #     data_TRNN = arr

    data_TRNN = np.transpose(arr, (1, 0, 2, 3))

    # 确保为非负整数计数（Poisson 假设）
    if not np.issubdtype(data_TRNN.dtype, np.integer):
        data_TRNN = np.rint(np.clip(data_TRNN, 0, None)).astype(np.int32)
    else:
        data_TRNN = data_TRNN.astype(np.int32, copy=False)

    T = data_TRNN.shape[0]
    if held_out_forecast_steps <= 0 or held_out_forecast_steps >= T:
        raise ValueError(f"held_out_forecast_steps must be in [1, {T-1}], got {held_out_forecast_steps}")

    # 切分：前 T - held_out_forecast_steps 为 smooth，最后若干步为 forecast
    data_smooth   = data_TRNN[: T - held_out_forecast_steps]     # (T_smooth, R, N, N)
    data_forecast = data_TRNN[T - held_out_forecast_steps : ]    # (held_out_forecast_steps, R, N, N)


    # 在 smooth 段随机挖位置（用于补全评估）
    if not (0.0 < held_out_smooth_percent < 1.0):
        raise ValueError("held_out_smooth_percent must be in (0,1)")
    mask = (np.random.random(size=data_smooth.shape) < held_out_smooth_percent)

    # init_data：tensor 情况下直接返回原 smooth 数据（不先做 4D 插值）
    # 若需要 4D 版插值，可在此扩展
    init_data = data_smooth

    return init_data, data_forecast, data_smooth, mask
# load data
# def load_data(data_dir, type = 'matrix', seed = None):
#     if seed != None:
#         np.random.seed(seed)
#     held_out_forecast_steps = 2
#     held_out_smooth_percent = 0.1
#     org_data = np.load(data_dir)
#     org_data = org_data['data']
#     #org_data = torch.load(data_dir)
#     data_forecast = org_data[-held_out_forecast_steps:]
#     data_smooth = org_data[:-held_out_forecast_steps]
#     if type == 'matrix':
#         mask = (np.random.random(size = data_smooth.shape) < held_out_smooth_percent).astype(bool)
#         masked_train_data = np.ma.array(data_smooth, mask = mask)
#         init_data = np.ascontiguousarray(init_missing_data(masked_train_data))
#     else:
#         mask = (np.random.random(size = data_smooth.shape) < held_out_smooth_percent).astype(bool)
#         masked_train_data = np.ma.array(data_smooth, mask = mask)
#         init_data = data_smooth
#     return init_data, data_forecast, data_smooth, mask # Note that the shapes of init_data and mask are same with data_smooth

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

    model = PRGDS_tensor(**params)
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

        if model.deep == 1:
            # compute expectation of training data
            smooth_expectation = model.phi[0] @ torch.diag(model.lam) @ model.the.T @ torch.diag(model.delt)
            full_process_expectation_collection.append(smooth_expectation.clone())
        elif model.deep == 2:
            # compute expectation of training data
            part1 = torch.einsum('ik, jk -> ijk', model.phi[0], model.phi[1])
            part2 = torch.einsum('k, tk, t -> kt', model.lam, model.the, model.delt)
            smooth_expectation = torch.einsum('ijk, kt -> ijt', part1, part2)
            full_process_expectation_collection.append(smooth_expectation.clone())

        if iter+1 > burnin:
            # if (iter-burnin+1) % 10 == 0:
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
                    expectation_collection.append(smooth_expectation.clone())
                    # compute expectation of forecasting next two steps
                    forecast_expectation = forecast(model, deep = 3)
                    forecast_expectation = torch.stack(forecast_expectation).flatten()
                    forecast_collection.append(forecast_expectation.clone())
                else:
                    ...
                transition_collection.append(model.pi.clone())
    return model, full_process_expectation_collection, expectation_collection, forecast_collection, transition_collection

def forecast(model, deep=1):
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
    t = 1
    for s in range(t):
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
        obs_Fy = (np.arange(y) >= int(obs)).astype(int)

        drps = np.linalg.norm(est_Fy - obs_Fy) ** 2
        drps_collection.append(np.copy(drps))
    drps_mean = np.mean(drps_collection)
    return drps_mean

def KLE(smooth_expectation, smooth_truth_count):
    ratio = (smooth_truth_count + 0.01) / (smooth_expectation + 0.01)
    kle = smooth_truth_count * np.log(ratio) - smooth_truth_count + smooth_expectation 
    kle = np.array(kle)
    kle_sum = np.sum(kle) 
    kle_mean = np.mean(kle)
    return kle_mean

def main(data_dir, K, S, burin, maxiter, seed, type = 'tensor') -> None:

    R = 15
    data_name = "GDELT"   # "YAGO"  "WIKI"  "ICEWS18"  WIKI GDELT
    # train_path = f'data/{data_name}'+"/train.txt"
    train_path = f'data/{data_name}' + "/train_sliced.txt"
    val_path = f'data/{data_name}' + "/valid_sliced.txt"
    test_path = f'data/{data_name}' + "/test_sliced.txt"

    adjacency_4d, rel_old2new, T, N = utils.load_4d_array_from_txts(train_path, val_path, test_path, R)

    init_data, data_forecast, data_smooth, mask = load_data_from_RTNN(adjacency_4d, type = type, seed = seed)

    model, fp_expectation_collection, expectation_collection, forecast_collection, transition_collection = run_model(torch.tensor(init_data), K, S, burin, maxiter)
    mask = torch.tensor(mask)
    if type == 'matrix':
        smooth_expectation = [(sample.T)[mask] for sample in expectation_collection] # list(tensor)
    else:
        smooth_expectation = [torch.permute(sample, dims=(3,0,1,2))[mask] for sample in expectation_collection]

    data_smooth = np.array(data_smooth)
    smooth_truth_count = torch.tensor(
        data_smooth[mask].astype(np.float32))  # 1-dim tensor, length=d i.e. num of held out obs count data

    smooth_expectation = np.stack(smooth_expectation)  # stack along iteration
    forecast_collection = np.stack(forecast_collection)  # stack along iteration

    # ....
    df = np.asarray(data_forecast)  # shape: (steps, R, N, N) 或 (steps, V)
    u = np.unique(df)

    is_binary = np.all(np.isin(u, [0, 1]))  # True 表示只有 0 和 1
    print("data_forecast unique values:", u)
    print("is_binary:", is_binary)

    fc = np.asarray(forecast_collection)  # shape: (num_iters, L)
    # 把每次迭代的 λ 转成概率 p = 1 - e^{-λ}
    fc_probs_each = 1.0 - np.exp(-np.clip(fc, 0, None))  # 保证 λ≥0 再转换
    # 后验均值概率（对迭代取平均）
    fc_probs_mean = fc_probs_each.mean(axis=0)  # shape: (L,)

    # 如需还原成 (steps, R, N, N) 方便比对/画图：
    steps = data_forecast.shape[0]  # 通常是 2
    rest_shape = data_forecast.shape[1:]  # (R, N, N) 或 (V,)
    fc_probs_mean_reshaped = fc_probs_mean.reshape((steps,) + rest_shape)
    print("fc_probs_mean_reshaped:", fc_probs_mean.shape)

    print("fc_probs_mean in [min,max]:", fc_probs_mean.min(), fc_probs_mean.max())

    # 假设：
    # forecast_collection: (num_iters, L)，是你前面 np.stack 后的结果
    # data_forecast: (steps=2, R, N, N)
    res = evaluate_forecast_from_collections(forecast_collection, data_forecast, threshold=0.5, remove_self_loop=True)
    print(res)

    xlsx_path = "results/eval_mirco.xlsx"
    utils.append_eval_excel(xlsx_path, data_name, R, T, N, res, threshold=0.5, sheet_name="Eval")
    print(f"[EXCEL] appended one row to {xlsx_path} (sheet: Eval)")

    '''
    data_smooth = np.array(data_smooth)
    smooth_truth_count = torch.tensor(data_smooth[mask].astype(np.float32)) # 1-dim tensor, length=d i.e. num of held out obs count data

    smooth_expectation = np.stack(smooth_expectation) # stack along iteration
    forecast_collection = np.stack(forecast_collection) # stack along iteration

    # compute KLE-S
    smooth_expectation = np.mean(smooth_expectation ,axis = 0)
    smooth_kle = KLE(smooth_expectation, smooth_truth_count)
    print('KLE of smoothing data:', smooth_kle)

    # compute KLE-F
    forecast_collection = np.mean(forecast_collection, axis = 0)
    forecast_kle = KLE(forecast_collection, data_forecast.flatten())
    print('KLE of forecasting data:', forecast_kle)

    
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



if __name__ == '__main__':
    params = {
    'data_dir': 'data/us_death_preprocessed.npz',
    'K': 25,
    'S': 1,
    'burin': 60,
    'maxiter': 20,
    'type':'tensor',
    'seed': None
    }

    main(**params)

    # smooth_kle_coll = []
    # forecast_kle_coll = []
    # for i in range(3):
    #     _,_, smooth_kle, forecast_kle = main(**params)
    #     smooth_kle_coll.append(smooth_kle)
    #     forecast_kle_coll.append(forecast_kle)
    # print('Mean of smooth_kle:', np.mean(smooth_kle_coll))
    # print('Mean of forecast_kle:', np.mean(forecast_kle_coll))
    # print('Standard deviation of smooth_kle:', np.std(smooth_kle_coll))
    # print('Standard deviation of forecast_kle:', np.std(forecast_kle_coll))
