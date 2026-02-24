import models
import model_with_no_relation
import utils
import test_micro

import numpy as np
import torch

from collections import Counter


def main():

    data_name = "GDELT"   # "YAGO"8  "WIKI"15  "ICEWS18"10  WIKI GDELT

    burnin_epochs = 40
    collection_epochs = 20
    R = 8

    # train_path = f'data/{data_name}'+"/train.txt"
    train_path = f'data/{data_name}' + "/train_sliced.txt"
    val_path = f'data/{data_name}'+"/valid_sliced.txt"
    test_path = f'data/{data_name}'+"/test_sliced.txt"


    adjacency_4d, rel_old2new, T, N = utils.load_4d_array_from_txts(train_path, val_path, test_path, R)
    print(adjacency_4d.shape, "T=", T, "N=", N)
    print("Top-R relations selected:", len(rel_old2new))

    # print("4D 数组形状:", adjacency_4d.shape)  # (R, T, N, N)
    print("某个关系/时间的非零数目:", (adjacency_4d[0, 0] > 0).sum())


    adj_train = adjacency_4d[:, :T - 1, :, :]  # (R, T-1, N, N)
    adj_test = adjacency_4d[:, T - 1:T, :, :]  # (R, 1,   N, N) 保持 T 维度为 1

    print("data_train:", adj_train.shape)  # (R, T-1, N, N)
    print("data_test :", adj_test.shape)  # (R, 1,   N, N)

    data_train = torch.tensor(adj_train)
    data_test = torch.tensor(adj_test)

    # 训练数据集
    model = model_with_no_relation.PGRE_nor(data_train)
    # print(model)
    delta, phi, psi, pi = model.model_train(burnin_epochs, collection_epochs)

    # 存储模型
    save_path = f"load_model/trained_model_{data_name}_R_{R}_{burnin_epochs}_{collection_epochs}.pth"
    # 将模型参数保存为字典
    torch.save({
        'delta': delta,  # 保存delta
        'phi': phi,  # 保存phi
        'psi': psi,  # 保存psi
        'pi': pi  # 保存pi
    }, save_path)


    # ==================测 试========================
    load_path = f"load_model/trained_model_{data_name}_R_{R}_{burnin_epochs}_{collection_epochs}.pth"
    checkpoint = torch.load(load_path)

    model_load = model_with_no_relation.PGRE_nor(data_test)  # 测试集评估
    model_load.delta_tr_m = checkpoint['delta']
    model_load.phi_nm = checkpoint['phi']
    model_load.psi_nm = checkpoint['psi']
    model_load.pi_rr = checkpoint['pi']

    # 生成 data_pred
    data_pred = torch.zeros(model_load.R, model_load.T, model_load.N, model_load.N)
    delta_next = model_load.delta_tr_m[-1]
    for r in range(model_load.R):
        for t in range(model_load.T):
            Prob = torch.einsum('m,im,jm->ij', delta_next[r], model_load.phi_nm, model_load.psi_nm) + model_load.eps
            data_pred[r, t] = 1 - torch.exp(-Prob)

    print("data_test", data_test.shape)
    print("data_pred shape:", data_pred.shape)

    results = test_micro.evaluate_link_prediction(data_pred, data_test, threshold=0.5)
    print(results)


if __name__ == "__main__":
    main()
