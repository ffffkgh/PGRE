import torch
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import psutil

from tqdm.auto import tqdm
from dataclasses import dataclass
from torch.distributions import Gamma, Multinomial, Dirichlet, Beta, Binomial, Poisson, Bernoulli

def CRT(n, a):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)  # 确保 a 是张量
    max_x = torch.max(n)
    rates = torch.arange(0, max_x).expand(*n.size(), -1)
    rates = a.unsqueeze(-1) / (a.unsqueeze(-1) + rates)
    bers = torch.bernoulli(rates)
    mask = (torch.arange(0, max_x).expand(*n.size(), -1) < n.unsqueeze(-1)).to(torch.int)
    bers = bers * mask
    return torch.sum(bers, dim=-1)

def Po_plus(rate):
    r1 = rate[rate>=1]
    r2 = rate[rate<1]
    m = torch.zeros_like(rate)
    m1 = torch.zeros_like(r1)
    m2 = torch.zeros_like(r2)

    while True:
        dex = (m1 == 0).nonzero(as_tuple=True)
        if dex[0].numel() == 0:
            break
        else:
            r_dex = r1[dex]
            temp = torch.poisson(r_dex)
            idex = temp > 0
            m1[dex] = torch.where(idex, temp, m1[dex])
    m[rate>=1] = m1

    while True:
        dex = (m2 == 0).nonzero(as_tuple=True)
        if dex[0].numel() == 0:
            break
        else:
            r_dex = r2[dex]
            temp = 1 + torch.poisson(r_dex)
            idex = torch.rand_like(temp) < (1 / temp)
            m2[dex] = torch.where(idex, temp, m2[dex])
    m[rate<1] = m2

    return m

# 定义一个装饰器函数，计算执行某个函数需要的时间，在需要计时的函数前加@timing_decorator来调用
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper


class DKG_PG():
    def __init__(self, data):

        self.eps = 1e-16
        self.max = 1e16

        # 超参
        self.a_0 = 1.  # phi
        self.a_1 = 1.  # psi
        self.g_0 = 1.  # ci
        self.f_0 = 1.  # ci
        self.tau = 1.  # delta
        self.b_0 = 0.1  # ksi
        self.e_0 = 0.1  # ksi
        self.beta = 1.  # upsilon
        self.gamma_0 = 50.  # upsilon
        self.d_0 = 1.  # beta
        self.h_0 = 1.  # beta

        self.data = data

        self.R = len(data)
        self.T = len(data[0])
        self.N = data[0][0].shape[0]
        self.K = int(self.N / 2)
        # self.M = int(self.R * self.K)
        # self.M = int(self.N / 2)
        self.M = 200

        self.x_tr_dotdotm = torch.zeros(self.T, self.R, self.M, dtype=torch.int32)
        # self.x_tr_ijm = torch.zeros(self.T, self.R, self.N, self.N, self.M, dtype=torch.int32)

        self.delta_tr_m = torch.ones(self.T, self.R, self.M) / self.M
        self.phi_nm = torch.ones(self.N, self.M)
        self.psi_nm = torch.ones(self.N, self.M)

        self.c_i = torch.ones(self.N)
        self.c_j = torch.ones(self.N)
        self.pi_rr = torch.ones(self.R, self.R) / self.R

        self.l_t_rdot_m = torch.zeros(self.T, self.R, self.M, dtype=torch.int32)
        self.l_t_dotr_m = torch.zeros(self.T, self.R, self.M, dtype=torch.int32)
        self.l_t_rr_m = torch.zeros(self.T, self.R, self.R, self.M)

        self.minus_ptm = torch.ones(self.T, self.M) / self.M

        self.ksi = 1.
        self.upsilon_r = torch.ones(self.R)
        print(f'R:{self.R}, T:{self.T}, M:{self.M}, N:{self.N}')
    # @timing_decorator
    def sample_x(self):
        self.x_tr_dotjm = torch.zeros(self.T, self.R, self.N, self.M, dtype=torch.int32)
        self.x_tr_idotm = torch.zeros(self.T, self.R, self.N, self.M, dtype=torch.int32)
        self.x_tr_dotdotm = torch.zeros(self.T, self.R, self.M, dtype=torch.int32)
        for r in range(self.R):
            for t in range(self.T):
                x_tr_idotm = torch.zeros(self.N, self.M)
                x_tr_dotjm = torch.zeros(self.N, self.M)
                non_zero_indices = torch.nonzero(self.data[r][t])
                x_ij = torch.zeros(self.N, self.N)
                x_ijm = torch.zeros(self.N, self.N, self.M, dtype=torch.int32)
                for index in non_zero_indices:
                    i = index[0]
                    j = index[1]
                    b_ij = int(self.data[r][t][i, j].item())

                    para = self.delta_tr_m[t, r] * self.phi_nm[i, :] * self.psi_nm[j, :]
                    rate = torch.sum(para)
                    x_ij[i, j] = Po_plus(rate)

                    pmf = (para / rate).clamp_min(self.eps)

                    x_ijm[i, j] = Multinomial(int(x_ij[i, j].item()), pmf).sample()

                    x_tr_idotm[i] += x_ijm[i, j]
                    x_tr_dotjm[j] += x_ijm[i, j]

                self.x_tr_dotjm[t, r] = x_tr_dotjm
                self.x_tr_idotm[t, r] = x_tr_idotm

                self.x_tr_dotdotm[t, r] = torch.sum(self.x_tr_dotjm[t, r], axis=0)

    # @timing_decorator
    def sample_phi(self):
        # 预计算减少冗余操作
        delta_sum = self.delta_tr_m.sum(dim=(0, 1))  # (M,)
        x_tr_idotm_sum = self.x_tr_idotm.sum(dim=(0, 1))  # (N, M)
        x_tr_dotjm_sum = self.x_tr_dotjm.sum(dim=(0, 1))  # (N, M)

        # 计算 phi_nm
        phi_alpha = self.a_0 + x_tr_idotm_sum  # (N, M)
        phi_beta = self.c_i[:, None] + delta_sum * (self.psi_nm.sum(dim=0) - self.psi_nm)  # Broadcasting (N, M)

        phi_alpha = phi_alpha.clamp(min=self.eps, max=self.max)
        phi_beta = phi_beta.clamp(min=self.eps, max=self.max)

        self.phi_nm = Gamma(phi_alpha, phi_beta).sample()

        # 计算 psi_nm
        psi_alpha = self.a_1 + x_tr_dotjm_sum  # (N, M)
        psi_beta = self.c_j[:, None] + delta_sum * (self.phi_nm.sum(dim=0) - self.phi_nm)  # Broadcasting (N, M)

        psi_alpha = psi_alpha.clamp(min=self.eps, max=self.max)
        psi_beta = psi_beta.clamp(min=self.eps, max=self.max)

        self.psi_nm = Gamma(psi_alpha, psi_beta).sample()

        # self.phi_nm = self.phi_nm / (self.phi_nm.sum(dim=1, keepdim=True) + self.eps)
        # self.psi_nm = self.psi_nm / (self.psi_nm.sum(dim=1, keepdim=True) + self.eps)


    # @timing_decorator
    def sample_ci(self):

        c_i_alpha = self.f_0 + self.M * self.a_0
        c_i_beta = self.g_0 + torch.sum(self.phi_nm, dim=1)

        self.c_i = Gamma(c_i_alpha, c_i_beta).sample()

        c_j_alpha = 1. + self.M * self.a_1
        c_j_beta = 1. + torch.sum(self.psi_nm, dim=1)

        self.c_j = Gamma(c_j_alpha, c_j_beta).sample()

    # @timing_decorator
    def sample_delta(self):

        # sample l
        self.l_t_rdot_m = torch.zeros(self.T, self.R, self.M, dtype=torch.int32)
        self.l_t_dotr_m = torch.zeros(self.T, self.R, self.M, dtype=torch.int32)
        self.l_t_rr_m = torch.zeros(self.T, self.R, self.R, self.M)
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                l_tr_m_x_crt = self.x_tr_dotdotm[t]
                l_tr_m_s_crt = torch.einsum('ij,jm -> im', self.pi_rr, self.delta_tr_m[t - 1]).clamp_min(self.eps)

            elif t == 0:
                l_tr_m_x_crt = self.x_tr_dotdotm[t] + self.l_t_dotr_m[t + 1]
                l_tr_m_s_crt = self.upsilon_r / self.M
                l_tr_m_s_crt = l_tr_m_s_crt.unsqueeze(1)
            else:
                l_tr_m_x_crt = self.x_tr_dotdotm[t] + self.l_t_dotr_m[t + 1]
                l_tr_m_s_crt = torch.einsum('ij,jm -> im', self.pi_rr, self.delta_tr_m[t - 1]).clamp_min(self.eps)
            self.l_t_rdot_m[t] = CRT(l_tr_m_x_crt, l_tr_m_s_crt)

            if t == 0:
                break
            else:
                for r in range(self.R):
                    for m in range(self.M):
                        if self.l_t_rdot_m[t, r, m] == 0:
                            self.l_t_rr_m[t, r, :, m] = 0
                        else:
                            pmf = self.pi_rr[r] * self.delta_tr_m[t - 1, :, m]  # debug
                            pmf = pmf / torch.sum(pmf)
                            self.l_t_rr_m[t, r, :, m] = Multinomial(self.l_t_rdot_m[t, r, m].item(),
                                                                    pmf.clamp_min(self.eps)).sample()
            self.l_t_dotr_m[t] = self.l_t_rr_m[t].sum(dim=0)

        # sample p
        self.s_m = torch.zeros(self.M)
        for i in range(self.N):
            for j in range(self.N):
                if j != i:
                    self.s_m = self.s_m + self.phi_nm[i] * self.psi_nm[j]

        for t in reversed(range(self.T)):
            if t == self.T - 1:
                self.minus_ptm[t] = self.tau / (self.tau + self.s_m)
            else:
                self.minus_ptm[t] = (self.tau) / (
                            self.tau + self.s_m - torch.log(self.minus_ptm[t + 1]).clamp_min(self.eps))

        # sample delta
        for t in range(self.T):
            if t == 0:
                delta_alpha = self.x_tr_dotdotm[t] + self.l_t_dotr_m[t + 1] + (self.upsilon_r / self.M).unsqueeze(1)
                delta_beta = self.tau + self.s_m - torch.log(self.minus_ptm[t + 1]).clamp_min(self.eps)
            elif t == self.T - 1:
                delta_alpha = self.x_tr_dotdotm[t] + torch.einsum('ij,jm -> im', self.pi_rr, self.delta_tr_m[t - 1])
                delta_beta = self.tau + self.s_m
            else:
                delta_alpha = self.x_tr_dotdotm[t] + self.l_t_dotr_m[t + 1] + torch.einsum('ij,jm -> im', self.pi_rr,
                                                                                           self.delta_tr_m[t - 1])
                delta_beta = self.tau + self.s_m - torch.log(self.minus_ptm[t + 1]).clamp_min(self.eps)
            delta_alpha = delta_alpha.clamp(min=self.eps, max=self.max)
            delta_beta = delta_beta.clamp(min=self.eps, max=self.max)

            self.delta_tr_m[t] = Gamma(delta_alpha, delta_beta).sample()

            # self.delta_tr_m = self.delta_tr_m.clamp(min=1e-6, max=self.max)

    #         print(self.delta_tr_m[0,0])

    # @timing_decorator
    def sample_pi(self):

        l_rr = torch.sum(self.l_t_rr_m[1:], dim=(0, 3))
        for r in range(self.R):
            c1 = self.upsilon_r * self.upsilon_r[r]
            c1[r] = self.ksi * self.upsilon_r[r]
            c2 = l_rr[:, r]
            self.pi_rr[:, r] = Dirichlet((c1 + c2).float()).sample()

    # @timing_decorator
    def sample_upsilon_r_ksi(self):
        # sample q_r
        l_r = torch.sum(self.l_t_dotr_m, axis=(0, 2))
        q_r = Beta(l_r + self.eps, self.upsilon_r * (self.ksi + torch.sum(self.upsilon_r) - self.upsilon_r)).sample()

        # sample h_rr
        m = torch.sum(self.l_t_rr_m[1:], dim=(0, 3))
        r = torch.outer(self.upsilon_r, self.upsilon_r)
        r[torch.eye(self.R, dtype=bool)] = self.ksi * self.upsilon_r
        #         if torch.isnan(r).any() or torch.isinf(r).any():
        #             print('error')
        #             r = torch.clamp(a, min=self.eps, max=1-self.min)  # 将 a 的值限制在合理范围内
        h_rr = CRT(m, r)

        # sample ksi
        ksi_alpha = self.b_0 + torch.diag(h_rr).sum()
        ksi_beta = self.e_0 - (self.upsilon_r * torch.log(1 - q_r + self.eps).clamp_min(self.eps)).sum()
        self.ksi = Gamma(ksi_alpha, ksi_beta).sample()

        # sample upsilon
        h_rr_diag = torch.diag(h_rr)
        h_rr[torch.eye(self.R, dtype=bool)] = 0
        n_r = h_rr_diag + torch.sum(h_rr, axis=0) + torch.sum(h_rr, axis=1) + self.l_t_rdot_m[0].sum(1)
        t_r = -torch.log(1 - q_r + self.eps) * (self.ksi + (torch.sum(self.upsilon_r) - self.upsilon_r)) - torch.sum(
            torch.log(1 - q_r).clamp_min(self.eps) * self.upsilon_r) + torch.log(1 - q_r).clamp_min(
            self.eps) * self.upsilon_r - (torch.log(self.minus_ptm[0] + self.eps) / self.M).sum()
        upsilon_alpha = self.gamma_0 / self.R + n_r
        upsilon_beta = self.beta + t_r

        upsilon_alpha = upsilon_alpha.clamp(min=self.eps, max=self.max)
        upsilon_beta = upsilon_beta.clamp(min=self.eps, max=self.max)

        self.upsilon_r = Gamma(upsilon_alpha, upsilon_beta).sample()

        # sample beta
        beta_alpha = self.d_0 + self.gamma_0
        beta_beta = self.h_0 + self.upsilon_r.sum()

        self.beta = Gamma(beta_alpha, beta_beta).sample()

    @timing_decorator
    def loop_sample(self):
        # print('/n')
        # print("phi_nm max:", self.phi_nm.max().item(), "phi_nm min:", self.phi_nm.min().item())
        # print("psi_nm max:", self.psi_nm.max().item(), "psi_nm min:", self.psi_nm.min().item())
        # print("delta  max:", self.delta_tr_m.max().item(), "delta min:", self.delta_tr_m.min().item())
        self.sample_x()
        self.sample_phi()
        self.sample_ci()
        self.sample_delta()  # bug1--de  delta的多项分布是索引是t-1
        self.sample_pi()  # bug2--de   时间是t=1到t=T-1
        self.sample_upsilon_r_ksi()

    def model_train(self, burnin, collection):
        progress_bar = tqdm(range(burnin + collection))
        delta_sample = torch.zeros(self.T, self.R, self.M)
        phi_sample = torch.zeros(self.N, self.M)
        psi_sample = torch.zeros(self.N, self.M)
        pi_sample = torch.zeros(self.R, self.R)

        for i in progress_bar:
            self.loop_sample()  # 执行一次采样
            if i > burnin and i % 10 == 0:
                delta_sample = delta_sample + self.delta_tr_m
                phi_sample = phi_sample + self.phi_nm
                psi_sample = psi_sample + self.psi_nm
                pi_sample = pi_sample + self.pi_rr

            if i % 50 == 0:
                mem_info = psutil.virtual_memory()
                total_mem = mem_info.total / (1024 ** 3)  # 转换为 GB
                available_mem = mem_info.available / (1024 ** 3)
                used_mem = mem_info.used / (1024 ** 3)

                print(f"CPU 总内存: {total_mem:.2f} GB")
                print(f"CPU 可用内存: {available_mem:.2f} GB")
                print(f"CPU 已用内存: {used_mem:.2f} GB")

        delta_ave = (delta_sample * 10) / collection
        phi_ave = (phi_sample * 10) / collection
        psi_ave = (psi_sample * 10) / collection
        pi_ave = (pi_sample * 10) / collection
        return delta_ave, phi_ave, psi_ave, pi_ave

    def model_pred(self):
        data_pred = torch.zeros(self.R, self.T, self.N, self.N)
        delta_prev = self.delta_tr_m[-1]
        for t in range(self.T):

            delta_next = torch.zeros(self.R, self.M)
            for i in range(10000):
                delta_next += Gamma(torch.einsum('ij,jm -> im', self.pi_rr, delta_prev).clamp_min(self.eps), self.tau).sample()
            delta_next = delta_next / 10000
            delta_next = self.delta_tr_m[-1]
            for r in range(self.R):
                Prob = torch.einsum('m,im,jm->ij', delta_next[r],self.phi_nm, self.psi_nm) + self.eps
                data_pred[r, t] = 1 - torch.exp(-Prob)
            delta_prev = delta_next

        return data_pred





if  __name__ == '__main__':
    # 加载数据
    file_path = 'adjacency_matrix_2022_no_self_loop.npy'
    adjacency_matrix_no_self_loop = np.load(file_path)

    # 计算R维度上每个事件类型的发生次数（矩阵中所有非零元素的总和）
    R_sums = adjacency_matrix_no_self_loop.sum(axis=(1, 2, 3))

    # 选取发生次数最高的20个事件类型的索引
    top_20_indices = np.argsort(R_sums)[-20:][::-1]  # 降序排列


    # 输出结果
    input_data = torch.tensor(adjacency_matrix_no_self_loop[top_20_indices])

    input_data[input_data <= 5] = 0

    # 模型训练
    model = DKG_PG(input_data) # 20*12*235*235
    print(f'M:{model.M},R:{model.R},N:{model.N}')

    burnin = 600
    collection = 200
    target_indices = [(0, 0, m) for m in range(30)]  # 可以根据需求调整
    tracked_values = {index: [] for index in target_indices}  # 创建字典存储每个索引的值
    delta_list = {index: [] for index in target_indices}  # 创建字典存储每个索引的值

    # 开始采样并记录值
    progress_bar = tqdm(range(burnin + collection))
    Probsamps = torch.zeros(model.R, model.T, model.N, model.N)
    ProbAve = torch.zeros_like(Probsamps)

    # 创建图像保存目录
    output_dir = "output_plots"
    os.makedirs(output_dir, exist_ok=True)


    for i in progress_bar:
        model.loop_sample()  # 执行一次采样
        for index in target_indices:
            t, r, m = index
            tracked_values[index].append(model.x_tr_dotdotm[t, r, m].item())  # 记录每个索引的值
            delta_list[index].append(model.delta_tr_m[t, r, m].item())

        #     if i % 1 == 0:  # 每隔 10 轮输出
        #         print(f"Iteration {i}, phi_max={model.phi_nm.max()}, delta_max={model.delta_tr_m.max()},psi_max={model.psi_nm.max()},x_tr_ijm={model.x_tr_ijm.max()}")
        for r in range(model.R):
            for t in range(model.T):
                Prob = torch.einsum('m,im,jm->ij', model.delta_tr_m[t][r], model.phi_nm, model.psi_nm) + model.eps
                Prob = 1 - torch.exp(-Prob)
                if i > burnin:
                    Probsamps[r, t] = Probsamps[r, t] + Prob
                    ProbAve[r, t] = Probsamps[r, t] / (i - burnin)
                else:
                    ProbAve[r, t] = Prob

        if (i + 1) % 100 == 0:
            fig, ax = plt.subplots(model.R, model.T, figsize=(15, 10), dpi=300)
            for r in range(model.R):
                for t in range(model.T):
                    ax[r, t].imshow(ProbAve[r, t], cmap='jet')
                    ax[r, t].set_title(f'r={r + 1}, t={t + 1}')

            # 保存图像文件
            fig.tight_layout()
            save_path = os.path.join(output_dir, f"prob_matrix_iter_{i + 1}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved: {save_path}")
            plt.close(fig)  # 关闭图像，释放内存

    # 绘制变化曲线
    plt.figure(figsize=(12, 8))
    for index, values in tracked_values.items():
        plt.plot(values, label=f"x_tr_dotdotm[{index[0]}, {index[1]}, {index[2]}]")

    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Change in x_tr_dotdotm Over Iterations (Multiple Indices)")
    plt.legend()
    plt.grid()
    # plt.show()

    # 保存变化曲线图
    curve_save_path = os.path.join(output_dir, "tracked_values_curve.png")
    plt.savefig(curve_save_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {curve_save_path}")
    plt.close()