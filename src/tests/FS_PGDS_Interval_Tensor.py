import time
import torch
import numpy as np
from dataclasses import dataclass
from torch.distributions import Gamma, Multinomial, Dirichlet, Beta, Binomial
from tqdm.auto import tqdm
from FS_DPGM import HGPDR,config,CRT

import os
import sys
from path import Path
import multiprocessing as mp
current_directory = os.getcwd()
parent_directory = Path(current_directory).parent
sys.path.append(parent_directory)
sys.path.append('/home/HuangRui/PointProcess/NodeGroup/prgds/src')
from apf.base.sample import Sampler


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper

class FS_PGDS_tensor():
    def __init__(self, data, K, S, tau, stationary=True, parallel=True):
        '''
        data: tensor, the first dimension must be time T, the shape is T*(i*j*a...)
        K: num of latent component
        '''
        self.T = data.shape[0] # int: num of time steps
        self.V = data.shape[1:] # torch.Size: (i*j*a...)
        self.K = K
        self.C = K
        self.deep = len(self.V) # int: num of dimensions except for t
        self.data = data.int() # tensor with int elements
        self.tau = torch.tensor(tau)
        self.stationary = stationary
        self.gamma0 = 50
        self.a0 = 0.1
        self.b0 = 0.1
        self.e0 = 0.1
        self.f0 = 0.1
        self.eps = 1e-32
        self.parallel = parallel

        self.complete_stage_len = self.T//S
        self.remainder_stage_len = self.T%S
        self.stage_index = [i*self.complete_stage_len + torch.arange(self.complete_stage_len) for i in range(S)]
        if self.remainder_stage_len != 0: 
            self.stage_index.append(np.arange(S*self.complete_stage_len, S*self.complete_stage_len+self.remainder_stage_len))
        self.S = len(self.stage_index) # stage num
        
        # Initial values (sample to update)
        self.phi = [torch.ones(v, self.K)/v for v in self.V] # list contains self.deep tensors with shape V_m*K
        self.the = torch.ones(self.T,self.K)
        self.pi = torch.ones(self.S, self.K, self.K)/self.K
        self.pit = torch.ones(self.T, self.K, self.K)/self.K 
        self.delt = torch.ones(self.T)
        self.A = torch.ones(self.S, self.K, self.K, dtype = torch.int32)
        self.Askkcc = torch.zeros(self.S, self.K, self.K, self.C, self.C)
        self.Askc = torch.zeros(self.S, self.K, self.C)
        self.Acc = torch.zeros(self.C, self.C)
        self.eta = torch.zeros(self.S, self.K)
        
        # Container which need to be updated unified
        self.ProbAve = torch.zeros(self.S, self.K, self.K)
        self.l_tkdot = torch.zeros(self.T, self.K, dtype = torch.int32)
        self.l_tdotk = torch.zeros(self.T, self.K, dtype = torch.int32)
        self.l_tkk = torch.zeros(self.T, self.K, self.K, dtype = torch.int32)
        self.l_skk = torch.zeros(self.S, self.K, self.K, dtype = torch.int32)
        self.zeta = torch.zeros(self.T)

        # expand method doesn't allocate a new memory thus .clone() is necessary
        self.n_tvk = torch.zeros_like(data, dtype = torch.int32).unsqueeze(-1).expand(*[-1]*(self.deep+1), self.K).clone()
        self.n_tdotk = torch.zeros(self.T, self.K, dtype = torch.int32)
        self.n_dotvk = torch.sum(self.n_tvk, axis = 0)
        self.n_tdotdot = torch.zeros(self.T, dtype = torch.int32)
        self.n_dotdotk = torch.zeros(self.K, dtype = torch.int32)

        # configuration of DPGM
        config.T = self.S # num of stage
        config.N = self.K # num of latent node
        config.K = self.C # num of group
        # super(GS_NBRGDS,self).__init__(config)
        # 由于HGPDR和GS_NBRGDS中有很多重名变量因此不能直接继承
        # 在GS_NBRGDS的构造函数中创建一个HGPDR的实例属性
        self.fs_dpgm_model = HGPDR(config)

    def multinomial_sample(self, index, pmf):
        n_k = Multinomial(self.data[tuple(index)].item(), pmf[tuple(index)] + self.eps).sample()
        self.n_tvk[tuple(index)] = n_k

    @timing_decorator
    def sample_n(self):
        # scale to num of non-zero values
        non_zero_indices = torch.nonzero(self.data)
        # compute parameters for multinomial sampling
        part1 = self.the
        einsum_eq_string_left = ','.join([f'{chr(108+i)}k' for i in range(self.deep)])
        einsum_eq_string_right = ''.join([chr(108+i) for i in range(self.deep)])
        # einsum_eq: lk,mk,nk,... -> lmn..k
        einsum_eq = einsum_eq_string_left + '->' + einsum_eq_string_right + 'k'
        part2 = torch.einsum(einsum_eq, *self.phi)
        pmf = torch.einsum('tk, l...k -> tl...k', part1, part2)
        # sampling for each t,v
        if self.parallel == False:
            for index in non_zero_indices:
                n_k = Multinomial(self.data[tuple(index)].item(), pmf[tuple(index)]+self.eps).sample()
                self.n_tvk[tuple(index)] = n_k
        else:
            with mp.Pool() as pool:
                args_list = [(index, pmf) for index in non_zero_indices]
                pool.starmap(self.multinomial_sample, args_list)

        self.n_tdotk = torch.einsum('t...k -> tk', self.n_tvk)
        self.n_dotvk = torch.sum(self.n_tvk, axis=0)
        self.n_tdotdot = torch.einsum('t... -> t', self.n_tvk)
        self.n_dotdotk = torch.einsum('...k -> k', self.n_tvk)

    @timing_decorator
    def sample_l(self):
        # sample l_tkdot and l_tkk, compute l_tdotk
        the_tm1 = self.the[:-1, :]
        the_tm1 = torch.vstack((torch.zeros(self.K), the_tm1)) # use zeros to take place
        r = self.tau * torch.einsum('tij, tj -> ti', self.pit, the_tm1) # T*K
        r[0, :] = self.tau
        for t in reversed(range(self.T)):
            for k in range(self.K):
                if t == self.T - 1:
                    m = self.n_tdotk[t,k]
                else:
                    # compute l_tdotk (from T-1 ~ 1)
                    self.l_tdotk[t + 1, k] = torch.sum(self.l_tkk[t + 1, :, k])
                    m = self.n_tdotk[t, k] + self.l_tdotk[t + 1, k]
                # sample l_tkdot
                self.l_tkdot[t, k] = CRT(m,r[t, k])
                pmf = self.pit[t, k, :] * the_tm1[t, :]
                # sample l_tkk
                if self.l_tkdot[t, k] == 0:
                    self.l_tkk[t, k, :] = 0
                else:
                    self.l_tkk[t, k, :] = Multinomial(self.l_tkdot[t, k].item(), pmf + self.eps).sample()
                # compute l_tdotk (t = 0)
                if t == 0:
                    self.l_tdotk[t, k] = torch.sum(self.l_tkk[t, :, k])
                else:
                    pass

        # compute l_skk
        for s in range(self.S):
            self.l_skk[s] = torch.sum(self.l_tkk[self.stage_index[s]], axis = 0)

        # compute zeta
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                self.zeta[t] = torch.log(1 + self.tau / (self.delt[t] + self.eps))
            else:
                self.zeta[t] = torch.log(1 + self.tau / (self.delt[t] + self.eps) + self.zeta[t+1])


    @timing_decorator
    def sample_the(self):
        # compute shape parameters
        l_tp1dotk = torch.vstack((self.l_tdotk[1:, :], torch.zeros(self.K))) 
        the_tm1 = torch.vstack((torch.zeros(self.K), self.the[:-1, :]))
        tau_pi_the = self.tau * torch.einsum('tij, tj -> ti', self.pit, the_tm1)
        shp = self.n_tdotk + l_tp1dotk + tau_pi_the + self.eps
        # compute rate parameters
        zeta_tp1 = torch.cat((self.zeta[1:], torch.tensor([0])))
        rte = self.tau + self.delt + self.tau * zeta_tp1
        rte = torch.tile(rte.view(-1, 1), (1, self.K))
        self.the = Gamma(shp, rte).sample()

    @timing_decorator
    def sample_delt(self):
        if self.stationary == True:
            shp = self.a0 + self.data.sum() + self.eps
            rte = self.b0 + (self.the).sum() + self.eps
            self.delt = torch.ones(self.T) * Gamma(shp, rte).sample()
        else:
            shp = self.a0 + self.n_tdotdot + self.eps
            rte = self.b0 + torch.sum(self.the, axis = 1) + self.eps
            self.delt = Gamma(shp, rte).sample()

    @timing_decorator
    def sample_pi(self):
        cond_A = self.A + torch.permute(self.A, dims = (0,2,1))+ self.l_skk + self.eps
        for s in range(self.S):
            for k in range(self.K):
                self.pi[s, :,k] = Dirichlet(cond_A[s, :,k] / torch.linalg.norm(cond_A[s, :,k])).sample()
            self.pit[self.stage_index[s]] = self.pi[s] 

    @timing_decorator
    def sample_eta(self):
        shp1 = torch.sum(self.l_skk, axis = 1) + self.eps
        shp2 = torch.sum(self.A + torch.permute(self.A, dims = (0,2,1)), axis = 1) + self.eps  #值变得太小了会导致eta=1
        self.eta = Beta(shp1, shp2).sample()

    @timing_decorator
    def sample_A(self):
        sampler = Sampler()
        M = self.fs_dpgm_model.phi
        R = self.fs_dpgm_model.lam_kk
        c1 = torch.einsum('tim, mn, tjn -> tij', M, R, M)
        c2 = self.eta / (1 - self.eta + self.eps)
        c3 = 1
        for s in range(self.S):
            for k1 in range(self.K):
                for k2 in range(self.K):
                    # sample A
                    cc = c1[s,k1,k2] * c2[s, k2] / (c3 + c2[s, k2])
                    if self.l_skk[s,k1,k2] == 0:
                        Askk = torch.poisson(cc)
                    else:
                        Askk = sampler.sbch(self.l_skk[s,k1,k2], cc)
                    if Askk < 0:
                        Askk = 0
                    elif Askk > 50:
                        Askk = 50
                    self.A[s, k1, k2] = Askk
                    # sample Askkcc
                    pmf = R * torch.outer(M[s, k1, :], M[s, k2, :])
                    pmf = (pmf.flatten() + self.eps)/((pmf + self.eps).sum())
                    if self.A[s, k1, k2].item() == 0:
                        Askkcc = torch.zeros(self.C ,self.C)
                        self.Askkcc[s, k1, k2, :, :] = 0
                    else:
                        Askkcc = Multinomial(self.A[s, k1, k2].to(torch.int32).item(), pmf).sample()
                        self.Askkcc[s, k1, k2, :, :] = Askkcc.reshape(self.C ,self.C)
        print('Sum over self.A: ', torch.sum(self.A))
        # compute Askc 需要去掉一部分
        self.Askc = torch.sum(self.Askkcc, axis = (2,4))
        # compute Acc  需要去掉一部分
        self.Acc = torch.sum(self.Askkcc, axis = (0,1,2))

    @timing_decorator
    def sample_dpgm(self):
        self.fs_dpgm_model.do_inference(self.Acc, self.Askc)
        M = self.fs_dpgm_model.phi
        R = self.fs_dpgm_model.lam_kk
        
        for s in range(self.S):
            Prob = M[s] @ R @ M[s].T
            self.ProbAve[s] = Prob

    @timing_decorator
    def sample_phi(self):
        einsum_eq_string_left = 't'+''.join([chr(108+i) for i in range(self.deep)])+'k'
        for d in range(self.deep):
            einsum_eq_string_right = einsum_eq_string_left[d+1] + 'k'
            einsum_eq = einsum_eq_string_left + '->' + einsum_eq_string_right
            phi_param = self.a0 + torch.einsum(einsum_eq, self.n_tvk) + self.eps # i_m * K
            # sample for each column in d-th phi
            for k in range(self.K):
                self.phi[d][:,k] = Dirichlet(phi_param[:,k]/torch.linalg.norm(phi_param[:,k])).sample()

if __name__ == '__main__':
    test = np.load('/home/HuangRui/PointProcess/NodeGroup/data/icews_preprocessed.npz', allow_pickle=True)
    data = torch.tensor(test['data'].astype(np.float32))
    # run model
    burnin = 60
    maxiter = 20
    params = {
    'tau': 1.,
    'stationary': True,
    'data' : data[:12][:100],
    'K' : 50, # latent components
    'S' : 5, # sub-intervals
    'parallel': True
    }

    model = FS_PGDS_tensor(**params)
    esti_steady = []
    transition_steady = []
    for iter in tqdm(range(burnin+maxiter)):
        model.sample_n()
        model.sample_l()
        model.sample_the()
        model.sample_delt()
        model.sample_phi()
        model.sample_pi()
        model.sample_eta()
        model.sample_A()
        model.sample_dpgm()

        if iter > burnin:
            transition_steady.append(model.pi.clone())