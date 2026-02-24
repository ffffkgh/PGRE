import time
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('Agg')

from dataclasses import dataclass
from torch.distributions import Gamma, Multinomial, Dirichlet, Beta, Binomial
from tqdm.auto import tqdm
import multiprocessing as mp
from joblib import Parallel, delayed

import os
import sys
from path import Path
current_directory = os.getcwd()
parent_directory = Path(current_directory).parent
sys.path.append(parent_directory)
sys.path.append('/home/HuangRui/PointProcess/NodeGroup/prgds/src')
from apf.base.sample import Sampler
from GS_DPGM import HGPDR,config,CRT
'''version 4: tensor data, interval results, graph guided pi prior.'''

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper

class NBRGDS_tensor():
    def __init__(self, data, K, S, tau, psi, alpha0, epsilon_the, epsilon_lam, stationary=True, parallel = False):
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
        self.psi = torch.tensor(psi)
        self.alpha0 = torch.tensor(alpha0)
        self.epsi_the = torch.tensor(epsilon_the)
        self.epsi_lam = torch.tensor(epsilon_lam)
        self.stationary = stationary
        self.a0 = 0.1
        self.b0 = 0.1
        self.eps = 1e-32
        self.parallel = parallel

        self.complete_stage_len = self.T//S # the length (num of time steps) of a single sub-interval 
        self.remainder_stage_len = self.T%S # the length (num of time steps) of the last uncomplete sub-interval
        self.stage_index = [i*self.complete_stage_len + torch.arange(self.complete_stage_len) for i in range(S)] # index for each sub-interval
        if self.remainder_stage_len != 0: 
            self.stage_index.append(np.arange(S*self.complete_stage_len, S*self.complete_stage_len+self.remainder_stage_len))
        self.S = len(self.stage_index) # the number of sub-interval
        
        # Initial values (sample to update)
        self.phi = [torch.ones(v, self.K, dtype = torch.float32)/v for v in self.V] # list contains self.deep tensors with shape V_m*K
        self.the = torch.ones(self.T,self.K, dtype = torch.float32)
        self.pi = torch.ones(self.S, self.K, self.K, dtype = torch.float32)/self.K
        self.pit = torch.ones(self.T, self.K, self.K, dtype = torch.float32)/self.K 
        self.lam = torch.ones(self.K, dtype = torch.float32)
        self.delt = torch.ones(self.T, dtype = torch.float32)
        self.g = torch.ones(self.K, dtype = torch.float32)
        self.gam = 1.
        self.beta = 1.
        self.hat_h_tk = torch.rand(self.T, self.K)
        
        # Container which need to be updated unified
        self.ProbAve = torch.zeros(self.S, self.K, self.K)
        self.h_tk = torch.zeros(self.T, self.K, dtype = torch.int32)
        self.tilde_h_tk = torch.zeros(self.T, self.K, dtype = torch.int32)
        self.tilde_h_tkk = torch.zeros(self.T, self.K, self.K, dtype = torch.int32)
        self.tilde_tilde_h_tkk = torch.zeros(self.T, self.K, self.K, dtype = torch.int32)
        self.tilde_h_skk = torch.zeros(self.S, self.K, self.K, dtype = torch.int32)

        # expand method doesn't allocate a new memory thus .clone() is necessary
        self.n_tvk = torch.zeros_like(data, dtype = torch.int32).unsqueeze(-1).expand(*[-1]*(self.deep+1), self.K).clone()
        self.n_tdotk = torch.zeros(self.T, self.K, dtype = torch.int32)
        self.n_dotvk = torch.sum(self.n_tvk, axis = 0)
        self.n_tdotdot = torch.zeros(self.T, dtype = torch.int32)
        self.n_dotdotk = torch.zeros(self.K, dtype = torch.int32)

    def multinomial_sample(self, index, pmf):
        n_k = Multinomial(self.data[tuple(index)].item(), pmf[tuple(index)] + self.eps).sample()
        self.n_tvk[tuple(index)] = n_k

    @timing_decorator
    def sample_n(self):
        # scale to num of non-zero values
        non_zero_indices = torch.nonzero(self.data)
        # compute parameters for multinomial sampling
        part1 = self.the @ torch.diag(self.lam)
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
    def sample_h(self):
        sampler = Sampler()
        if self.epsi_the != 0:
            for t in range(self.T):
                for k in range(self.K):
                    v = self.epsi_the - 1
                    a = 2*torch.sqrt(self.the[t,k] * (self.tau) * self.hat_h_tk[t,k])
                    self.h_tk[t,k] = sampler.bessel(v,a)
            self.h_tk[self.h_tk < 0] = 0 # sometimes be -1
        else:
            tilde_h_tdotk = torch.sum(self.tilde_h_tkk, axis=1)
            tilde_h_tp1_dotk = tilde_h_tdotk[1:,:]
            tilde_h_tp1_dotk = torch.vstack((tilde_h_tp1_dotk, torch.zeros(self.K)))
            m_tk = self.n_tdotk + tilde_h_tp1_dotk
            zeta_numerator = self.tau * self.hat_h_tk # T * K
            zeta_denominator = self.tau + torch.outer(self.delt, self.lam) - self.tau * torch.log(1/(1+self.psi)) # T * K
            zeta = torch.divide(zeta_numerator, zeta_denominator)
            for t in range(self.T):
                for k in range(self.K):
                    if m_tk[t, k] == 0:
                        self.h_tk[t, k] = torch.poisson(zeta[t, k])
                    else:
                        self.h_tk[t, k] = sampler.sbch(m_tk[t, k], zeta[t, k])
            self.h_tk[self.h_tk < 0] = 0

    @timing_decorator
    def sample_aux_h(self):
        the_tm1 = self.the[:-1, :]
        the_tm1 = torch.vstack((self.lam, the_tm1))
        mat_tk = torch.einsum('tij, tj -> ti', self.pit, the_tm1)

        # sample hat_h_tk
        shp = self.tau*mat_tk+self.h_tk + self.eps
        rte = self.psi + 1.
        self.hat_h_tk = Gamma(shp, rte).sample()

        # sample tilde_htk
        m = self.h_tk
        r = self.tau*mat_tk
        self.tilde_h_tk = CRT(m, r)
    
        # sample tilde_h_tkk
        for t in range(self.T):
            for k in range(self.K):
                n = (self.tilde_h_tk[t,k]).to(int).item()
                if n == 0: 
                    self.tilde_h_tkk[t,k,:] = 0.
                else:
                    p = self.pit[t, k, :] * the_tm1[t,:]
                    p = (p + self.eps) / (p + self.eps).sum()
                    self.tilde_h_tkk[t,k,:] = Multinomial(n, p).sample()

        # compute tilde_hskk
        for s in range(self.S):
            self.tilde_h_skk[s] = torch.sum(self.tilde_h_tkk[self.stage_index[s]], axis = 0) 

    @timing_decorator
    def sample_the(self):
        tilde_h_tdotk = torch.sum(self.tilde_h_tkk, axis=1)
        tilde_h_tp1_dotk = tilde_h_tdotk[1:,:]
        tilde_h_tp1_dotk = torch.vstack((tilde_h_tp1_dotk, torch.zeros(self.K)))
        shp = self.epsi_the + self.h_tk + self.n_tdotk + tilde_h_tp1_dotk + self.eps  # T*K
        rte = self.tau + torch.matmul(self.delt.view(-1,1), self.lam.view(1,-1)) - self.tau*torch.log(1/(1+self.psi))+self.eps  # T*K
        self.the = Gamma(shp, rte).sample()

    @timing_decorator
    def sample_lam(self):
        shp = self.epsi_lam/self.K + self.g + self.n_dotdotk + torch.sum(self.tilde_h_tkk[0], axis=0) +self.eps
        rte = self.beta + torch.matmul(self.delt.float(), self.the.float()) - self.tau*torch.log(1/(1+self.psi))+self.eps
        self.lam = Gamma(shp, rte).sample()

    @timing_decorator
    def sample_delt(self):
        if self.stationary == True:
            shp = self.a0 + self.data.sum() + self.eps
            rte = self.b0 + (torch.matmul(self.the, self.lam)).sum() + self.eps
            self.delt = torch.ones(self.T) * Gamma(shp, rte).sample()
        else:
            shp = self.a0 + self.n_tdotdot + self.eps
            rte = self.b0 + torch.matmul(self.the, self.lam) + self.eps
            self.delt = Gamma(shp, rte).sample()
    
    @timing_decorator
    def sample_g(self):
        sampler = Sampler()
        for k in range(self.K):
            v = self.epsi_lam/self.K - 1
            a = 2*torch.sqrt(self.lam[k] * self.beta * (self.gam/self.K))
            self.g[k] = sampler.bessel(v,a)
        self.g[self.g < 0] = 0

    @timing_decorator
    def sample_gam(self):
        shp = self.a0 + (self.g).sum() + self.eps
        rte = self.b0 + 1
        self.gam = Gamma(shp, rte).sample()
    
    @timing_decorator
    def sample_beta(self):
        shp = self.alpha0 + self.epsi_lam + (self.g).sum() + self.eps
        rte = self.alpha0 + (self.lam).sum() +self.eps
        self.beta = Gamma(shp, rte).sample()

    @timing_decorator
    def sample_pi(self):
        cond_A = self.a0 + self.tilde_h_skk + self.eps
        for s in range(self.S):
            for k in range(self.K):
                self.pi[s,:,k] = Dirichlet(cond_A[s,:,k] / torch.linalg.norm(cond_A[s, :,k])).sample()
            self.pit[self.stage_index[s]] = self.pi[s] 

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
    'psi':2.,
    'alpha0': 10.,
    'epsilon_the': 0.1,
    'epsilon_lam': 1.,
    'stationary': True,
    'data' : data[:12][:100],
    'K' : 50, # latent components
    'S' : 5, # sub-intervals
    'parallel': True
    }

    modle = NBRGDS_tensor(**params)
    esti_steady = []
    transition_steady = []
    for iter in tqdm(range(burnin+maxiter)):
        modle.sample_n()
        modle.sample_h()
        modle.sample_aux_h()
        modle.sample_the()
        modle.sample_lam()
        modle.sample_delt()
        modle.sample_g()
        modle.sample_gam()
        modle.sample_beta()
        modle.sample_phi()
        modle.sample_pi()

        if iter > burnin:
            transition_steady.append(modle.pi.clone())