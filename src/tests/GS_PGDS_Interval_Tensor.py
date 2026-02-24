import time
import torch
import numpy as np
from dataclasses import dataclass
from torch.distributions import Gamma, Multinomial, Dirichlet, Beta, Binomial
from tqdm.auto import tqdm
from GS_DPGM import HGPDR,config,CRT
import multiprocessing as mp
'''version 5: PGDS, tensor data, interval results, graph guided pi prior.'''


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper

class GS_PGDS_tensor():
    def __init__(self, data, K, S, tau, stationary=True, parallel = False):
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
        self.W = torch.poisson(torch.ones(self.S, self.K, self.K))
        self.Z = torch.triu((self.W >= 1).float(), 1)
        self.D = torch.rand(self.K, self.K)*0.1
        self.A = torch.tile(self.D, (self.S,1,1))*self.Z
        self.eta = torch.zeros(self.S, self.K)
        
        # Container which need to be updated unified
        self.ProbAve = torch.zeros(self.S, self.K, self.K)
        self.l_tkdot = torch.zeros(self.T, self.K, dtype = torch.int32)
        self.l_tdotk = torch.zeros(self.T, self.K, dtype = torch.int32)
        self.l_tkk = torch.zeros(self.T, self.K, self.K, dtype = torch.int32)
        self.l_skk = torch.zeros(self.S, self.K, self.K, dtype = torch.int32)
        self.tilde_l_tkk = torch.zeros(self.T, self.K, self.K, dtype = torch.int32)
        self.tilde_l_skk = torch.zeros(self.S, self.K, self.K, dtype = torch.int32)
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
        # super(model,self).__init__(config)
        # 由于HGPDR和model中有很多重名变量因此不能直接继承
        # 在model的构造函数中创建一个HGPDR的实例属性
        self.dpgm_model = HGPDR(config)

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
                # sample l_tkk (l_tkk t=0 is useless thus we don't need to modify pmf)
                if self.l_tkdot[t, k] == 0:
                    self.l_tkk[t, k, :] = 0
                else:
                    self.l_tkk[t, k, :] = Multinomial(self.l_tkdot[t, k].item(), pmf + self.eps).sample()
                if t == 0:
                    # compute l_tdotk (t = 0)
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
    def sample_tilde_lskk(self):
        # recover the shape of matrix A to tkk
        Atkk = torch.zeros(self.T, self.K, self.K)
        for s in range(self.S):
            Atkk[self.stage_index[s]] = self.A[s]
        # sample 
        m = self.l_tkk.clone()
        r = (Atkk + torch.permute(Atkk, dims = (0,2,1))).clone()
        self.tilde_l_tkk = CRT(m,r)
        # compute tilde_l_skk
        for s in range(self.S):
            self.tilde_l_skk[s] = torch.sum(self.tilde_l_tkk[self.stage_index[s]], axis=0)

    @timing_decorator 
    def sample_Z(self):
        M = self.dpgm_model.phi
        R = self.dpgm_model.lam_kk
        mrm = torch.einsum('tmi,ij,tnj -> tmn', M, R, M) # S*K*K
        for s in range(self.S):
            for k1 in range(self.K):
                for k2 in range(k1, self.K): # just sampling upper triangle
                    kk = self.tilde_l_skk[s,k1,k2]
                    if kk == 0:
                        pkk0 = torch.exp(-mrm[s,k1,k2]) * 1 
                        pkk1 = (1-pkk0)*torch.exp(self.D[k1,k2]*torch.log(1-self.eta[s,k2]+self.eps))
                        self.Z[s,k1,k2] = Binomial(1, pkk1/(pkk0+pkk1+self.eps)).sample()
                    else:
                        self.Z [s,k1,k2] = 1

    @timing_decorator
    def sample_D(self):
        tilde_l_dotkk = torch.sum(self.tilde_l_tkk, axis=0)
        shp = self.e0 + tilde_l_dotkk + self.eps
        rte = self.f0 - torch.einsum('sij, sj -> ij', self.Z + torch.permute(self.Z, dims = (0,2,1)), torch.log(1-self.eta+self.eps))
        self.D = Gamma(shp, rte).sample()
        self.A = torch.tile(self.D, (self.S,1,1))*self.Z

    @timing_decorator
    def sample_dpgm(self):
        B =[]
        m_idx = []
        n_idx = []
        self.Z = torch.triu(self.Z, 1)
        for s in range(self.S):
            m_t, n_t = torch.nonzero(self.Z[s], as_tuple = True)
            b = self.Z[s][m_t, n_t]
            B.append(b)
            m_idx.append(m_t)
            n_idx.append(n_t)

        self.dpgm_model.do_inference(B, m_idx, n_idx)
        M = self.dpgm_model.phi
        R = self.dpgm_model.lam_kk
        
        for s in range(self.S):
            Prob = M[s] @ R @ M[s].T + self.eps
            Prob = 1 - torch.exp(-Prob)
            self.ProbAve[s] = Prob
        
        # calibrate phi and lam_kk
        self.dpgm_model.phi[self.dpgm_model.phi>10] = self.eps
        self.dpgm_model.lam_kk[self.dpgm_model.lam_kk>10] = self.eps

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
    'data' : data[:, 900:1000],
    'K' : 70, # latent components
    'S' : 6, # sub-intervals
    'parallel': True
    }

    model = GS_PGDS_tensor(**params)
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
        model.sample_tilde_lskk()
        model.sample_Z()
        model.sample_D()
        model.sample_dpgm()

        if iter > burnin:
            transition_steady.append(model.pi.clone())