import time
import torch
import numpy as np
from dataclasses import dataclass
from torch.distributions import Gamma, Multinomial, Dirichlet, Beta, Binomial
from tqdm.auto import tqdm
from GS_DPGM import HGPDR,config,CRT
import multiprocessing as mp


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper

class PGDS_tensor():
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
        self.nu = torch.ones(self.K)
        self.delt = torch.ones(self.T)
        self.xi = 1.
        self.beta = 1.
        
        # Container which need to be updated unified
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
        r[0, :] = self.tau * self.nu
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
                    # compute l_tdotk (t = 0) which is useless actually
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
        the_tm1 = torch.vstack((torch.zeros(self.K), self.the[:-1, :])) # use zeros to take place
        tau_pi_the = self.tau * torch.einsum('tij, tj -> ti', self.pit, the_tm1)
        tau_pi_the[0,:] = self.tau * self.nu
        shp = self.n_tdotk + l_tp1dotk + tau_pi_the + self.eps
        # compute rate parameters
        zeta_tp1 = torch.cat((self.zeta[1:], torch.tensor([0])))
        rte = self.tau + self.delt + self.tau * zeta_tp1
        rte = torch.tile(rte.view(-1, 1), (1, self.K))
        self.the = Gamma(shp, rte).sample()

    @timing_decorator
    def sample_delt(self):
        if self.stationary == True:
            shp = self.a0 + (self.data).sum() + self.eps
            rte = self.b0 + (self.the).sum() + self.eps
            self.delt = torch.ones(self.T) * Gamma(shp, rte).sample()
        else:
            shp = self.a0 + self.n_tdotdot + self.eps
            rte = self.b0 + torch.sum(self.the, axis = 1) + self.eps
            self.delt = Gamma(shp, rte).sample()
    
    @timing_decorator
    def sample_beta(self):
        shp = self.a0 + self.gamma0 + self.eps
        rte = self.a0 + (self.nu).sum() +self.eps
        self.beta = Gamma(shp, rte).sample()

    @timing_decorator
    def sample_xi_nu(self):
        # sample q_k
        l_dottk = torch.sum(self.l_skk, axis = (0,1))
        q_k = Beta(l_dottk, self.nu * (self.xi + (self.nu.sum()).repeat(self.K) - self.nu)).sample()
        # sample h_kk
        m = torch.sum(self.l_skk, axis = 0)
        r = torch.outer(self.nu, self.nu)
        r[torch.eye(self.K, dtype=bool)] = self.xi * self.nu
        h_kk = CRT(m, r)
        # sample xi
        shp = self.a0 + torch.trace(h_kk)
        rate = self.a0 - torch.matmul(self.nu, torch.log(1-q_k+self.eps))
        self.xi = Gamma(shp, rate).sample()
        # compute n_k
        h_kk[torch.eye(self.K, dtype = bool)] = 0
        n_k = torch.diag(h_kk) + torch.sum(h_kk, axis = 0) + torch.sum(h_kk, axis = 1) + self.l_tkdot[0,:]
        rho_k = - torch.log(1-q_k+self.eps)*(self.xi + (self.nu.sum()).repeat(self.K) - self.nu) - \
                torch.sum(torch.log(1-q_k+self.eps) * self.nu) - torch.log(1-q_k+self.eps) * self.nu + self.tau * self.zeta[0]
        shp = self.gamma0 / self.K + n_k
        rate = self.beta + rho_k
        self.nu = Gamma(shp, rate).sample()

    @timing_decorator
    def sample_pi(self):
        cond_A = torch.outer(self.nu, self.nu)
        cond_A[torch.eye(self.K, dtype=bool)] = self.xi * self.nu
        cond_A = cond_A + torch.sum(self.l_skk, axis = 0)
        for s in range(self.S):
            for k in range(self.K):
                self.pi[s, :,k] = Dirichlet(cond_A[:,k] / torch.linalg.norm(cond_A[:,k])).sample()
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
