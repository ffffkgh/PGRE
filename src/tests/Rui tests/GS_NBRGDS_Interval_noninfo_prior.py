from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import sys
from path import Path
sys.path.append(Path(__file__).parent.parent)
from apf.base.sample import Sampler
from IPython import embed
from DPGM import HGPDR, config, CRT
import torch
import time
'''version 2: 2 dimension data and interval results, but specify noninformative prior over pi.'''

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper

class GS_NBRGDS(HGPDR):
    def __init__(self,V,T,K,S, data, tau, epsilon_the, epsilon_lam, 
                 alpha0, psi, steady=True, seed=None):
        # Hyper Parameter
        self.nvt = data               # V*T
        self.V = V
        self.T = T
        self.K = K
        self.C = K                                                
        self.tau = tau       
        self.epsi_the = epsilon_the
        self.epsi_lam = epsilon_lam
        self.alpha0 = alpha0
        self.psi = psi
        self.steady = steady
        self.a0 = 0.1
        self.b0 = 0.1
        self.c0 = 0.1                                                       
        self.e0 = 0.1                                                       
        self.f0 = 0.1
        
        self.complete_stage_len = T//S
        self.remainder_stage_len = T%S
        self.stage_index = [i*self.complete_stage_len + np.arange(self.complete_stage_len) for i in range(S)]
        if self.remainder_stage_len != 0: 
            self.stage_index.append(np.arange(S*self.complete_stage_len, S*self.complete_stage_len+self.remainder_stage_len))
        self.S = len(self.stage_index) # stage num

        # Initial values (sample to update)
        self.phi = np.ones((V,K))/V   # V*K
        self.the = np.ones((T,K))
        self.pi = np.ones((self.S,K,K))/K  # S*K*K
        self.pit = np.ones((T,K,K))/K 
        self.lam = np.ones(K)         # K
        self.delt = np.ones(T)        # T
        self.g = np.ones(K)           # K
        self.gam = 1                  # num
        self.beta = 1                 # num
        self.hat_htk = np.random.uniform(size = (T,K))  # T*K
        self.W = np.random.poisson(np.ones((self.S, self.K, self.K)))
        self.Z = (self.W>=1).astype(int)                                    # S*K*K binary 
        self.D = np.random.uniform(0,1, size = (K,K))                       # initial value K*K 
        self.A = np.tile(self.D, (self.S,1,1))*self.Z                       # S*K*K                                             
        #self.M = np.random.uniform(0,0.2, size = (self.S, self.K, self.C))  # M和R的初值比较重要
        #self.R = np.random.uniform(0,0.2, size = (self.C, self.C))          # 使sample Z时的pkk0和pkk1不要有太大差距

        # Container which need to be updated unified
        self.eta = np.zeros((self.S, self.K))
        self.ProbAve = torch.zeros(self.S, self.K, self.K)
        self.htk = np.zeros((self.T, self.K), dtype = np.int32)
        self.tilde_htk = np.zeros((self.T, self.K), dtype = np.int32)
        self.tilde_htkk = np.zeros((self.T, self.K, self.K), dtype = np.int32)
        self.tilde_tilde_htkk = np.zeros((self.T, self.K, self.K), dtype=np.int32)
        self.tilde_hskk = np.zeros((self.S, self.K, self.K), dtype = np.int32)
        self.tilde_tilde_hskk = np.zeros((self.T, self.K, self.K), dtype = np.int32)
        self.nvtk = np.zeros((self.V, self.T, self.K), dtype = np.int32)
        self.n_tk = np.zeros((self.T, self.K), dtype = np.int32)
        self.nv_k = np.zeros((self.V, self.K), dtype = np.int32)
        self.n_t_ = np.zeros(self.T, dtype=np.int32)
        self.n__k = np.zeros(self.K, dtype=np.int32)

        config.T = self.S # num of stage
        config.N = self.K # num of latent node
        config.K = self.C # num of group
        #super(GS_NBRGDS,self).__init__(config)
        #由于HGPDR和GS_NBRGDS中有很多重名变量因此不能直接继承
        #在GS_NBRGDS的构造函数中创建一个HGPDR的实例属性
        self.dpgm_model = HGPDR(config)

        self.eps = 1e-32
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    #------------------------------------------------------------------
    #@timing_decorator
    def sample_nvtk(self):
        for v in range(self.V):
            for t in range(self.T):
                if self.nvt[v,t] == 0:
                    pass
                else:
                    p = self.lam * self.the[t,:] * self.phi[v,:]
                    p = (p + self.eps) / (p + self.eps).sum()
                    self.nvtk[v,t,:] = np.random.multinomial(self.nvt[v,t], p)

        self.n_tk = np.sum(self.nvtk, axis = 0)
        self.nv_k = np.sum(self.nvtk, axis = 1)
        self.n__k = np.sum(self.nvtk, axis = (0,1))
        self.n_t_ = np.sum(self.nvtk, axis = (0,2))
    
    #------------------------------------------------------------------
    #@timing_decorator
    def sample_htk(self):
        sampler = Sampler()
        if self.epsi_the != 0:
            for t in range(self.T):
                for k in range(self.K):
                    v = self.epsi_the - 1
                    a = 2*np.sqrt(self.the[t,k] * (self.tau) * self.hat_htk[t,k])
                    self.htk[t,k] = sampler.bessel(v,a)
            self.htk[self.htk < 0] = 0 # sometimes be -1
        else:
            htp1_k = self.tilde_htk[1:,:]
            htp1_k = np.vstack((htp1_k, np.zeros(self.K)))
            m = self.n_tk + htp1_k
            zeta = np.divide(self.tau*self.hat_htk, \
                             self.tau + (np.array([self.delt]).T @ np.array([self.lam])) \
                             - self.tau*np.log(1/(1+self.psi)))
            for t in range(self.T):
                for k in range(self.K):
                    if m[t,k] == 0:
                        self.htk[t, k] = np.random.poisson(zeta[t,k])
                    else:
                        self.htk[t, k] = sampler.sbch(m[t,k], zeta[t,k])

    #@timing_decorator
    def sample_aux_h(self):
        thetm1 = self.the[:-1, :]
        thetm1 = np.vstack((self.lam, thetm1))
        mat_tk = torch.einsum('tij, tj -> ti', torch.tensor(self.pit), torch.tensor(thetm1))

        # sample hat_htk
        shp = self.tau*mat_tk+self.htk + self.eps
        rte = self.psi + 1
        self.hat_htk = np.random.gamma(shp, 1/rte)

        # sample tilde_htk
        m = torch.tensor(self.htk)
        r = self.tau*mat_tk
        self.tilde_htk = np.array(CRT(m, r))
    
        # sample tilde_htkk
        for t in range(self.T):
            for k in range(self.K):
                n = self.tilde_htk[t,k]
                p = self.pit[t, k, :] * thetm1[t,:]
                p = (p + self.eps) / (p + self.eps).sum()
                self.tilde_htkk[t,k,:] = np.random.multinomial(n, p)

        # compute tilde_hskk
        for s in range(self.S):
            self.tilde_hskk[s] = np.sum(self.tilde_htkk[self.stage_index[s]], axis = 0) 

    #--------------------------------------------------------------------
    #@timing_decorator
    def sample_the(self):
        htp1_k = self.tilde_htk[1:,:]
        htp1_k = np.vstack((htp1_k, np.zeros(self.K)))
        shp = self.n_tk + self.htk + self.epsi_the + htp1_k + self.eps      # T*K
        rte = self.tau + (np.array([self.delt]).T @ np.array([self.lam])) - self.tau*np.log(1/(1+self.psi))  # K
        self.the = np.random.gamma(shp, 1/rte)

    #--------------------------------------------------------------------
    #@timing_decorator
    def sample_lam(self):
        shp = self.epsi_lam/self.K + self.g + self.n__k + np.sum(self.tilde_htkk[0], axis=0)+self.eps
        rte = self.beta + (np.array([self.delt]) @ self.the).flatten() -self.tau*np.log(1/(1+self.psi)) + self.eps
        self.lam = np.random.gamma(shp, 1/rte)

    #--------------------------------------------------------------------
    #@timing_decorator
    def sample_delt(self):
        if self.steady == True:
            shp = self.a0 + self.nvt.sum() + self.eps
            rte = self.b0 + (self.the @ np.array([self.lam]).T).sum() + self.eps
            self.delt = np.ones(self.T) * np.random.gamma(shp, 1/rte)
        else:
            shp = self.a0 + self.n_t_ + self.eps
            rte = self.b0 + (self.the @ np.array([self.lam]).T).flatten() + self.eps
            self.delt = np.random.gamma(shp, 1/rte)

    #--------------------------------------------------------------------
    #@timing_decorator
    def sample_g(self):
        sampler = Sampler()
        for k in range(self.K):
            v = self.epsi_lam/self.K - 1
            a = 2*np.sqrt(self.lam[k] * self.beta * (self.gam/self.K))
            self.g[k] = sampler.bessel(v,a)
        self.g[self.g < 0] = 0
    #--------------------------------------------------------------------
    #@timing_decorator
    def sample_gam(self):
        shp = self.a0 + (self.g).sum() + self.eps
        rte = self.b0 + 1
        self.gam = np.random.gamma(shp, 1/rte)
    #--------------------------------------------------------------------
    #@timing_decorator
    def sample_beta(self):
        shp = self.alpha0 + self.epsi_lam + (self.g).sum() + self.eps
        rte = self.alpha0 + (self.lam).sum() +self.eps
        self.beta = np.random.gamma(shp, 1/rte)
    #-------------------------------------------------------------------
    #@timing_decorator
    def sample_pi(self):
        # a inductive prior over pi
        cond_A = np.ones((self.S, self.K, self.K))*0.01
        #cond_A[:, np.eye(self.K, dtype=bool)] = 0.1
        cond_A = cond_A + self.tilde_hskk + self.eps
        for s in range(self.S):
            for k in range(self.K):
                self.pi[s, :,k] = np.random.dirichlet(cond_A[s, :,k] / np.linalg.norm(cond_A[s, :,k]))
            self.pit[self.stage_index[s]] = self.pi[s] 

    #--------------------------------------------------------------------
    #@timing_decorator
    def sample_phi(self):
        phi_param = self.a0 + self.nv_k + self.eps
        for k in range(self.K):
            self.phi[:,k] = np.random.dirichlet(phi_param[:,k]/np.linalg.norm(phi_param[:,k]))

###################################################################################

if __name__ == "__main__":
    V=10
    T=30
    K=10
    sys = np.zeros((V, T))
    high_value = 50
    ############# Only Stage 1 #############
    sys[0:4,0:10]     = high_value        # 1st community
    sys[4:8,10:20]   = high_value        # 2st community
    sys[8:10,20:30] = high_value        # 3st community

    burnin = 600
    maxiter = 200
    params = {
    'tau': 1,
    'epsilon_the': 0.1,
    'epsilon_lam': 1,
    'psi':2,
    'alpha0': 10,
    'steady': False
    }
    params['data'] = sys
    params['V'] = sys.shape[0]        
    params['T'] = sys.shape[1]
    params['K'] = 10
    params['S'] = 3

    gs_nbrgds = GS_NBRGDS(**params)
    gs_nbrgds.D = np.ones((gs_nbrgds.K, gs_nbrgds.K))
    for iter in tqdm(range(burnin+maxiter)):
        gs_nbrgds.sample_nvtk()
        gs_nbrgds.sample_htk()
        gs_nbrgds.sample_aux_h()
        gs_nbrgds.sample_the()
        gs_nbrgds.sample_lam()
        gs_nbrgds.sample_delt()
        gs_nbrgds.sample_g()
        gs_nbrgds.sample_gam()
        gs_nbrgds.sample_beta()
        gs_nbrgds.sample_phi()
        gs_nbrgds.sample_pi()


    """
    import scipy.io as sio
    icew = sio.loadmat('/home/HuangRui/PointProcess/PRGDS/src/tests/DATA/icews.mat')
    icew_data = icew['data'][:,900:1000]
    data = icew_data
    ########################## parameters setting ##################################
    burnin = 40
    maxiter = 20
    params = {
    'tau': 1,
    'epsilon_the': 0.1,
    'epsilon_lam': 1,
    'psi':10,
    'alpha0': 10,
    'steady': False
    }
    params['data'] = data.T
    params['V'] = data.shape[1]
    params['K'] = 25
    params['T'] = data.shape[0]
    params['S'] = 3
    nbrgds = GS_NBRGDS(**params)
    ############################# sampling setting ##################################
    for iter in tqdm(range(burnin+maxiter)):
        nbrgds.sample_nvtk()
        nbrgds.sample_htk()
        nbrgds.sample_aux_h()
        nbrgds.sample_the()
        nbrgds.sample_lam()
        nbrgds.sample_delt()
        nbrgds.sample_g()
        nbrgds.sample_gam()
        nbrgds.sample_beta()
        nbrgds.sample_phi()
        nbrgds.sample_pi()
        nbrgds.sample_eta()
        nbrgds.sample_2tilde_htkk()
        nbrgds.sample_Z()
        nbrgds.sample_D()
        nbrgds.sample_dpgm()

        #if iter > burnin:
        #    esti = np.array(nbrgds.phi @ np.diag(nbrgds.lam) @ (nbrgds.the).T @ np.diag(nbrgds.delt))

    plt.imshow(nbrgds.ProbAve[-1] + nbrgds.ProbAve[-1].T, cmap='jet')
    plt.colorbar()
    plt.savefig('test.png')
    """



