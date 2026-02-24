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
from DPGM import HGPDR, config, Po_plus, CRT
import torch
import time
'''version 1: 2 dimension data and time step results.'''

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper

class GS_NBRGDS():
    def __init__(self,V,T,K, data, tau, epsilon_the, epsilon_lam, 
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

        # Initial values (sample to update)
        self.phi = np.ones((V,K))/V   # V*K
        self.the = np.ones((T,K))
        self.pi = np.ones((T,K,K))/K  # T*K*K 
        self.lam = np.ones(K)         # K
        self.delt = np.ones(T)        # T
        self.g = np.ones(K)           # K
        self.gam = 1                  # num
        self.beta = 1                 # num
        self.hat_htk = np.random.uniform(size = (T,K))  # T*K
        self.W = np.random.poisson(np.ones((self.T, self.K, self.K)))
        self.Z = (self.W>=1).astype(int)                                    # T*K*K binary 
        self.D = np.random.uniform(0,1, size = (K,K))                       # initial value K*K 
        self.A = np.tile(self.D, (self.T,1,1))*self.Z                                                                   
        self.M = np.random.uniform(0,0.2, size = (self.T, self.K, self.C))  # M和R的初值比较重要
        self.R = np.random.uniform(0,0.2, size = (self.C, self.C))          # 使sample Z时的pkk0和pkk1不要有太大差距
        self.tilde_tilde_htkk = np.zeros((self.T, self.K, self.K), dtype=np.int32)

        # Container which need to be updated unified
        self.eta = np.zeros((self.T, self.K))
        self.ProbAve = torch.zeros(self.T, self.K, self.K)
        self.htk = np.zeros((self.T, self.K), dtype = np.int32)
        self.tilde_htk = np.zeros((self.T, self.K), dtype = np.int32)
        self.tilde_htkk = np.zeros((self.T, self.K, self.K), dtype = np.int32)
        self.nvtk = np.zeros((self.V, self.T, self.K), dtype = np.int32)
        self.n_tk = np.zeros((self.T, self.K), dtype = np.int32)
        self.nv_k = np.zeros((self.V, self.K), dtype = np.int32)
        self.n_t_ = np.zeros(self.T, dtype=np.int32)
        self.n__k = np.zeros(self.K, dtype=np.int32)

        self.eps = 1e-32
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    #------------------------------------------------------------------
    @timing_decorator
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
    @timing_decorator
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

    @timing_decorator
    def sample_aux_h(self):
        thetm1 = self.the[:-1, :]
        thetm1 = np.vstack((self.lam, thetm1))

        #diag_block_pi = torch.block_diag(*(torch.tensor(self.pi))) #(T*K)*(T*K) 
        #diag_block_thetm1 = torch.block_diag(*(torch.tensor(thetm1))).T #(T*K)*T
        #mat_tk = (diag_block_pi @ diag_block_thetm1).T # T*(T*K)
        #mat_tk = torch.stack([mat_tk[i, i*K:(i+1)*K] for i in range(T)]) # T*K
        mat_tk = torch.einsum('tij, tj -> ti', torch.tensor(self.pi), torch.tensor(thetm1))

        # sample hat_htk
        shp = self.tau*mat_tk+self.htk + self.eps
        rte = self.psi + 1
        self.hat_htk = np.random.gamma(shp, 1/rte)

        # sample tilde_htk
        m = torch.tensor(self.htk)
        r = self.tau*mat_tk
        self.tilde_htk = np.array(CRT(m, r))
    
        # sample_tilde_htkk
        for t in range(self.T):
            for k in range(self.K):
                n = self.tilde_htk[t,k]
                p = self.pi[t, k, :] * thetm1[t,:]
                p = (p + self.eps) / (p + self.eps).sum()
                self.tilde_htkk[t,k,:] = np.random.multinomial(n, p)
    #--------------------------------------------------------------------
    @timing_decorator
    def sample_the(self):
        htp1_k = self.tilde_htk[1:,:]
        htp1_k = np.vstack((htp1_k, np.zeros(self.K)))
        shp = self.n_tk + self.htk + self.epsi_the + htp1_k + self.eps      # T*K
        rte = self.tau + (np.array([self.delt]).T @ np.array([self.lam])) - self.tau*np.log(1/(1+self.psi))  # K
        self.the = np.random.gamma(shp, 1/rte)

    #--------------------------------------------------------------------
    @timing_decorator
    def sample_lam(self):
        shp = self.epsi_lam/self.K + self.g + self.n__k + np.sum(self.tilde_htkk[0], axis=0)+self.eps
        rte = self.beta + (np.array([self.delt]) @ self.the).flatten() -self.tau*np.log(1/(1+self.psi)) + self.eps
        self.lam = np.random.gamma(shp, 1/rte)

    #--------------------------------------------------------------------
    @timing_decorator
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
    @timing_decorator
    def sample_g(self):
        sampler = Sampler()
        for k in range(self.K):
            v = self.epsi_lam/self.K - 1
            a = 2*np.sqrt(self.lam[k] * self.beta * (self.gam/self.K))
            self.g[k] = sampler.bessel(v,a)
        self.g[self.g < 0] = 0
    #--------------------------------------------------------------------
    @timing_decorator
    def sample_gam(self):
        shp = self.a0 + (self.g).sum() + self.eps
        rte = self.b0 + 1
        self.gam = np.random.gamma(shp, 1/rte)
    #--------------------------------------------------------------------
    @timing_decorator
    def sample_beta(self):
        shp = self.alpha0 + self.epsi_lam + (self.g).sum() + self.eps
        rte = self.alpha0 + (self.lam).sum() +self.eps
        self.beta = np.random.gamma(shp, 1/rte)
    #-------------------------------------------------------------------
    @timing_decorator
    def sample_pi(self):
        # Note: if all of elements in a column equal to 0
        # then np.random.dirichlet(col) will be nan 
        cond_A = self.A + self.tilde_htkk + self.eps
        for t in range(self.T):
            for k in range(self.K):
                self.pi[t, :,k] = np.random.dirichlet(cond_A[t, :,k] / np.linalg.norm(cond_A[t, :,k]))

    @timing_decorator
    def sample_eta(self):
        shp1 = np.sum(self.tilde_htkk, axis = 1) + self.eps
        shp2 = np.sum(self.A, axis = 1) + self.eps  #值变得太小了会导致eta=1
        self.eta = np.random.beta(shp1, shp2)
    @timing_decorator
    def sample_2tilde_htkk(self):
        #for t in range(self.T):
        m = torch.tensor(self.tilde_htkk)
        r = torch.tensor(self.A)
        self.tilde_tilde_htkk = np.array(CRT(m,r))
            #for k1 in range(self.K):
            #    for k2 in range(self.K):
            #        self.tilde_tilde_htkk[t,k1,k2] = self.CRT(self.tilde_htkk[t,k1,k2], self.A[t,k1,k2])
    @timing_decorator
    def sample_Z(self):
        mrm = np.einsum('tmi,ij,tnj -> tmn', self.M, self.R, self.M) # T*K*K
        for t in range(self.T):
            for k1 in range(self.K):
                for k2 in range(self.K):
                    kk = self.tilde_tilde_htkk[t,k1,k2]
                    if kk == 0:
                        pkk0 = np.exp(-mrm[t,k1,k2]) * 1  # 1表示Zkk=0时hkk=0的概率为1
                        pkk1 = (1-pkk0)*np.exp(self.D[k1,k2]*np.log(1-self.eta[t,k2]+self.eps))
                        self.Z[t,k1,k2] = np.random.binomial(1, pkk1/(pkk0+pkk1))
                    else:
                        self.Z[t,k1,k2] = 1
    @timing_decorator
    def sample_D(self):
        tilde_tilde_h_dot_kk = np.sum(self.tilde_tilde_htkk, axis=0)
        shp = self.e0 + tilde_tilde_h_dot_kk + self.eps
        rte = self.f0 - np.einsum('tij, tj -> ij', self.Z, np.log(1-self.eta+self.eps))
        #rte[rte<0] = self.f0
        self.D = np.random.gamma(shp, 1/rte)
    @timing_decorator
    def compute_A(self):
        self.A = np.tile(self.D, (self.T,1,1))*self.Z
    @timing_decorator
    def sample_dpgm(self):
        B =[]
        m_idx = []
        n_idx = []
        config.T = self.T
        config.N = self.K # num of latent node
        config.K = self.C # num of group
        for t in range(config.T):
            m_t, n_t = torch.nonzero(torch.tensor(self.Z[t]), as_tuple = True)
            b = self.Z[t][m_t, n_t]
            B.append(b)
            m_idx.append(m_t)
            n_idx.append(n_t)

        model = HGPDR(config)
        k = model.do_inference(B, m_idx, n_idx)

        for t in range(config.T):
            Prob = model.phi[t] @ model.lam_kk @ model.phi[t].T + self.eps
            Prob = 1 - torch.exp(-Prob)
            self.ProbAve[t] = Prob
        self.M = model.phi
        self.R = model.lam_kk
    #--------------------------------------------------------------------
    @timing_decorator
    def sample_phi(self):
        phi_param = self.a0 + self.nv_k + self.eps
        #phi_param[phi_param<=0] = self.eps
        for k in range(self.K):
            self.phi[:,k] = np.random.dirichlet(phi_param[:,k]/np.linalg.norm(phi_param[:,k]))

###################################################################################

if __name__ == "__main__":
    import scipy.io as sio
    icew = sio.loadmat('/home/HuangRui/PointProcess/PRGDS/src/tests/DATA/icews.mat')
    icew_data = icew['data'][:,:1000]
    data = icew_data
    ########################## parameters setting ##################################

    V = len(data.T)
    K = 25
    T = len(data)
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
    params['V'] = V
    params['K'] = K         
    params['T'] = T
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
        nbrgds.compute_A()
        nbrgds.sample_dpgm()

        #if iter > burnin:
        #    esti = np.array(nbrgds.phi @ np.diag(nbrgds.lam) @ (nbrgds.the).T @ np.diag(nbrgds.delt))

    plt.imshow(nbrgds.ProbAve[-1] + nbrgds.ProbAve[-1].T, cmap='jet')
    plt.colorbar()
    plt.savefig('test.png')




