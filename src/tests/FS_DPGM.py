import torch
import time
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from dataclasses import dataclass
from torch.distributions import Gamma, Multinomial
from IPython import embed

@dataclass
class ExperimentConfig:
    N: int
    K: int
    T: int
    train_ratio: float
    burnin: int
    collection: int
    binary: bool

config = ExperimentConfig(60, 30, 6, 1., 1500, 1500, True)
eps = 1e-32

def CRT(n, a):
    max_x = torch.max(n)
    rates = torch.arange(0, max_x).expand(*n.size(), -1)
    rates = a.unsqueeze(-1) / (a.unsqueeze(-1) + rates + eps)
    bers = torch.bernoulli(rates)
    mask = (torch.arange(0, max_x).expand(*n.size(), -1) < n.unsqueeze(-1)).to(torch.int) # lead to kernel crash because of overload memory
    bers = bers * mask
    return torch.sum(bers, dim=-1)

class HGPDR:
    def __init__(self, config: ExperimentConfig):
        self.N = config.N
        self.K = config.K
        self.T = config.T
        self.binary = config.binary

        self.gamma_0 = 1.
        self.c_0 = 1.
        self.xi = 1.
        self.beta = 1.

        self.e_0 = 1.
        self.h_0 = 1.
        self.f_0 = 1.
        self.g_0 = 1.
        self.v_0 = 1.
        self.a_0 = 1.
        self.b_0 = 1.
        self.tau = 1.
        self.eps = 1e-32

        self.r = torch.ones(self.K) / self.K
        self.phi_o = Gamma(1., 1.).sample([self.N, self.K])
        self.phi = Gamma(1., 1.).sample([self.T, self.N, self.K])

        lam_kk = self.r.view(-1, 1) @ self.r.view(1, -1)
        lam_kk = torch.triu(lam_kk, 1) + torch.triu(lam_kk, 1).T
        lam_kk[torch.eye(self.K, self.K, dtype=torch.bool)] = self.r * self.xi
        self.lam_kk = lam_kk

    def do_inference(self, x_dot_kk_dot_t, x_mnk_ddot):
      diag_mask = torch.eye(self.K, self.K, dtype=torch.bool)
      triu_mask = torch.triu(torch.ones(self.K, self.K, dtype=torch.bool), diagonal=1)

      # Sample l_kk then update r and temp
      ## Calculate the quantities theta_kk and one_minus_p_tilde_kk
      theta_kk = torch.einsum('ijk,ikl->jl', self.phi.permute(0,-1,1), self.phi.sum(1, keepdim=True) - self.phi)
      theta_kk[diag_mask] = theta_kk[diag_mask] / 2
      one_minus_p_tilde_kk = torch.clamp_min(self.beta / (self.beta + theta_kk), eps)

      ## update r and temp
      r_xi = torch.unsqueeze(self.r, 0).repeat(self.K, 1).clone()
      r_xi[diag_mask] = self.r # substite self.xi for self.r ???
      l_kk = CRT(x_dot_kk_dot_t, torch.diag(self.r) @ r_xi + eps)
      temp = torch.sum(r_xi * torch.log(one_minus_p_tilde_kk), dim=-1)
      self.r = Gamma(self.gamma_0/self.K + l_kk.sum(dim=-1), self.c_0 - temp).sample()


      # Sample xi
      ell = torch.sum(CRT(x_dot_kk_dot_t[diag_mask], self.xi * self.r + eps))
      self.xi = Gamma(0.01 + ell, 0.01 - torch.sum(self.r * torch.log(one_minus_p_tilde_kk))).sample().item()


      # Sample lam_kk
      r_kk = self.r.view(-1, 1) @ self.r.view(1, -1)
      r_kk[diag_mask] = self.xi * self.r
      lam_kk = torch.zeros(self.K, self.K) + eps
      lam_kk[diag_mask] = Gamma(x_dot_kk_dot_t[diag_mask] + r_kk[diag_mask] + self.eps, self.beta + theta_kk[diag_mask]).sample()
      lam_kk[triu_mask] = Gamma(x_dot_kk_dot_t[triu_mask] + r_kk[triu_mask] + self.eps, self.beta + theta_kk[triu_mask]).sample()
      self.lam_kk = lam_kk + torch.triu(lam_kk, 1).T


      # Sample beta
      self.beta = Gamma(1 + r_kk[diag_mask | triu_mask].sum(), 1 + self.lam_kk[diag_mask | triu_mask].sum()).sample().item()


      # Sample gamma_0
      l_k_tilde = CRT(l_kk.sum(dim=-1), torch.full((self.K,), self.gamma_0/self.K))
      temp = torch.clamp_min(self.c_0 / (self.c_0 - temp), eps)
      self.gamma_0 = Gamma(self.e_0 + l_k_tilde.sum(), self.h_0 - 1/self.K * torch.log(temp).sum()).sample().item()


      # Sample c_0
      self.c_0 = Gamma(0.01 + self.gamma_0, 0.01 + self.r.sum()).sample().item()


      # Sample phi
      omega_nk = (self.phi.sum(dim=1, keepdim=True) - self.phi) @ self.lam_kk     # size=(T,N,K)
      re_omega_nk = torch.flip(omega_nk.clone(), dims=[0])

      ## Backward: sample ro_nk from T to 1
      ro_nk = torch.zeros_like(re_omega_nk)
      ro_nk[0] = re_omega_nk[0] / (re_omega_nk[0] + self.tau)
      for t in range(1, self.T):
          temp_ro = re_omega_nk[t] - torch.log(torch.clamp_min(1 - ro_nk[t-1], eps))
          ro_nk[t] = temp_ro / (self.tau + temp_ro)
      ro_nk = torch.flip(ro_nk, dims=[0])

      ## Backward: sample y_nk from T to 1
      y_nk = torch.zeros_like(omega_nk)           # size=(T,N,K)
      y_nk[-1] = CRT(x_mnk_ddot[-1], self.phi[-2])
      # for t in range(self.T - 2, 0, -1):
      #     y_nk[t] = CRT(x_mnk_ddot[t] + y_nk[t+1], self.phi[t-1])
      y_nk[1:self.T-1] = CRT(x_mnk_ddot[1:self.T-1] + y_nk[2:self.T], self.phi[:self.T-2])
      y_nk[0] = CRT(x_mnk_ddot[0] + y_nk[1], self.phi_o)
      y_nk_o = CRT(x_mnk_ddot[0], torch.full_like(x_mnk_ddot[0], self.g_0))

      ## Forward: from 1 to T sample phi
      self.phi_o = Gamma(self.g_0 + y_nk[0], self.f_0 - torch.log(torch.clamp_min(1 - ro_nk[0], eps))).sample()
      phi_lam = self.phi[0] @ self.lam_kk     # size=(N,K)
      temp = phi_lam.sum(dim=0)               # size=(K)
      for i in torch.randperm(self.N):
          temp = temp - phi_lam[i]            # (K) - (K)
          self.phi[0, i] = Gamma(self.phi_o[i] + y_nk[1, i] + x_mnk_ddot[0, i], self.tau + temp - torch.log(torch.clamp_min(1 - ro_nk[1, i], eps))).sample()
          temp = temp + self.phi[0, i] @ self.lam_kk

      for t in range(1, self.T-1):
          phi_lam = self.phi[t] @ self.lam_kk     # size=(N,K)
          temp = phi_lam.sum(dim=0)               # size=(K)
          for i in torch.randperm(self.N):
              temp = temp - phi_lam[i]            # (K) - (K)
              self.phi[t, i] = Gamma(self.phi[t-1, i] + y_nk[t+1, i] + x_mnk_ddot[t, i], self.tau + temp - torch.log(torch.clamp_min(1 - ro_nk[t+1, i], eps))).sample()
              temp = temp + self.phi[t, i] @ self.lam_kk

      phi_lam = self.phi[-1] @ self.lam_kk    # size=(N,K)
      temp = phi_lam.sum(dim=0)               # size=(K)
      for i in torch.randperm(self.N):
          temp = temp - phi_lam[i]            # (K) - (K)
          self.phi[-1, i] = Gamma(self.phi[-1, i] + x_mnk_ddot[-1, i], self.tau + temp).sample()
          temp = temp + self.phi[-1, i] @ self.lam_kk


      # Sample g_0
      self.g_0 = Gamma(1. + torch.sum(y_nk_o), 1.).sample().item()


      # Sample f_0
      self.f_0 = Gamma(self.g_0 * self.N * self.K + self.a_0, self.b_0 + torch.sum(self.phi_o)).sample().item()


      # calibrate phi and lam_kk
      self.phi[self.phi > 10] = self.eps
      self.lam_kk[self.lam_kk > 10] = self.eps

      return len(torch.nonzero(x_mnk_ddot.sum(dim=0).sum(dim=0)))
    