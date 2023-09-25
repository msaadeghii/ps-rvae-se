#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Original script by Xiaoyu BIE (xiaoyu.bie@inrai.fr). Adapted and modified by Mosataf Sadeghi (mostafa.sadeghi@inrai.fr).
License agreement in LICENSE.txt

"""

import os
import time
import torch
from torch import optim
import numpy as np
import librosa
import soundfile as sf
from src.utils import EvalMetrics
from abc import ABC, abstractmethod
from src.utils.custom_optimizer import SGLD, pSGLD


class EM(ABC):
    def __init__(
        self, X, v, W, H, g, R, niter=100, device="cpu", verbose=False, boosting=False
    ):
        self.X = X  # (F, N) niosy STFT
        self.v = v  # video features
        self.X_abs_2 = np.abs(X) ** 2  # (F, N) mixture power spectrogram
        self.W = W  # (F, K) NMF noise model
        self.H = H  # (K, N) NMF moise model
        self.g = g  # ( , N) gain parameters, varies cross time
        self.niter = niter  # number of iteration for EM steps

        self.verbose = verbose
        self.device = device

        F, N = self.X.shape
        self.Vs = torch.zeros(R, F, N).to(
            self.device
        )  # (R, F, N) variance of clean speech
        self.Vs_scaled = torch.zeros_like(self.Vs).to(
            self.device
        )  # (R, F, N) Vs multiplied by gain
        self.Vx = torch.zeros_like(self.Vs).to(
            self.device
        )  # (R, F, N) variance of noisy speech

        self.params_np2tensor()
        self.compute_Vb()

    def params_np2tensor(self):
        self.X_abs_2 = torch.from_numpy(self.X_abs_2.astype(np.float32)).to(self.device)
        self.W = torch.from_numpy(self.W.astype(np.float32)).to(self.device)
        self.H = torch.from_numpy(self.H.astype(np.float32)).to(self.device)
        self.g = torch.from_numpy(self.g.astype(np.float32)).to(self.device)

    @abstractmethod
    def compute_Vs(self, Z):
        pass

    def compute_Vs_scaled(self):
        self.Vs_scaled = self.g * self.Vs

    def compute_Vb(self):
        self.Vb = self.W @ self.H

    def compute_Vx(self):
        self.Vx = self.Vs_scaled + self.Vb

    def compute_loss(self):
        return torch.mean(torch.log(self.Vx) + self.X_abs_2 / self.Vx)

    @abstractmethod
    def E_step(self):
        pass

    def M_step(self):
        # M-step aims to update W, H and g

        # update W
        num = (self.X_abs_2 * torch.sum(self.Vx**-2, dim=0)) @ self.H.T  # (F, K)
        den = torch.sum(self.Vx**-1, dim=0) @ self.H.T  # (F, K)
        self.W = self.W * torch.sqrt(num / den)  # (F, K)

        # update variancec
        self.compute_Vb()
        self.compute_Vx()

        # update H
        num = self.W.T @ (self.X_abs_2 * torch.sum(self.Vx**-2, dim=0))  # (K, N)
        den = self.W.T @ torch.sum(self.Vx**-1, axis=0)  # (K, N)
        self.H = self.H * torch.sqrt(num / den)  # (K, N)

        # update variance of noise and noisy speech
        self.compute_Vb()
        self.compute_Vx()

        # normalize W and H, don't change Vb
        norm_col_W = torch.sum(torch.abs(self.W), dim=0)
        self.W = self.W / norm_col_W.unsqueeze(0)
        self.H = self.H * norm_col_W.unsqueeze(1)

        # update g
        num = torch.sum(
            self.X_abs_2 * torch.sum(self.Vs * (self.Vx**-2), dim=0), dim=0
        )  # (, N)
        den = torch.sum(torch.sum(self.Vs * (self.Vx**-1), dim=0), dim=0)  # (, N)
        self.g = self.g * torch.sqrt(num / den)  # (, N)

        # update variance of scaled clean speech and noisy speech
        self.compute_Vs_scaled()
        self.compute_Vx()

    @abstractmethod
    def compute_WienerFilter(self):
        pass

    def run_EM(self):
        loss = np.zeros(self.niter)

        start_time = time.time()

        for n in range(self.niter):
            # update encoder
            self.E_step()

            # update noise params and gain
            self.M_step()

            loss[n] = self.compute_loss()
            if self.verbose and n % 10 == 0:
                print("iter: {}/{} - loss: {:.4f}".format(n + 1, self.niter, loss[n]))

        end_time = time.time()
        elapsed = end_time - start_time

        WFs, WFn = self.compute_WienerFilter()

        self.S_hat = WFs * self.X
        self.N_hat = WFn * self.X

        return loss, elapsed


class VEM(EM):
    def __init__(
        self,
        X,
        v,
        W,
        H,
        g,
        dvae,
        optimizer,
        beta,
        niter=100,
        nepochs_E_step=1,
        lr=1e-3,
        nsamples_E_step=1,
        nsamples_WF=1,
        device="cpu",
        verbose=False,
    ):
        super().__init__(
            X=X,
            v=v,
            W=W,
            H=H,
            g=g,
            R=nsamples_E_step,
            niter=niter,
            device=device,
            verbose=verbose,
        )

        # E-step params
        self.nepochs_E_step = nepochs_E_step
        self.nsamples_E_step = nsamples_E_step
        self.nsamples_WF = nsamples_WF

        # DVAE model
        self.dvae = dvae
        self.dvae.train()
        self.optimizer = optimizer
        self.beta = beta

    def sample_Z(self, nsamples=5):
        # (F, N) -> (1, F, N) -> (N, 1, F) -> (N, R, F)
        data_batch = self.X_abs_2.unsqueeze(0).permute(-1, 0, 1).repeat(1, nsamples, 1)

        with torch.no_grad():
            Z, _, _ = self.dvae.inference(
                data_batch
            )  # (N, R, F) -> (N, R, L), R is considered as batch size

        return Z

    def compute_Vs(self, Z):
        with torch.no_grad():
            Vs = torch.exp(self.dvae.generation_x(Z))  # (N, R, L) -> (N, R, F)

        self.Vs = Vs.permute(1, -1, 0)  # (R, F, N)

    def E_step(self):
        for epoch in range(self.nepochs_E_step):
            # Forward
            data_in = self.X_abs_2.T.unsqueeze(1)  # (F, N) -> (N, 1, F)
            data_out = torch.exp(self.dvae(data_in))  # logvar -> variance
            Vs = data_out.squeeze().T  # (N, 1, F) -> (F, N)
            Vx = self.g * Vs + self.Vb  # (F, N)

            # Compute loss and optimize
            z_mean = self.dvae.z_mean
            z_logvar = self.dvae.z_logvar
            z_mean_p = self.dvae.z_mean_p
            z_logvar_p = self.dvae.z_logvar_p
            loss_recon = torch.sum(self.X_abs_2 / Vx + torch.log(Vx))
            loss_KLD = -0.5 * torch.sum(
                z_logvar
                - z_logvar_p
                - torch.div(
                    z_logvar.exp() + (z_mean - z_mean_p).pow(2), z_logvar_p.exp()
                )
            )
            # loss_tot = loss_recon + self.beta * loss_KLD
            loss_tot = loss_recon + loss_KLD
            self.optimizer.zero_grad()
            loss_tot.backward()
            self.optimizer.step()

        # sample Z
        Z = self.sample_Z(self.nsamples_E_step)  # (N, R, L)

        # compute variance
        self.compute_Vs(Z)
        self.compute_Vs_scaled()
        self.compute_Vx()

    def compute_WienerFilter(self):
        # sample z
        Z = self.sample_Z(self.nsamples_WF)  # (N, R, L)

        # compute variance
        self.compute_Vs(Z)  # (R, F, N)
        self.compute_Vs_scaled()
        self.compute_Vx()

        # compute Wiener Filters
        WFs = torch.mean(self.Vs_scaled / self.Vx, dim=0)  # (F, N)
        WFn = torch.mean(self.Vb / self.Vx, dim=0)  # (, N)

        WFs = WFs.cpu().detach().numpy()
        WFn = WFn.cpu().detach().numpy()

        return WFs, WFn


class LDEM(EM):
    def __init__(
        self,
        X,
        v,
        W,
        H,
        g,
        dvae,
        optimizer="sgld",
        beta=None,
        niter=100,
        nepochs_E_step=1,
        tv_param=0.0,
        lr=5e-3,
        eta=0.01,
        num_samples=10,
        nsamples_E_step=1,
        nsamples_WF=1,
        device="cuda",
        verbose=False,
    ):
        super().__init__(
            X=X,
            v=v,
            W=W,
            H=H,
            g=g,
            R=nsamples_E_step,
            niter=niter,
            device=device,
            verbose=verbose,
        )

        # DVAE model
        self.device = device
        self.dvae = dvae
        self.dvae.eval()
        self.beta = beta
        self.tv_param = tv_param  # total variation regularization
        self.sqrt_eta = torch.sqrt(torch.Tensor([eta])).to(
            self.device
        )  # noise variance for the proposal distribution

        self.num_samples = num_samples
        if v != None:
            self.v = v.repeat(1, self.num_samples, 1)

        # E-step params
        # Initialize the latent variable
        with torch.no_grad():
            data_in = self.X_abs_2.T.unsqueeze(1)  # (F, N) -> (N, 1, F)
            _ = self.dvae(data_in, v=v)
            _, Z, _ = self.dvae.inference(data_in, v=v)
        self.Z = Z.clone() + self.sqrt_eta * torch.randn(
            Z.shape[0], self.num_samples, Z.shape[2]
        ).to(
            device
        )  # (N, R, L=32)

        self.nepochs_E_step = nepochs_E_step
        self.nsamples_E_step = nsamples_E_step
        self.nsamples_WF = nsamples_WF
        self.optimizer_type = optimizer
        # optimizer for the E-step
        if self.optimizer_type == "sgld":
            self.optimizer = SGLD([self.Z.requires_grad_()], lr=lr)
        elif self.optimizer_type == "psgld":
            self.optimizer = pSGLD([self.Z.requires_grad_()], lr=lr)
        elif self.optimizer_type == "adam":
            self.optimizer = optim.Adam([self.Z.requires_grad_()], lr=lr)
        else:
            raise ValueError(
                "Unknown optimizer type. Should be one of adam, sgld, or psgld."
            )

        self.dvae.train()  # vae in train mode

    def compute_Vs(self, Z):
        Vs = torch.exp(self.dvae.generation_x(Z, v=self.v))  # (N, R, L) -> (N, R, F)
        self.Vs = Vs.detach().permute(1, -1, 0)  # (R, F, N)
        return Vs

    def total_variation_loss(self, Z):
        tv = (torch.abs(Z[1:, :, :] - Z[:-1, :, :]).pow(1)).sum()
        return tv

    def E_step(self):
        def closure():
            # reset gradients
            self.optimizer.zero_grad()
            # Forward
            Vs = self.compute_Vs(self.Z).transpose(0, -1)
            Vx = self.g * Vs + self.Vb[:, None, :]  # (F, N)

            # Compute loss and optimize
            z_mean_p, z_logvar_p = self.dvae.generation_z(self.Z, v=self.v)
            log_prior_term = torch.sum((self.Z - z_mean_p).pow(2) / z_logvar_p.exp())
            log_likelihood_term = torch.sum(
                self.X_abs_2[:, None, :] / Vx + torch.log(Vx)
            )
            loss_tot = (
                log_likelihood_term
                + log_prior_term
                + self.tv_param * self.total_variation_loss(self.Z)
            )

            loss_tot.backward()
            return loss_tot

        for epoch in range(self.nepochs_E_step):
            self.optimizer.step(closure)

        if self.optimizer_type in ["sgld", "psgld"]:
            self.Z.data += self.sqrt_eta * torch.randn_like(self.Z)

        # compute variance
        self.compute_Vs(self.Z.detach())
        self.compute_Vs_scaled()
        self.compute_Vx()

    def compute_WienerFilter(self):
        # compute variance
        self.compute_Vs(self.Z)  # (R, F, N)
        self.compute_Vs_scaled()
        self.compute_Vx()

        # compute Wiener Filters
        WFs = torch.mean(self.Vs_scaled / self.Vx, dim=0)  # (F, N)
        WFn = torch.mean(self.Vb / self.Vx, dim=0)  # (, N)

        WFs = WFs.cpu().detach().numpy()
        WFn = WFn.cpu().detach().numpy()

        return WFs, WFn


# %% MALAEM: Speech enhancement with Metropolis-adjusted Langevin algorithm (MALA) for the E-step.
class MALAEM(EM):
    def __init__(
        self,
        X,
        v,
        W,
        H,
        g,
        dvae,
        beta=None,
        niter=100,
        nepochs_E_step=10,
        lr=5e-3,
        eta=0.01,
        num_samples=1,
        nsamples_E_step=1,
        nsamples_WF=1,
        burn_in=5,
        device="cuda",
        verbose=False,
    ):
        super().__init__(
            X=X,
            v=v,
            W=W,
            H=H,
            g=g,
            R=nsamples_E_step,
            niter=niter,
            device=device,
            verbose=verbose,
        )

        # DVAE model
        self.device = device
        self.dvae = dvae
        self.dvae.eval()
        self.beta = beta
        self.lr = lr
        self.niter = niter

        self.sqrt_eta = torch.sqrt(torch.Tensor([eta])).to(
            self.device
        )  # noise variance for the proposal distribution

        self.num_samples = num_samples
        self.burn_in = burn_in

        # E-step params
        # Initialize the latent variable
        with torch.no_grad():
            data_in = self.X_abs_2.T.unsqueeze(1)  # (F, N) -> (N, 1, F)
            _ = self.dvae(data_in)
            _, Z, _ = self.dvae.inference(data_in)
        self.Z = Z.clone() + self.sqrt_eta * torch.randn_like(Z).to(device)  # (N, L=32)
        self.Z.requires_grad_()

        self.nepochs_E_step = max(nepochs_E_step, 10)
        self.nsamples_E_step = nsamples_E_step
        self.nsamples_WF = nsamples_WF

        self.dvae.train()  # vae in train mode

    def taming_grad(self, g):
        tamed_g = g / (1.0 + self.lr * torch.linalg.matrix_norm(g))
        # tamed_g = g / (1.0 + self.lr * torch.abs(g))
        return tamed_g

    def MALA(self):
        self.Z = self.Z.squeeze()
        N, L = self.Z.shape

        Z_sampled = torch.zeros(N, self.nepochs_E_step - self.burn_in, L)

        cpt = 0
        averaged_acc_rate = 0  # acceptance probability
        for n in range(self.nepochs_E_step):
            U_x, grad_U_x = (
                self.loss(self.Z)[1],
                torch.autograd.grad(self.loss(self.Z)[0], [self.Z], create_graph=False)[
                    0
                ].data,
            )
            Z_tmp = (
                self.Z
                - self.lr * self.taming_grad(grad_U_x)
                + torch.sqrt(torch.tensor([2 * self.lr]).to(self.device))
                * torch.randn_like(self.Z).to(self.device)
            )
            U_y, grad_U_y = (
                self.loss(Z_tmp)[1],
                torch.autograd.grad(self.loss(Z_tmp)[0], [Z_tmp], create_graph=False)[
                    0
                ].data,
            )

            acc_prob = (
                -U_y
                + U_x
                + 1.0
                / (4 * self.lr)
                * (
                    torch.sum(
                        (Z_tmp - self.Z + self.lr * self.taming_grad(grad_U_x)) ** 2,
                        dim=1,
                    ).unsqueeze(1)
                    - torch.sum(
                        (self.Z - Z_tmp + self.lr * self.taming_grad(grad_U_y)) ** 2,
                        dim=1,
                    ).unsqueeze(1)
                )
            )

            is_acc = (torch.log(torch.rand(N, 1).to(self.device)) < acc_prob).squeeze()

            # averaged_acc_rate += (
            #     torch.sum(is_acc).cpu().numpy()
            #     / np.prod(is_acc.shape)
            #     * 100
            #     / (self.nepochs_E_step)
            # )

            self.Z.data[is_acc, :] = Z_tmp.data[is_acc, :]

            if n > self.burn_in - 1:
                Z_sampled[:, cpt, :] = self.Z
                cpt += 1

        # print("averaged acceptance rate: %f" % (averaged_acc_rate))
        self.Z = self.Z.unsqueeze(1)

        return Z_sampled

    def loss(self, Z):
        # Forward
        Vs = self.compute_Vs(Z).transpose(0, -1)
        Vx = self.g * Vs + self.Vb[:, None, :]  # (F, N)

        # Compute loss and optimize
        z_mean_p, z_logvar_p = self.dvae.generation_z(Z)  # (N, L)

        log_prior_term = torch.sum((Z - z_mean_p).pow(2) / z_logvar_p.exp(), dim=-1).t()
        log_likelihood_term = torch.sum(
            self.X_abs_2[:, None, :] / Vx + torch.log(Vx), dim=0
        )
        loss_i = log_likelihood_term + log_prior_term
        return loss_i.sum(), loss_i.squeeze(1).t()

    def compute_Vs(self, Z):
        if Z.ndim == 2:
            Z = Z.unsqueeze(1)
        Vs = torch.exp(self.dvae.generation_x(Z))  # (N, R, L) -> (N, R, F)
        self.Vs = Vs.detach().permute(1, -1, 0)  # (R, F, N)
        return Vs

    def E_step(self):
        Z_mean = self.MALA().to(self.device)
        self.Z.data += self.sqrt_eta * torch.randn_like(self.Z)

        # compute variance
        self.compute_Vs(Z_mean)
        self.compute_Vs_scaled()
        self.compute_Vx()

    def compute_WienerFilter(self):
        # Sample Z
        Z_mean = self.MALA().to(self.device)
        # compute variance
        self.compute_Vs(Z_mean)  # (R, F, N)
        self.compute_Vs_scaled()
        self.compute_Vx()

        # compute Wiener Filters
        WFs = torch.mean(self.Vs_scaled / self.Vx, dim=0)  # (F, N)
        WFn = torch.mean(self.Vb / self.Vx, dim=0)  # (, N)

        WFs = WFs.cpu().detach().numpy()
        WFn = WFn.cpu().detach().numpy()

        return WFs, WFn


# %% MHEM: Speech enhancement with Metropolis-Hastings sampling method for the E-step.
class MHEM(EM):
    def __init__(
        self,
        X,
        v,
        W,
        H,
        g,
        dvae,
        beta=None,
        niter=100,
        nepochs_E_step=10,
        lr=5e-3,
        eta=0.01,
        num_samples=1,
        nsamples_E_step=1,
        nsamples_WF=1,
        burn_in=5,
        device="cuda",
        verbose=False,
    ):
        super().__init__(
            X=X,
            v=v,
            W=W,
            H=H,
            g=g,
            R=nsamples_E_step,
            niter=niter,
            device=device,
            verbose=verbose,
        )

        # DVAE model
        self.device = device
        self.dvae = dvae
        self.dvae.eval()
        self.beta = beta
        self.lr = lr
        self.niter = niter

        self.sqrt_eta = torch.sqrt(torch.Tensor([eta])).to(
            self.device
        )  # noise variance for the proposal distribution

        self.num_samples = num_samples
        self.burn_in = burn_in

        self.v = None
        if v != None:
            self.v = v.repeat(1, self.num_samples, 1)

        # E-step params
        # Initialize the latent variable
        with torch.no_grad():
            data_in = self.X_abs_2.T.unsqueeze(1)  # (F, N) -> (N, 1, F)
            _ = self.dvae(data_in, v=v)
            _, Z, _ = self.dvae.inference(data_in, v=v)
        self.Z = Z.clone() + self.sqrt_eta * torch.randn_like(Z).to(device)  # (N, L=32)

        self.nepochs_E_step = max(nepochs_E_step, 10)
        self.nsamples_E_step = nsamples_E_step
        self.nsamples_WF = nsamples_WF

        self.dvae.train()  # vae in train mode

    def MH(self):
        self.Z = self.Z.squeeze()
        N, L = self.Z.shape

        Z_sampled = torch.zeros(N, self.nepochs_E_step - self.burn_in, L)

        cpt = 0
        averaged_acc_rate = 0  # acceptance probability
        for n in range(self.nepochs_E_step):
            U_x = self.loss(self.Z)[1]

            Z_tmp = self.Z + self.sqrt_eta * torch.randn_like(self.Z).to(self.device)
            U_y = self.loss(Z_tmp)[1]

            acc_prob = (
                -U_y
                + U_x
                + 0.5
                * (
                    torch.sum(
                        (
                            (
                                (self.Z - self.z_mean_p) ** 2
                                - (Z_tmp - self.z_mean_p) ** 2
                            )
                            / self.z_logvar_p.exp()
                        ),
                        dim=1,
                    ).unsqueeze(1)
                )
            )

            is_acc = (torch.log(torch.rand(N, 1).to(self.device)) < acc_prob).squeeze()

            # averaged_acc_rate += (
            #     torch.sum(is_acc).cpu().numpy()
            #     / np.prod(is_acc.shape)
            #     * 100
            #     / (self.nepochs_E_step)
            # )

            self.Z.data[is_acc, :] = Z_tmp.data[is_acc, :]

            if n > self.burn_in - 1:
                Z_sampled[:, cpt, :] = self.Z
                cpt += 1

        # print("averaged acceptance rate: %f" % (averaged_acc_rate))
        self.Z = self.Z.unsqueeze(1)

        return Z_sampled

    def loss(self, Z):
        # Forward
        Vs = self.compute_Vs(Z).transpose(0, -1)
        Vx = self.g * Vs + self.Vb[:, None, :]  # (F, N)

        # Compute loss and optimize
        self.z_mean_p, self.z_logvar_p = self.dvae.generation_z(Z, v=self.v)  # (N, L)

        loss_i = torch.sum(self.X_abs_2[:, None, :] / Vx + torch.log(Vx), dim=0)
        return loss_i.sum(), loss_i.squeeze(1).t()

    def compute_Vs(self, Z):
        if Z.ndim == 2:
            Z = Z.unsqueeze(1)
        Vs = torch.exp(
            self.dvae.generation_x(
                Z, v=self.v.repeat(1, Z.shape[1], 1) if self.v != None else None
            )
        )  # (N, R, L) -> (N, R, F)
        self.Vs = Vs.detach().permute(1, -1, 0)  # (R, F, N)
        return Vs

    def E_step(self):
        # Sample Z
        Z_mean = self.MH().to(self.device)

        # compute variance
        self.compute_Vs(Z_mean)
        self.compute_Vs_scaled()
        self.compute_Vx()

    def compute_WienerFilter(self):
        # Sample Z
        Z_mean = self.MH().to(self.device)
        # compute variance
        self.compute_Vs(Z_mean)  # (R, F, N)
        self.compute_Vs_scaled()
        self.compute_Vx()

        # compute Wiener Filters
        WFs = torch.mean(self.Vs_scaled / self.Vx, dim=0)  # (F, N)
        WFn = torch.mean(self.Vb / self.Vx, dim=0)  # (, N)

        WFs = WFs.cpu().detach().numpy()
        WFn = WFn.cpu().detach().numpy()

        return WFs, WFn
