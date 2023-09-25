#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Original script by Xiaoyu BIE (xiaoyu.bie@inrai.fr). Adapted and modified by Mosataf Sadeghi (mostafa.sadeghi@inrai.fr).
License agreement in LICENSE.txt

"""
import os
import numpy as np
import torch
import pickle
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from .em_algorithms import VEM, LDEM, MHEM, MALAEM
from src.utils import myconf
from src.model import build_RVAE


def set_require_grad(layer_to_optim):
    for layer in layer_to_optim:
        for para in layer.parameters():
            para.requires_grad = True


def enhance(
    mix_file,
    output_file,
    clean_file,
    video_file=[],
    algo_type="vem",
    ckpt_path="./",
    nmf_rank=8,
    niter=100,
    nepochs_E_step=1,
    nsamples_E_step=1,
    num_samples=10,
    nsamples_WF=1,
    optimizer="psgld",
    lr=5e-3,
    device="cuda",
    monitor_perf=False,
    verbose=False,
):
    # Read config
    path, fname = os.path.split(ckpt_path)
    cfg_file = os.path.join(path, "config.ini")
    cfg = myconf()
    cfg.read(cfg_file)
    beta = cfg.getfloat("Training", "beta")

    # Build DVAE model, set only encoder to be optimzed
    dvae = build_RVAE(cfg=cfg, device=device)
    layer_to_optim = [
        dvae.mlp_x_gx,
        dvae.rnn_g_x,
        dvae.mlp_z_gz,
        dvae.rnn_g_z,
        dvae.mlp_g_z,
        dvae.inf_mean,
        dvae.inf_logvar,
    ]

    # Load model weights and initialize the optimizer
    dvae.load_state_dict(torch.load(ckpt_path, map_location=device))

    ## Count the number of parameters in the model
    # num_params = sum(p.numel() for p in dvae.parameters() if p.requires_grad)
    # print("Number of parameters in the model:", num_params / 1_000_000)

    # Read STFT params
    wlen_sec = cfg.getfloat("STFT", "wlen_sec")
    hop_percent = cfg.getfloat("STFT", "hop_percent")
    fs = cfg.getint("STFT", "fs")
    zp_percent = cfg.getint("STFT", "zp_percent")
    wlen = wlen_sec * fs
    wlen = int(np.power(2, np.ceil(np.log2(wlen))))  # pwoer of 2
    hop = int(hop_percent * wlen)
    nfft = wlen + zp_percent * wlen
    win = np.sin(np.arange(0.5, wlen + 0.5) / wlen * np.pi)

    # Read mix audio
    x, fs_x = sf.read(mix_file)
    x = x / np.max(np.abs(x))
    T_orig = len(x)
    X = librosa.stft(x, n_fft=nfft, hop_length=hop, win_length=wlen, window=win)
    F, N = X.shape

    # Read video
    v = None

    # Initialize noise matrix
    eps = np.finfo(float).eps
    np.random.seed(23)
    W_init = np.maximum(np.random.rand(F, nmf_rank), eps)
    np.random.seed(23)
    H_init = np.maximum(np.random.rand(nmf_rank, N), eps)
    g_init = np.ones(N)

    if algo_type == "vem":
        for para in dvae.parameters():
            para.requires_grad = False
        set_require_grad(layer_to_optim)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, dvae.parameters()), lr=lr
        )
        # Load VEM algo
        SE_algo = VEM(
            X=X,
            v=v,
            W=W_init,
            H=H_init,
            g=g_init,
            dvae=dvae,
            optimizer=optimizer,
            beta=beta,
            niter=niter,
            nepochs_E_step=nepochs_E_step,
            nsamples_WF=nsamples_WF,
            device=device,
            verbose=verbose,
        )

    elif algo_type == "ldem":
        # Load LDEM algo
        SE_algo = LDEM(
            X=X,
            v=v,
            W=W_init,
            H=H_init,
            g=g_init,
            dvae=dvae,
            optimizer=optimizer,
            beta=beta,
            niter=niter,
            nepochs_E_step=nepochs_E_step,
            nsamples_WF=nsamples_WF,
            num_samples=num_samples,
            device=device,
            verbose=verbose,
        )

    elif algo_type == "mhem":
        # Load MHEM algo
        SE_algo = MHEM(
            X=X,
            v=v,
            W=W_init,
            H=H_init,
            g=g_init,
            dvae=dvae,
            beta=beta,
            niter=niter,
            nepochs_E_step=nepochs_E_step,
            nsamples_WF=nsamples_WF,
            device=device,
            verbose=verbose,
        )
    elif algo_type == "malaem":
        # Load MALAEM algo
        SE_algo = MALAEM(
            X=X,
            v=v,
            W=W_init,
            H=H_init,
            g=g_init,
            dvae=dvae,
            beta=beta,
            niter=niter,
            nepochs_E_step=nepochs_E_step,
            nsamples_WF=nsamples_WF,
            device=device,
            verbose=verbose,
        )
    else:
        raise ValueError(
            "Unknown SE algorithm type. Should be one of ldem, vem, mhem, or malaem."
        )

    # Run enhancement
    loss, time_consume = SE_algo.run_EM()

    S_hat = SE_algo.S_hat
    N_hat = SE_algo.N_hat

    s_hat = librosa.istft(
        S_hat, hop_length=hop, win_length=wlen, window=win, length=T_orig
    )
    n_hat = librosa.istft(
        N_hat, hop_length=hop, win_length=wlen, window=win, length=T_orig
    )

    sf.write(output_file, s_hat, fs_x)

    return s_hat, time_consume
