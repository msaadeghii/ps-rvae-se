# Posterior Sampling Algorithms for Speech Enhancement based on RVAE

This repository contains codes associated with the following paper:

> [Mostafa Sadeghi](https://msaadeghii.github.io/) and [Romain Serizel](https://members.loria.fr/RSerizel/), "[**Posterior Sampling Algorithms for Unsupervised Speech Enhancement with Recurrent Variational Autoencoder**](https://arxiv.org/abs/2309.10439),"  preprint, 2023.

This work presents efficient posterior sampling techniques based on Langevin dynamics and Metropolis-Hasting algorithms, adapted to the EM-based speech enhancement with RVAE.

`SE_demo.ipynb` provides a quick demo of different algorithms applied to a test noisy speech signal.

## Acknowledgement

In the development of this repository, we have largely utilised the [DVAE-SE](https://github.com/XiaoyuBIE1994/DVAE_SE) and [Langevin-dynamics](https://github.com/alisiahkoohi/Langevin-dynamics) repositories.

## Bibtex

```
@article{sadeghi2023posterior,
  title={Posterior sampling algorithms for unsupervised speech enhancement with recurrent variational autoencoder},
  author={Sadeghi, Mostafa and Serizel, Romain},
  journal={arXiv preprint arXiv:2309.10439},
  year={2023}
}

```
