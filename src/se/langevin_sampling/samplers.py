# Source: https://github.com/alisiahkoohi/Langevin-dynamics

import torch
from .precondSGLD import pSGLD
from .SGLD import SGLD
import copy


class LangevinDynamics(object):
    def __init__(self, x, func, lr=1e-2, lr_final=1e-4, max_itr=1e4, device="cpu"):
        self.x = x
        self.optim = SGLD([self.x], lr, weight_decay=0.0)
        self.lr = lr
        self.lr_final = lr_final
        self.max_itr = max_itr
        self.func = func
        self.lr_fn = self.decay_fn(lr=lr, lr_final=lr_final, max_itr=max_itr)
        self.counter = 0.0

    def sample(self):
        self.lr_decay()
        self.optim.zero_grad()
        loss = self.func(self.x)
        loss.backward()
        self.optim.step()
        self.counter += 1
        return copy.deepcopy(self.x.data), loss.item()

    def decay_fn(self, lr=1e-2, lr_final=1e-4, max_itr=1e4):
        if lr == lr_final:

            def lr_fn(t):
                return lr

        else:
            gamma = -0.55
            b = max_itr / ((lr_final / lr) ** (1 / gamma) - 1.0)
            a = lr / (b**gamma)

            def lr_fn(t, a=a, b=b, gamma=gamma):
                return a * ((b + t) ** gamma)

        return lr_fn

    def lr_decay(self):
        for param_group in self.optim.param_groups:
            param_group["lr"] = self.lr_fn(self.counter)


class MetropolisAdjustedLangevin(object):
    def __init__(self, x, func, lr=1e-2, lr_final=1e-4, max_itr=1e4, device="cpu"):
        self.x = [
            torch.zeros(x.shape, device=x.device, requires_grad=True),
            torch.zeros(x.shape, device=x.device, requires_grad=True),
        ]
        self.x[0].data = x.data
        self.x[1].data = x.data

        self.device = x.device

        self.loss = [
            torch.zeros([1], device=x.device),
            torch.zeros([1], device=x.device),
        ]
        self.loss[0] = func(self.x[0])
        self.loss[1].data = self.loss[0].data

        self.grad = [
            torch.zeros(x.shape, device=x.device),
            torch.zeros(x.shape, device=x.device),
        ]
        self.grad[0].data = torch.autograd.grad(
            self.loss[0], [self.x[0]], create_graph=False
        )[0].data
        self.grad[1].data = self.grad[0].data

        self.optim = SGLD([self.x[1]], lr, weight_decay=0.0)
        self.lr = lr
        self.lr_final = lr_final
        self.max_itr = max_itr
        self.func = func
        self.lr_fn = self.decay_fn(lr=lr, lr_final=lr_final, max_itr=max_itr)
        self.counter = 0.0

    def sample(self):
        accepted = False
        iter_up = 0
        self.lr_decay()
        while not accepted:
            self.x[1].grad = self.grad[1].data
            self.P = self.optim.step()
            self.loss[1] = self.func(self.x[1])
            self.grad[1].data = torch.autograd.grad(
                self.loss[1], [self.x[1]], create_graph=False
            )[0].data

            alpha = torch.tensor([min([1.0, self.sample_prob()])]).to(self.device)
            if torch.log(torch.rand([1]).to(self.device)) <= torch.log(alpha):
                self.grad[0].data = self.grad[1].data
                self.loss[0].data = self.loss[1].data
                self.x[0].data = self.x[1].data
                accepted = True
            else:
                self.x[1].data = self.x[0].data

            iter_up += 1

        self.counter += 1
        return copy.deepcopy(self.x[1].data), self.loss[1].item()

    def proposal_dist(self, idx):
        return (
            -(0.25 / self.lr_fn(self.counter))
            * torch.norm(
                self.x[idx]
                - self.x[idx ^ 1]
                - self.lr_fn(self.counter) * self.grad[idx ^ 1] / self.P
            )
            ** 2
        )

    def sample_prob(self):
        return (
            -self.loss[1] + self.loss[0] + self.proposal_dist(0) - self.proposal_dist(1)
        )

    def decay_fn(self, lr=1e-2, lr_final=1e-4, max_itr=1e4):
        if lr == lr_final:

            def lr_fn(t):
                return lr

        else:
            gamma = -0.55
            b = max_itr / ((lr_final / lr) ** (1 / gamma) - 1.0)
            a = lr / (b**gamma)

            def lr_fn(t, a=a, b=b, gamma=gamma):
                return a * ((b + t) ** gamma)

        return lr_fn

    def lr_decay(self):
        for param_group in self.optim.param_groups:
            param_group["lr"] = self.lr_fn(self.counter)
