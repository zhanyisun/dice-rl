import math
import torch
import torch.nn.functional as F
from torch import nn
from collections import namedtuple
import os
import numpy as np
from PIL import Image

Sample = namedtuple("Sample", "trajectories")

class FlowMatchingModel(nn.Module):
    def __init__(
        self,
        network,
        horizon_steps: int,
        obs_dim: int,
        action_dim: int,
        # core hyperparams
        flow_steps: int = 100,
        integration_method: str = "euler",    # "euler" | "heun" | "rk4"
        schedule: str = "linear",           # "cosine" | "linear"
        init_scale: float = 1.0,
        # loss hyperparams
        t_sampling: str = "uniform",        # "uniform" | "beta" | "stratified"
        weight_method: str = "none",        # "none" | "importance"
        device: str = "cuda:0",
        cond_steps: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.network    = network.to(device)
        self.device     = device
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.cond_steps = cond_steps

        # integration
        self.flow_steps = int(flow_steps)
        self.integration_method = integration_method
        self.schedule   = schedule
        self.init_scale = init_scale

        # loss/t sampling
        self.t_sampling = t_sampling
        self.weight_method = weight_method
        
        print('FlowMatchingModel initialized with:')
        print(f'  integration_method: {integration_method}')
        print(f'  schedule: {schedule}')
        print(f'  t_sampling: {t_sampling}')
        print(f'  weight_method: {weight_method}')
        print(f'  flow_steps: {flow_steps}')
        
    @torch.no_grad()
    def forward(self, 
                cond, 
                init_noise: torch.Tensor = None,
                deterministic=True,
                ):
        B = cond["state"].shape[0]
        H, D = self.horizon_steps, self.action_dim

        if init_noise is not None:
            x = self.init_scale * init_noise
        else:
            x = self.init_scale * torch.randn((B, H, D), device=self.device)

        # build t grid in [1→0]
        if self.schedule == "cosine":
            θ = torch.linspace(0, math.pi/2, self.flow_steps+1, device=self.device)
            t_steps = torch.cos(θ)**2
            t_steps = torch.flip(t_steps, dims=[0])  # 1→0
        else:
            t_steps = torch.linspace(0.0, 1.0, self.flow_steps + 1, device=self.device)

        # integrate
        for i in range(self.flow_steps):
            t0, t1 = t_steps[i], t_steps[i+1]
            dt = t1 - t0
            t0b = torch.full((B,), t0, device=self.device)

            if self.integration_method == "euler":
                v0 = self.network(x, t0b, cond=cond)
                x = x + dt * v0

            elif self.integration_method == "heun":
                v0 = self.network(x, t0b, cond=cond)
                x_pred = x + dt * v0
                t1b = torch.full((B,), t1, device=self.device)
                v1 = self.network(x_pred, t1b, cond=cond)
                x = x + dt * 0.5 * (v0 + v1)

            elif self.integration_method == "rk4":
                k1 = self.network(x,        t0b,           cond=cond)
                k2 = self.network(x + 0.5*dt*k1, t0b + 0.5*dt, cond=cond)
                k3 = self.network(x + 0.5*dt*k2, t0b + 0.5*dt, cond=cond)
                k4 = self.network(x +    dt*k3,  torch.full((B,), t1, device=self.device), cond=cond)

                x = x + dt*(k1 + 2*k2 + 2*k3 + k4)/6

            else:
                raise ValueError(f"Unknown integration_method='{self.integration_method}'")

        return Sample(x)        
    
    @torch.no_grad()
    def surrogate_loglik_rf(
        self,
        a: torch.Tensor,          # [B, H, D] action to score
        cond: dict,               # contains 'state': [B, ...]
        k_t: int = 8,             # # of t samples
        k_x0: int = 4,            # # of base samples per t
    ) -> torch.Tensor:
        B, H, D = a.shape
        device = a.device
        t = torch.rand(k_t, device=device).clamp(1e-5, 1-1e-5)           # [k_t]
        x0 = torch.randn(k_x0, B, H, D, device=device)                   # [k_x0,B,H,D]

        tB = t.view(k_t, 1, 1, 1, 1).expand(-1, k_x0, B, H, D)
        aE = a.view(1, 1, B, H, D).expand(k_t, k_x0, -1, -1, -1)
        x0E = x0.view(1, k_x0, B, H, D).expand(k_t, -1, -1, -1, -1)

        x_t = (1.0 - tB) * x0E + tB * aE                     # [k_t,k_x0,B,H,D]
        v_tgt = (aE - x0E)                                   # [k_t,k_x0,B,H,D]
        
        x_t_flat = x_t.reshape(-1, H, D)
        t_flat   = tB[..., 0, 0].reshape(-1)                 # broadcasted t per sample
        
        cond_rep = {}
        for k, v in cond.items():
            if k == "state":
                # v is [B, cond_steps, state_dim]
                # [k_t*k_x0*B, cond_steps, state_dim]
                v_exp = v.unsqueeze(0).unsqueeze(0)  # [1, 1, B, cond_steps, state_dim]
                v_exp = v_exp.expand(k_t, k_x0, -1, -1, -1)  # [k_t, k_x0, B, cond_steps, state_dim]
                cond_rep[k] = v_exp.reshape(-1, v.shape[1], v.shape[2])  # [k_t*k_x0*B, cond_steps, state_dim]
            else:
                cond_rep[k] = v

        v_pred = self.network(x_t_flat, t_flat, cond=cond_rep).reshape(k_t, k_x0, B, H, D)
        err = (v_pred - v_tgt).pow(2).mean(dim=(3,4))        # [k_t,k_x0,B]
        fisher = err.mean(dim=(0,1))                         # [B]  (expectation over t,x0)

        # likelihood proxy negate so higher = better
        return -fisher

    def loss(self, x_start, cond):
        B = x_start.shape[0]
        device = x_start.device
        t = torch.rand(B, device=device)

        t = t.clamp(1e-5, 1-1e-5)

        # noisy interpolation
        noise = torch.randn_like(x_start)
        x_t   = t.view(B,1,1)*x_start + (1-t).view(B,1,1)*noise
        v_tgt = x_start - noise

        v_pred = self.network(x_t, t, cond=cond)

        if self.weight_method == "none":
            flow_loss = F.mse_loss(v_pred, v_tgt)
        elif self.weight_method == "importance":
            w = (1.0/(1-t)).view(B,1,1)
            flow_loss = (w * (v_pred - v_tgt).pow(2)).mean()
        else:
            raise ValueError(f"Unknown weight_method='{self.weight_method}'")
        
        return flow_loss
    
    def forward_from_features(self, features: torch.Tensor, init_noise: torch.Tensor = None):
        B = features.shape[0] if features.ndim >= 2 else 1
        H, D = self.horizon_steps, self.action_dim
        
        # init noise - same as original forward
        if init_noise is not None:
            x = self.init_scale * init_noise
        else:
            x = self.init_scale * torch.randn((B, H, D), device=features.device)

        t_steps = torch.linspace(0.0, 1.0, self.flow_steps + 1, device=features.device)

        # integrate - same as original forward (euler only for simplicity)
        for i in range(self.flow_steps):
            t0, t1 = t_steps[i], t_steps[i+1]
            dt = t1 - t0
            t0b = torch.full((B,), t0, device=features.device)

            if self.integration_method == "euler":
                # Use forward_from_features if available, otherwise fall back to regular forward
                v0 = self.network.forward_from_features(x=x, time=t0b, state=features)
                x = x + dt * v0

        return Sample(x)
    
    def _base_log_prob(self, z0: torch.Tensor) -> torch.Tensor:
        B, H, D = z0.shape
        dim = H * D
        quad = 0.5 * (z0 ** 2).view(B, -1).sum(dim=1)
        const = 0.5 * dim * math.log(2 * math.pi)
        return -(quad + const)

    def _divergence_from_v_hutch(self, x, v, n_mc: int = 1) -> torch.Tensor:
        B = x.shape[0]
        div_total = torch.zeros(B, device=x.device, dtype=x.dtype)
        if (not x.requires_grad) or (not v.requires_grad) or (v.grad_fn is None):
            return div_total
        for k in range(n_mc):
            e = torch.randn_like(x) 
            ve = (v * e).sum()       # scalar v·e
            # grad_x (v·e) = (J_v)^T e
            grad = torch.autograd.grad(
                ve, x,
                create_graph=False,
                retain_graph=(k < n_mc - 1), 
                only_inputs=True
            )[0]
            # e^T (J_v)^T e  = (grad · e)
            div_est = (grad * e).view(B, -1).sum(dim=1)
            div_total += div_est

        return div_total / float(n_mc)

    @torch.no_grad()
    def log_prob(
        self,
        a: torch.Tensor,                  # [B, H, D] actions to score (x1)
        cond: dict,                       # contains 'state': [B, ...]
        n_mc: int = 4,
        steps: int = None,
        solver: str = "euler",
    ) -> torch.Tensor:
        assert solver == "euler", "This example shows Euler for clarity."
        steps = steps or self.flow_steps
        device = a.device
        B = a.shape[0]

        # reverse time grid: 1 -> 0  (dt is negative)
        t_grid = torch.linspace(1.0, 0.0, steps + 1, device=device)

        # init augmented state at t=1
        z = a.clone()                     # z_1 = a
        f = torch.zeros(B, device=device) # accumulator for ∫ div v dt

        self.network.eval()

        for i in range(steps):
            t = t_grid[i]                 # current time
            t_next = t_grid[i + 1]        # next time
            dt = t_next - t               # dt < 0
            t_batch = torch.full((B,), t.item(), device=device)

            # ---- evaluate both v and div at the same (z_old, t) ----
            with torch.enable_grad():
                z_old = z.detach().requires_grad_(True)
                v = self.network(x=z_old, time=t_batch, cond=cond)   # [B, H, D]
                div = self._divergence_from_v_hutch(z_old, v, n_mc=n_mc)  # [B]
                v = v.detach()

            # z' = -v(z,t)  =>  z_{t+dt} = z_t + dt * v(z_t, t)
            z = z + dt * v
            # f' =  div v(z,t)  =>  f_{t+dt} = f_t + dt * div(z_t, t)
            f = f - dt * div

        # at t=0, (z0, f0)
        log_p0 = self._base_log_prob(z)   # [B]
        return log_p0 - f
    
    def log_prob_from_features(
        self,
        a: torch.Tensor,                  # [B, H, D] actions to score
        features: torch.Tensor,           # [B, feature_dim] pre-extracted visual features
        n_mc: int = 1,
        steps: int = 10,
        solver: str = "euler",
    ) -> torch.Tensor:
        assert solver == "euler", "This example shows Euler for clarity."
        assert steps == 10
        steps = self.flow_steps
        device = a.device
        B = a.shape[0]

        # reverse time grid: 1 -> 0  (dt is negative)
        t_grid = torch.linspace(1.0, 0.0, steps + 1, device=device)

        # init augmented state at t=1
        z = a.clone()                     # z_1 = a
        f = torch.zeros(B, device=device) # accumulator for ∫ div v dt

        self.network.eval()

        for i in range(steps):
            t = t_grid[i]                 # current time
            t_next = t_grid[i + 1]        # next time
            dt = t_next - t               # dt < 0
            t_batch = torch.full((B,), t.item(), device=device)

            with torch.enable_grad():
                z_old = z.detach().requires_grad_(True)
                v = self.network.forward_from_features(x=z_old, time=t_batch, state=features)   # [B, H, D]
                div = self._divergence_from_v_hutch(z_old, v, n_mc=n_mc)  # [B]
                v = v.detach()

            # z' = -v(z,t)  =>  z_{t+dt} = z_t + dt * v(z_t, t)
            z = z + dt * v
            # f' =  div v(z,t)  =>  f_{t+dt} = f_t + dt * div(z_t, t)
            f = f - dt * div

        log_p0 = self._base_log_prob(z)   # [B]
        return log_p0 - f
