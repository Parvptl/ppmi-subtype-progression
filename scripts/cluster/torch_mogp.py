"""PyTorch port of `mogp.objective`, used ONLY to benchmark GPU acceleration.

This is not part of the scientific pipeline. It exists to answer one question
before any experiment is launched: does moving the multi-output GP marginal
likelihood onto the A30 GPU make a fit faster than the audited NumPy code?

The port computes exactly the quantity `ppmi_subtype.mogp.objective` computes
--- the negative log marginal likelihood with the linear mean profiled out by
GLS, summed over patients --- and gets the gradient from autograd instead of
the hand-written `_dK_list`. `benchmark_workloads.py` checks value and gradient
against the NumPy implementation before timing anything; if they disagree the
timing is not reported.
"""

from __future__ import annotations

import numpy as np
import torch

from ppmi_subtype.gpr import KERNEL_COMPONENTS, SQRT3
from ppmi_subtype.mogp import Packed, parameter_layout


def _chol2(l00, l11, l10):
    zero = torch.zeros((), dtype=l00.dtype, device=l00.device)
    L = torch.stack([torch.stack([l00, zero]), torch.stack([l10, l11])])
    return L @ L.T


class TorchMOGP:
    """Holds a packed cohort on one device and evaluates the objective there."""

    def __init__(self, packed: Packed, kernel: str, coregion: str,
                 jitter: float = 1e-8, device: str = "cpu"):
        self.kernel, self.coregion, self.jitter = kernel, coregion, jitter
        self.layout = parameter_layout(kernel, coregion)
        dev = torch.device(device)
        self.device = dev
        f64 = torch.float64
        T = torch.as_tensor(packed.T, dtype=f64, device=dev)
        mask = torch.as_tensor(packed.mask, device=dev)
        M, n = T.shape
        self.M, self.n = M, n
        self.T = T
        self.valid = (mask[:, :, None] & mask[:, None, :]).to(f64)
        self.dist = (T[:, :, None] - T[:, None, :]).abs()
        self.lin = T[:, :, None] * T[:, None, :]
        mask2 = mask.repeat_interleave(2, dim=1)
        self.diag_add = mask2.to(f64) * jitter + (~mask2).to(f64)
        y = torch.as_tensor(packed.Y, dtype=f64, device=dev).reshape(M, 2 * n)
        self.y = y * mask2.to(f64)
        X = torch.zeros((M, 2 * n, 4), dtype=f64, device=dev)
        m = mask.to(f64)
        X[:, 0::2, 0] = m
        X[:, 0::2, 1] = T * m
        X[:, 1::2, 2] = m
        X[:, 1::2, 3] = T * m
        self.X = X
        self.eye_n = torch.eye(n, dtype=f64, device=dev).expand(M, n, n)

    def _expand(self, block, B):
        M, n = self.M, self.n
        out = (block * self.valid)[:, :, None, :, None] * B[None, None, :, None, :]
        return out.reshape(M, 2 * n, 2 * n)

    def nll(self, theta: torch.Tensor) -> torch.Tensor:
        vals = {}
        for (name, role), raw in zip(self.layout, theta):
            vals[name] = torch.exp(raw) if role == "log" else raw
        zero = torch.zeros((), dtype=theta.dtype, device=theta.device)
        K = torch.zeros((self.M, 2 * self.n, 2 * self.n), dtype=theta.dtype,
                        device=theta.device)
        for c in KERNEL_COMPONENTS[self.kernel]:
            B = _chol2(vals[f"B_{c}_L00"], vals[f"B_{c}_L11"],
                       vals.get(f"B_{c}_L10", zero))
            if c == "const":
                k = torch.ones_like(self.dist)
            elif c == "linear":
                k = self.lin
            elif c == "rbf":
                k = torch.exp(-0.5 * (self.dist / vals[f"lengthscale_{c}"]) ** 2)
            else:
                a = SQRT3 * self.dist / vals[f"lengthscale_{c}"]
                k = (1.0 + a) * torch.exp(-a)
            K = K + self._expand(k, B)
        Se = _chol2(vals["noise_L00"], vals["noise_L11"], vals.get("noise_L10", zero))
        K = K + self._expand(self.eye_n, Se)
        K = K + torch.diag_embed(self.diag_add)

        L = torch.linalg.cholesky(K)
        Kinv_X = torch.cholesky_solve(self.X, L)
        Kinv_y = torch.cholesky_solve(self.y[:, :, None], L)[:, :, 0]
        A = torch.einsum("pia,pib->ab", self.X, Kinv_X)
        b = torch.einsum("pia,pi->a", self.X, Kinv_y)
        beta = torch.linalg.solve(A, b)
        resid = self.y - torch.einsum("pia,a->pi", self.X, beta)
        alpha = torch.cholesky_solve(resid[:, :, None], L)[:, :, 0]
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum()
        return 0.5 * (logdet + (resid * alpha).sum())

    def objective(self, theta_np: np.ndarray):
        """(value, gradient) as NumPy, the signature scipy.optimize expects."""
        theta = torch.tensor(theta_np, dtype=torch.float64, device=self.device,
                             requires_grad=True)
        try:
            val = self.nll(theta)
            (g,) = torch.autograd.grad(val, theta)
        except RuntimeError:          # Cholesky failure
            return 1e12, np.zeros_like(theta_np)
        v = float(val.detach().cpu())
        if not np.isfinite(v):
            return 1e12, np.zeros_like(theta_np)
        return v, g.detach().cpu().numpy()
