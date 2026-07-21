"""Separable CMA-ES (diagonal covariance) — the gradient-free optimizer.

This is the optimizer TinyRouter and Sakana-Fugu-base use to train the routing
head on a binary/soft reward without gradients. The "separable" variant keeps a
diagonal covariance, so cost scales linearly with the parameter count (Ros &
Hansen, 2008) — the right choice for ~10^3-10^4-parameter heads.

Standard CMA-ES equations (Hansen's tutorial) with the diagonal restriction and
the (n+2)/3 learning-rate acceleration for the separable case. The optimizer
*minimizes*; callers maximizing a score pass its negation.
"""

from __future__ import annotations

import numpy as np


class SepCMAES:
    def __init__(self, x0: np.ndarray, sigma0: float = 0.3, seed: int = 0,
                 popsize: int | None = None):
        self.n = x0.size
        self.mean = x0.astype(np.float64).copy()
        self.sigma = float(sigma0)
        self.rng = np.random.default_rng(seed)

        n = self.n
        self.lam = popsize or (4 + int(np.floor(3 * np.log(n))))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mu_eff = 1.0 / np.sum(self.w ** 2)

        # adaptation rates
        self.c_sigma = (self.mu_eff + 2) / (n + self.mu_eff + 5)
        self.d_sigma = (1 + 2 * max(0.0, np.sqrt((self.mu_eff - 1) / (n + 1)) - 1)
                        + self.c_sigma)
        self.c_c = (4 + self.mu_eff / n) / (n + 4 + 2 * self.mu_eff / n)
        c1 = 2 / ((n + 1.3) ** 2 + self.mu_eff)
        cmu = min(1 - c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff)
                  / ((n + 2) ** 2 + self.mu_eff))
        accel = (n + 2) / 3.0  # separable acceleration
        self.c_1 = min(1.0, c1 * accel)
        self.c_mu = min(1 - self.c_1, cmu * accel)

        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        self.p_sigma = np.zeros(n)
        self.p_c = np.zeros(n)
        self.C = np.ones(n)          # diagonal of covariance
        self.gen = 0
        self._last_z: np.ndarray | None = None

    def ask(self) -> np.ndarray:
        """Return a (lam, n) population of candidate solutions."""
        z = self.rng.standard_normal((self.lam, self.n))
        self._last_z = z
        y = z * np.sqrt(self.C)                    # (lam, n)
        return self.mean + self.sigma * y

    def tell(self, solutions: np.ndarray, fitness: np.ndarray) -> None:
        """Update the distribution. `fitness` is minimized (lower is better)."""
        order = np.argsort(fitness)                # ascending -> best first
        sel = order[:self.mu]
        y = (solutions[sel] - self.mean) / self.sigma     # (mu, n)
        z = self._last_z[sel]                             # (mu, n)

        y_w = self.w @ y                                  # (n,)
        z_w = self.w @ z                                  # (n,) = C^{-1/2} y_w for diag C

        self.mean = self.mean + self.sigma * y_w

        self.p_sigma = ((1 - self.c_sigma) * self.p_sigma
                        + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * z_w)
        ps_norm = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (ps_norm / self.chiN - 1))

        denom = np.sqrt(1 - (1 - self.c_sigma) ** (2 * (self.gen + 1)))
        h_sigma = 1.0 if ps_norm / denom < (1.4 + 2 / (self.n + 1)) * self.chiN else 0.0

        self.p_c = ((1 - self.c_c) * self.p_c
                    + h_sigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_w)

        delta_h = (1 - h_sigma) * self.c_c * (2 - self.c_c)
        rank_mu = self.w @ (y ** 2)                       # (n,)
        self.C = ((1 - self.c_1 - self.c_mu) * self.C
                  + self.c_1 * (self.p_c ** 2 + delta_h * self.C)
                  + self.c_mu * rank_mu)
        self.C = np.maximum(self.C, 1e-20)
        self.gen += 1
