"""
PyTorch modules for computing robust losses with
for (KL-regularized) CVaR, constrained-chi^2 and penalized-chi^2
uncertainty sets.
Includes losses appropriate for our porposed batch and MLMC gradient estimators
as well as losses for the dual-SGM and primal-dual methods.
"""

import torch
import torch.nn as nn
import numpy as np
import logging

GEOMETRIES = ('cvar', 'chi-square')
MIN_REL_DIFFERENCE = 1e-5


def chi_square_value(p, v, reg):
    """Returns <p, v> - reg * chi^2(p, uniform) for Torch tensors"""
    m = p.shape[0]

    with torch.no_grad():
        chi2 = (0.5 / m) * reg * (torch.norm(m * p - torch.ones(m, ), p=2) ** 2)

    return torch.dot(p, v) - chi2


def cvar_value(p, v, reg):
    """Returns <p, v> - reg * KL(p, uniform) for Torch tensors"""
    m = p.shape[0]

    with torch.no_grad():
        idx = torch.nonzero(p)  # where is annoyingly backwards incompatible
        kl = np.log(m) + (p[idx] * torch.log(p[idx])).sum()

    return torch.dot(p, v) - reg * kl


def fenchel_kl_cvar(v, alpha):
    """Returns the empirical mean of the Fenchel dual for KL CVaR"""
    v -= np.log(1 / alpha)
    v1 = v[torch.lt(v, 0)]
    v2 = v[torch.ge(v, 0)]
    w1 = torch.exp(v1) / alpha - 1
    w2 = (v2 + 1) * (1 / alpha) - 1
    return (w1.sum() + w2.sum()) / v.shape[0]


def bisection(eta_min, eta_max, f, tol=1e-6, max_iter=500):
    """Expects f an increasing function and return eta in [eta_min, eta_max]
    s.t. |f(eta)| <= tol (or the best solution after max_iter iterations"""
    lower = f(eta_min)
    upper = f(eta_max)

    # until the root is between eta_min and eta_max, double the length of the
    # interval starting at either endpoint.
    while lower > 0 or upper < 0:
        length = eta_max - eta_min
        if lower > 0:
            eta_max = eta_min
            eta_min = eta_min - 2 * length
        if upper < 0:
            eta_min = eta_max
            eta_max = eta_max + 2 * length

        lower = f(eta_min)
        upper = f(eta_max)

    for _ in range(max_iter):
        eta = 0.5 * (eta_min + eta_max)

        v = f(eta)

        if torch.abs(v) <= tol:
            return eta

        if v > 0:
            eta_max = eta
        elif v < 0:
            eta_min = eta

    # if the minimum is not reached in max_iter, returns the current value
    logging.warning('Maximum number of iterations exceeded in bisection')
    return 0.5 * (eta_min + eta_max)


def huber_loss(x, delta=1.):
    """ Standard Huber loss of parameter delta
    https://en.wikipedia.org/wiki/Huber_loss
    returns 0.5 * x^2 if |a| <= \delta
            \delta * (|a| - 0.5 * \delta) o.w.
    """
    if torch.abs(x) <= delta:
        return 0.5 * (x ** 2)
    else:
        return delta * (torch.abs(x) - 0.5 * delta)


class RobustLoss(nn.Module):
    """PyTorch module for the batch robust loss estimator"""
    def __init__(self, size, reg, geometry, tol=1e-4,
                 max_iter=1000, debugging=False):
        """
        Parameters
        ----------
        size : float
            Size of the uncertainty set (\rho for \chi^2 and \alpha for CVaR)
            Set float('inf') for unconstrained
        reg : float
            Strength of the regularizer, entropy if geometry == 'cvar'
            $\chi^2$ divergence if geometry == 'chi-square'
        geometry : string
            Element of GEOMETRIES
        tol : float, optional
            Tolerance parameter for the bisection
        max_iter : int, optional
            Number of iterations after which to break the bisection
        """
        super().__init__()
        self.size = size
        self.reg = reg
        self.geometry = geometry
        self.tol = tol
        self.max_iter = max_iter
        self.debugging = debugging

        self.is_erm = size == 0

        if geometry not in GEOMETRIES:
            raise ValueError('Geometry %s not supported' % geometry)

        if geometry == 'cvar' and self.size > 1:
            raise ValueError(f'alpha should be < 1 for cvar, is {self.size}')

    def best_response(self, v, decayed_rho=None):
        if decayed_rho is not None:
            size = decayed_rho
        else:
            size = self.size
        reg = self.reg
        m = v.shape[0]

        if self.geometry == 'cvar':
            if self.reg > 0:
                if size == 1.0:
                    return torch.ones_like(v) / m

                def p(eta):
                    x = (v - eta) / reg
                    return torch.min(torch.exp(x),
                                     torch.Tensor([1 / size]).type(x.dtype)) / m

                def bisection_target(eta):
                    return 1.0 - p(eta).sum()

                eta_min = reg * torch.logsumexp(v / reg - np.log(m), 0)
                eta_max = v.max()

                if torch.abs(bisection_target(eta_min)) <= self.tol:
                    return p(eta_min)
            else:
                cutoff = int(size * m)
                surplus = 1.0 - cutoff / (size * m)

                p = torch.zeros_like(v)
                idx = torch.argsort(v, descending=True)
                p[idx[:cutoff]] = 1.0 / (size * m)
                if cutoff < m:
                    p[idx[cutoff]] = surplus
                return p

        if self.geometry == 'chi-square':
            if (v.max() - v.min()) / v.max() <= MIN_REL_DIFFERENCE:
                return torch.ones_like(v) / m

            if size == float('inf'):
                assert reg > 0

                def p(eta):
                    return torch.relu(v - eta) / (reg * m)

                def bisection_target(eta):
                    return 1.0 - p(eta).sum()

                eta_min = min(v.sum() - reg * m, v.min())
                eta_max = v.max()

            else:
                assert size < float('inf')

                # failsafe for batch sizes small compared to
                # uncertainty set size
                if m <= 1 + 2 * size:
                    out = (v == v.max()).float()
                    out /= out.sum()
                    return out

                if reg == 0:
                    def p(eta):
                        pp = torch.relu(v - eta)
                        return pp / pp.sum()

                    def bisection_target(eta):
                        pp = p(eta)
                        w = m * pp - torch.ones_like(pp)
                        return 0.5 * torch.mean(w ** 2) - size

                    eta_min = -(1.0 / (np.sqrt(2 * size + 1) - 1)) * v.max()
                    eta_max = v.max()
                else:
                    def p(eta):
                        pp = torch.relu(v - eta)

                        opt_lam = max(
                            reg, torch.norm(pp) / np.sqrt(m * (1 + 2 * size))
                        )

                        return pp / (m * opt_lam)

                    def bisection_target(eta):
                        return 1 - p(eta).sum()

                    eta_min = v.min() - 1
                    eta_max = v.max()

        eta_star = bisection(
            eta_min, eta_max, bisection_target,
            tol=self.tol, max_iter=self.max_iter)

        if self.debugging:
            return p(eta_star), eta_star
        return p(eta_star)

    def forward(self, v, decayed_rho=None):
        """Value of the robust loss
        Note that the best response is computed without gradients
        Parameters
        ----------
        v : torch.Tensor
            Tensor containing the individual losses on the batch of examples
        Returns
        -------
        loss : torch.float
            Value of the robust loss on the batch of examples
        """
        with torch.no_grad():
            p = self.best_response(v, decayed_rho)
        return p