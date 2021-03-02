import torch.nn.functional as F
import torch
import math
import numpy as np
from scipy import optimize
import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

labelsize = 10
legendsize = 10
mpl.rcParams['xtick.labelsize'] = labelsize
mpl.rcParams['ytick.labelsize'] = labelsize
mpl.rcParams['font.size'] = labelsize
plt.style.use('seaborn-deep')
colormap = plt.cm.gist_ncar


def bisection(eta_min, eta_max, f, tol=1e-6, max_iter=1000):
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
        print("lower = {}, upper = {}".format(lower, upper))

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
    print('Maximum number of iterations exceeded in bisection')

    print(eta_min, eta_max)
    return 0.5 * (eta_min + eta_max)


def compute_best_response(baselined_losses, rho, p_train, tol=1e-4):
    # losses and p_train are tensors
    def p(eta):
        pp = torch.relu(baselined_losses - eta)
        q = pp * p_train / (pp * p_train).sum()
        # fixme: originally, we should return q; can I add the following
        #  constraint to prevent that some groups are assigned 0 mass in q;
        #  if so, what is the best form instead of this hard clipping?
        cq = torch.clamp(q/p_train, min=0.1)
        return cq * p_train / (cq * p_train).sum()

    def bisection_target(eta):
        pp = p(eta)
        w = pp - p_train
        return 0.5 * torch.sum(w ** 2) - rho

    eta_min = -(1.0 / (np.sqrt(2 * rho + 1) - 1)) * baselined_losses.max()
    eta_max = baselined_losses.max()
    eta_star = bisection(
        eta_min, eta_max, bisection_target,
        tol=tol, max_iter=1000)

    q = p(eta_star)
    print("eta_star: ", eta_star)
    print(baselined_losses)
    print(p_train)
    print(q)
    input()
    return q


def project_to_cs_ball(v, rho, p_train):
    """Numpy/Scipy projection to chi-square ball of radius rho"""
    n = len(v)

    def cs_div(p):
        return 0.5 * np.mean((p / p_train - 1)**2)

    # first, check if a simplex projection is within the chi-square ball
    target_simplex = lambda eta: np.sum(np.maximum(v - eta, 0)) - 1.0
    eta_min_simplex = v.min() - 1 / n
    eta_max_simplex = v.max()
    eta_simplex = optimize.brentq(
        target_simplex, eta_min_simplex, eta_max_simplex)
    p_candidate = np.maximum(v - eta_simplex, 0)
    if cs_div(p_candidate) <= rho:
        return p_candidate

    # second, compute a chi-square best response
    def target_cs(eta, return_p=False):
        p = np.maximum(v - eta, 0)
        if p.sum() == 0.0:
            p[np.argmax(v)] = 1.0
        else:
            p /= p.sum()
        err = cs_div(p) - rho
        return p if return_p else err
    eta_max_cs = v.max()
    eta_min_cs = v.min()
    if target_cs(eta_max_cs) <= 0:
        return target_cs(eta_max_cs, return_p=True)
    while target_cs(eta_min_cs) > 0.0:  # find left interval edge for bisection
        eta_min_cs = 2 * eta_min_cs - eta_max_cs
    eta_cs = optimize.brentq(
        target_cs, eta_min_cs, eta_max_cs)
    p_candidate = target_cs(eta_cs, return_p=True)
    assert np.abs(cs_div(p_candidate) - rho) < rho * 1e-2
    return p_candidate


def compute_primal_dual_q(q_last, reduce_group_losses, rho=1.0, step_size=0.01, clip=None):
    # all of the args are numpy arrays
    np_group_losses = reduce_group_losses
    # fixme: or as in your code, reduce_group_losses[i] is the sum of losses of group i instead of mean
    #  and self.step_size / (batch_size * (self.h_fun + 1e-8))?
    coefs = step_size / ((q_last + 1e-8) * len(reduce_group_losses))
    q_update = coefs * np_group_losses
    if clip is not None:
        q_update = np.minimum(q_update, clip)
    q = q_last + q_update
    q = project_to_cs_ball(q, rho, p_train)
    print(q)


if __name__ == '__main__':
    # debug best response
    p_train = np.array([0.016749707678881984, 0.24883687424058096, 0.12823659011863794, 0.24883687424058096, 0.09388764567221664, 0.00672033746261537, 0.008906120565944633, 0.05648945416885125, 0.1876013543693391, 0.00373504148235112])
    rho = 0.1
    # losses[i] is the average losses of group i in a batch
    losses = np.array([7.150322, 5.925216, 6.436857, 5.886299, 6.113926, 7.217082, 6.935157, 6.830746, 5.37761,  7.416435])

    print(np.argsort(p_train))
    print(np.argsort(losses))
    best_q = compute_best_response(torch.from_numpy(losses), rho, torch.from_numpy(p_train))

    # debug primal dual
    q = p_train
    # fixme: for the current implementation, it's super sensitive to step_size,
    #  e.g. step_size < 1e-5, can lead to the assertion of line 126 break.
    step_size = 1e-5
    new_q = compute_primal_dual_q(q, losses, rho, step_size)











