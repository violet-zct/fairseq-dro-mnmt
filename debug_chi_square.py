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


def compute_best_response(v, size, p_train, reg=0, tol=1e-4, max_iter=1000):
    m = v.shape[0]
        
    if m <= 1 + 2 * size:
        out = (v == v.max()).float()
        out /= out.sum()
        return out

    if reg == 0:
        def p(eta):
            pp = p_train * torch.relu(v - eta)
            q = pp / pp.sum()
            cq = torch.clamp(q / p_train, min=0.2)
            return cq * p_train / (cq * p_train).sum()

        def bisection_target(eta):
            pp = p(eta)
            return 0.5 * (p_train * ((pp / p_train - 1) ** 2)).sum() - size

        eta_min = -(1.0 / (np.sqrt(2 * size + 1) - 1)) * v.max()
        eta_max = v.max()
    else:
        def p(eta):
            pp = p_train * torch.relu(v - eta)
            
            opt_lam = torch.sqrt((p_train * (torch.relu(v - eta) ** 2)).sum())
            opt_lam = max(
                reg, opt_lam / np.sqrt(1 + 2 * size)
            )

            return pp / opt_lam

        def bisection_target(eta):
            return 1 - p(eta).sum()

        eta_min = v.min() - 1
        eta_max = v.max()

    eta_star = bisection(
        eta_min, eta_max, bisection_target,
        tol=tol, max_iter=max_iter)
    return p(eta_star)


def project_to_cs_ball(v, rho, p_train):
    """Numpy/Scipy projection to chi-square ball of radius rho"""
    n = len(v)

    def cs_div(p):
        return 0.5 * (np.square(p / p_train - 1) * p_train).sum()

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
        p = p_train * np.maximum(v - eta, 0)
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
    coefs = step_size * q_last / p_train
    q_update = coefs * np_group_losses
    if clip is not None:
        q_update = np.minimum(q_update, clip)
    q = q_last + q_update
    # print(q)
    q = project_to_cs_ball(q, rho, p_train)
    return q


if __name__ == '__main__':
    # debug best response
    # def compute_best_response(v, size, p_train, reg=0, tol=1e-4, max_iter=1000):
    p_train = np.array([0.016749707678881984, 0.24883687424058096, 0.12823659011863794, 0.24883687424058096, 0.09388764567221664, 0.00672033746261537, 0.008906120565944633, 0.05648945416885125, 0.1876013543693391, 0.00373504148235112])
    rho = 1.0
    # losses[i] is the average losses of group i in a batch
    losses = np.array([7.150322, 5.925216, 6.436857, 5.886299, 6.113926, 7.217082, 6.935157, 6.830746, 5.37761,  7.416435])
    # losses = np.array([2.915552, 3.078025, 3.135855, 3.118232, 2.828697, 3.115064, 3.074514, 3.129607, 2.574514, 3.229607])
    # print(np.argsort(p_train))
    # print(np.argsort(losses))
    best_q = compute_best_response(torch.from_numpy(losses), rho, torch.from_numpy(p_train))
    print("p train: ")
    print(" ".join(["{:.6f}".format(ii) for ii in p_train]))
    print("best response q: ")
    print(" ".join(["{:.6f}".format(ii) for ii in best_q.numpy()]))

    # debug primal dual
    # rho roughly checking 0.1, 1, 10
    m = len(p_train)
    q = np.ones(len(p_train)) / len(p_train)
    q = p_train
    K = 30000
    for k in range(K):
        idx = np.random.choice(np.arange(m), p=q)
        stoc_losses = np.zeros_like(losses)
        stoc_losses[idx] = losses[idx]
        step_size = 1
        q = compute_primal_dual_q(q, stoc_losses, rho, step_size)

        if k % 2000 == 0:
            print(q)
            
        #loss = (new_q * losses).sum()
    print("primal dual after {} steps:".format(K))
    print(" ".join(["{:.6f}".format(ii) for ii in q]))










