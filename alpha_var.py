import numpy as np
import math
from scipy.stats import norm

def sample_rewards(mdp, D, c):
    # TODO: Sample reward functions
    rfs = []
    return rfs

# mdp: (transition, gamma, s_0)

def get_v(mdp, r, pi):
    # TODO: estimate value function for given policy
    v = 0
    return v

def get_v_expert(mdp, r, D):
    v = 0
    for d in D:
        vi = 0
        gamma = 1
        for s,a in d:
            vi += gamma*r(s,a)
            gamma *= mdp[1]
        v+=vi/len(D)
    return v

def alpha_var(mdp, pi_eval, D, c, alpha, delta, rewards = None):
    if not rewards:
        rewards = sample_rewards(mdp, D, c)
    
    Z = []

    for reward in rewards:
        Z.append(get_v_expert(mdp, reward, D) - get_v(mdp, reward, pi_eval))
    
    Z = sorted(Z)
    N = len(Z)
    k = math.ceil(N*alpha + norm.ppf(1-delta)*math.sqrt(N*alpha*(1-alpha)) - 0.5)

    return Z[k-1]