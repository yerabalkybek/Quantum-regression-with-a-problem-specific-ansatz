import numpy as np
from numpy import outer, trace, dot, vdot, pi, log2, exp, sin, cos, sqrt, sign, diag, linspace, arange, array, inf, zeros, eye, arccos, arcsin, arctan, mean, std, concatenate, kron, sign, ceil, log, unique
from numpy.random import uniform, normal, randint, choice
from scipy.linalg import eig, eigh, norm, expm, sqrtm
from numpy.linalg import svd, norm, inv, pinv
from scipy.stats import sem
from scipy.optimize import minimize
from functools import reduce, partial
from itertools import product
import qutip as qp
from multiprocessing import Pool
from time import time

P0 = np.array([[1., 0.],
               [0., 0.]]) # |0><0|
P1 = np.array([[0., 0.],
               [0., 1.]]) # |1><1|
X = np.array([[0.,1.],
              [1.,0.]]) # X Pauli matrix
Y = np.array([[0.,-1.j],
              [1.j, 0.]]) # Y Pauli matrix
Z = np.array([[1., 0.],
              [0.,-1.]]) # Z Pauli matrix
I = np.array([[1.,0.],
              [0.,1.]]) # 2x2 identity matrix

# some functions # 

def kron_A_N(A, N): # fast kron(A, eye(N))
    m,n = A.shape
    out = zeros((m, N, n, N), dtype=A.dtype)
    r = arange(N)
    out[:, r, :, r] = A
    out.shape = (m*N, n*N)
    return out
    
def kron_N_A(A, N): # fast kron(eye(N), A)
    m,n = A.shape
    out = zeros((N, m, N, n), dtype=A.dtype)
    r = np.arange(N)
    out[r, :, r, :] = A
    out.shape = (m*N, n*N)
    return out

def kron_A_I_diag(A, N):
    m = len(A)
    out = zeros((N, m), dtype=A.dtype)
    out[arange(N)] = A
    out = out.T.reshape(m*N)
    return out

def kron_I_A_diag(A, N):
    m = len(A)
    out = zeros((N, m), dtype=A.dtype)
    out[arange(N)] = A
    out = out.reshape(m*N)
    return out

def trace_distance(A, B):
    sub = A - B
    return trace(sqrtm(dot(sub.conj().T, sub))).real/2

def fidelity(A, B):
    res = reduce(dot, [sqrtm(A), B, sqrtm(A)])
    res = sqrtm(res)
    return trace(res).real**2

def sup_fidelity(A, B):
    """ An upper bound for the usual fidelity. """
    t1 = trace(A@B).real
    t2 = max(0, 1 - trace(A@A).real)
    t3 = max(0, 1 - trace(B@B).real)
    return t1 + sqrt(t2)*sqrt(t3)

def partial_trace(dm, m=None, n=None, subsystem=0):
    """ Simple and fast, but cuts only in halves. """
    if (m is None) or (n is None): # cut in equal halves
        N = log2(len(dm))
        m = int(N / 2)
        n = int(N - m)
        m = 2**m
        n = 2**n
    if subsystem == 0:
        return trace(dm.reshape((m, n, m, n)), axis1=0, axis2=2)
    elif subsystem == 1:
        return trace(dm.reshape((m, n, m, n)), axis1=1, axis2=3)

def concurrence_pure(dm):
    dm_red = partial_trace(dm)
    return sqrt(2*(1 - trace(dm_red@dm_red).real))

def concurrence(dm):
    YY = kron(Y, Y)
    dm_t = YY@dm.conj()@YY
    R = dm_t@dm
    lambdas = [l if l > 0 else 0 for l in np.sort(eig(R)[0].real)]
    c = sqrt(lambdas[3]) - sqrt(lambdas[2]) - sqrt(lambdas[1]) - sqrt(lambdas[0])
    return max(0, c)

def two_subsys_negativity(dm):
    def partial_transpose(A, n, m):
        A_c = array(A)
        Bt = A[:n, m:].copy()
        Ct = A[n:, :m].copy()
        A_c[:n, m:] = Ct
        A_c[n:, :m] = Bt
        return A_c
    dm_ptrans = partial_transpose(dm, int(len(dm)/2), int(len(dm)/2))
    lambda_min = eigh(dm_ptrans)[0][0]
    return 2*abs(min(0, lambda_min))

def gen_even_ent_data(n, n_inp=2, mixed=True, marks="neg", n_chunks=100, eps=0):
    """ Generates a data set of states with evenly distributed entanglements. """
    
    d = 2**n_inp
    
    if marks == "neg":
        ent_measure_func = two_subsys_negativity
    elif marks == "con":
        ent_measure_func = concurrence 
    
    ent_count_max = int(ceil(n/n_chunks))
    ent_line = linspace(0, 1, n_chunks + 1)[1:]
    ent_counts = [0]*n_chunks
    count = 0
    states = []
    labels = []
    while count < n:
        print("%d" %count, end="\r")
        if mixed == True:
            state = (qp.rand_dm(d, distribution='ginibre', rank=randint(1, d + 1))).full() # lame, but works faster for mixed states
            ent = ent_measure_func(state)
        else:
            state = (qp.rand_ket(d)).full().reshape(-1)
            ent = ent_measure_func(outer(state, state.conj().T))
        if ent >= eps:
            ent_diffs = ent_line - ent
            ind = np.abs(ent_diffs).argmin()
            if sign(ent_diffs[ind]) == -1:
                ind += 1
            if ent_counts[ind] < ent_count_max:
                ent_counts[ind] += 1        
                count += 1
                states.append(state)
                labels.append(ent)
            
    return array(states), array(labels)


# quantum state generators # 

def rand_sv(n_qubits):
    """ Generates a random pure state as a vector. """
    d = 2**n_qubits
    sv = uniform(-1, 1, d) + 1j*uniform(-1, 1, d)
    return sv/norm(sv)

def rand_dm(n_qubits):
    """ Generates a random mixed state as a full-rank density matrix. """
    d = 2**n_qubits
    H = uniform(-1, 1, [d, d]) + 1j*uniform(-1, 1, [d, d])
    dm = H@H.conj().T
    dm = dm/trace(dm).real
    return dm


# Fisher informations #

def cfi(channel_func, dm, p, channel_args, povm, n_copies=1, n_ext=0, dp=1e-5):
    """ Computes classical Fisher information. Only for channels! """
    dm_n = reduce(kron, [channel_func(dm, p, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    dm_n_p = reduce(kron, [channel_func(dm, p+dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    dm_n_m = reduce(kron, [channel_func(dm, p-dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    fi = 0
    for op in povm:
        prob = trace(dot(dm_n, op)).real
        if prob > 0:
            prob_p = trace(dot(dm_n_p, op)).real
            prob_m = trace(dot(dm_n_m, op)).real
            der = (prob_p - prob_m)/(2*dp)
            fi += der**2/prob
    return fi

def qfi(channel_func, dm, p, channel_args, n_copies=1, n_ext=0, dp=1e-2):
    """ Computes quantum Fisher information. Only for channels! """
    dm_n = reduce(kron, [channel_func(dm, p, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    dm_n_p = reduce(kron, [channel_func(dm, p+dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    fi = 8*(1 - sqrt(fidelity(dm_n, dm_n_p))) / dp**2
    return fi

def qfi_central(channel_func, dm, p, channel_args, n_copies=1, n_ext=0, dp=1e-2):
    """ Computes quantum Fisher information via "central differences". Only for channels! """
    dm_n_p = reduce(kron, [channel_func(dm, p+dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    dm_n_m = reduce(kron, [channel_func(dm, p-dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    fi = 8*(1 - sqrt(fidelity(dm_n_m, dm_n_p))) / dp**2/4
    return fi

def sup_qfi(channel_func, dm, p, channel_args, n_copies=1, n_ext=0, dp=1e-5):
    """ Computes an upper bound (?) for quantum Fisher information. Only for channels! """
    dm_n = reduce(kron, [channel_func(dm, p, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    dm_n_p = reduce(kron, [channel_func(dm, p+dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    fi = 8*(1 - sqrt(sup_fidelity(dm_n, dm_n_p))) / dp**2
    return fi

import scipy.sparse as sp

from scipy import arcsin, sqrt, pi
from scipy.linalg import sqrtm
import numpy as np
import numpy.linalg as la
import math
UNITS = np.array([1, 1j])

def randnz(shape, norm=1 / np.sqrt(2), seed=None):
    # This function is intended for internal use.

    if seed is not None:
        np.random.seed(seed=seed)
    if norm == 'ginibre':
        norm = 1
    return np.sum(np.random.randn(*(shape + (2,))) * UNITS, axis=-1) * norm

def rand_unitary_haar(N=2, dims=None, seed=None):
    if dims is not None:
        _check_dims(dims, N, N)
    else:
        dims = [[N], [N]]
    Z = randnz((N, N), seed=seed)
    
    Q, R = la.qr(Z)

    Lambda = np.diag(R).copy()
    Lambda /= np.abs(Lambda)
    
    U = Q * Lambda
    return U

def swap_matrix(n_qudits, q1, q2, dim=2):
    """
       Can be rewritten as a sparse matrix.
       Optimizable! Use fast kron with the identity, kron_A_N and kron_N_A.
    """
    p1, p2 = sorted([q1, q2])
    SW = zeros([dim**n_qudits, dim**n_qudits])
    SW_ph = zeros([dim, dim])
    for i in range(dim):
        for j in range(dim):
            SW_ph[[i], [j]] = 1
            SW_ij = array(SW_ph)
            SW_ph[[i], [j]] = 0
            SW_ph[[j], [i]] = 1
            SW_ji = array(SW_ph)
            SW_ph[[j], [i]] = 0
            op = reduce(kron, [eye(dim**p1), SW_ij, eye(dim**(p2 - p1 - 1)), SW_ji, eye(dim**(n_qudits - p2 - 1))])
            SW += op    
    return SW

from numpy import log
def log3(a):
    return log(a)/log(3)

def train_obs(dms, labels, ansatz, Q=None, R=None, n_copies=1, method="BFGS", w_ls=1e0, w_var=1e-4, x0=None, options={}, save_data=False, file_name=None, fvals=[]):
########################
    print(len(dms))
    n_inp = int(log2(len(dms[0])))
    n_tot = n_inp*n_copies
    n_pars = len(ansatz)
    n_train = len(labels)
    print("n_tot", n_tot)
    print("n_copies", n_copies)
    print("n_inp", n_inp)
    # eats less memory than [reduce(kron, [dm]*n_copies) for dm in dms]
    ########################
    dms_cop = zeros([n_train, 2**n_tot, 2**n_tot], dtype=complex)
    for i in range(n_train):
        dms_cop[i] = reduce(kron, [dms[i]]*n_copies)
            
    if Q is None and R is None:
        time_start = time()
        Q = zeros((n_train, n_pars))
        R = zeros((n_train, n_pars, n_pars))
        for i in range(n_train):
            print("Computing Q and R: i=%d" %i, end="\r")
            for j in range(n_pars):
                op_loc = dms_cop[i]@ansatz[j]
                Q[i][j] = trace(op_loc).real
                for k in range(n_pars):
                    R[i][j][k] = trace(op_loc@ansatz[k]).real
        time_finish = time() - time_start
        print("Computing Q and R: completed in %.2f s" %time_finish, end="\r")
    
    fval_cont = [0]
    def fun(x):
        expecs = array([x@Q[i] for i in range(n_train)])
        disps = array([x@R[i]@x for i in range(n_train)]) - expecs**2
        f = w_ls*np.sum((array(expecs) - array(labels))**2)
        f += w_var*np.sum(array(disps))
        fval_cont[0] = f
        return f
    
    if save_data == True and file_name is None:
        file_name = "c=%d-m=%d=l=%d-w_ls=%f-w_var=%f-n_train=%d" %(n_copies, w_ls, w_var, n_train)
        
    time_loc = time()
    
    def callback(x):
        fvals.append(fval_cont[0])
        print("\t\t\tIteration: %d | Cost: %.8f | Time passed: %d s" %(len(fvals), fval_cont[0], time() - time_loc), end="\r")
        if save_data == True:
            np.save(file_name + "-pars", x)
            np.save(file_name + "-fvals", fvals)
        return None
    
    if method in ["Nelder-Mead", "L-BFGS-B", "SLSQP", "TNC", "Powell", "COBYLA", "COBYQA"]:
        bounds = [(-100, 100)]*n_pars # some arbitrary values 
    else:
        bounds = None
    if x0 is None:
        x0 = normal(0, 1e-5, n_pars)
    
    optimization_result = minimize(fun=fun, x0=x0, bounds=bounds, method=method, callback=callback, options=options) # "maxiter": int(1e10)

    return fvals, optimization_result

from itertools import permutations
from sympy.combinatorics.permutations import Permutation
from math import factorial

def aux_info_disp(dms, labels, pars, ansatz, n_copies=1):
 ###########
    n_inp = int(log2(len(dms[0])))
    n_tot = n_inp*n_copies
    n_pars = len(ansatz)
    n_train = len(labels)
                
    # slower than array([reduce(kron, [dm]*n_copies) for dm in dms]), but eats less memory
    dms_cop = zeros([n_train, 2**n_tot, 2**n_tot], dtype=complex)
    for i in range(n_train):
        dms_cop[i] = reduce(kron, [dms[i]]*n_copies)
    
    H = sum([par*op for par, op in zip(pars, ansatz)])
    H_sq = H@H
    
    expecs = array([trace(H@dm).real for dm in dms_cop])
    disps = array([trace(H_sq@dm).real for dm in dms_cop]) - expecs**2
        
    return expecs, disps, H

def aux_info(dms, labels, pars, ansatz, n_copies=1, iso=False):
 ###########
    n_inp = int(log2(len(dms[0])))
    n_tot = n_inp*n_copies
    n_pars = len(ansatz)
    n_train = len(labels)
    if iso:
        dm_channel=dm_iso_negSq_channel
    else:
        dm_channel=dm_pure_negSq_channel
    # slower than array([reduce(kron, [dm]*n_copies) for dm in dms]), but eats less memory

    dms_cop = zeros([n_train, 2**n_tot, 2**n_tot], dtype=complex)
        
    H = sum([par*op for par, op in zip(pars, ansatz)])
    H_sq = H@H
    
    dnegSq=1e-4
    d=2
    dms_negSq=[]
    for i in range(n_train):
        dms_cop[i] = reduce(kron, [dms[i]]*n_copies)
        negSq=labels[i]
        dm_p = reduce(np.kron, [dm_channel(np.eye(2), negSq+dnegSq)]*n_copies)
        dm_m = reduce(np.kron, [dm_channel(np.eye(2), negSq-dnegSq)]*n_copies)
        ddm=(dm_p-dm_m)/(2*dnegSq)
        dms_negSq.append(trace(H@ddm).real)
    
    expecs = array([trace(H@dm).real for dm in dms_cop])
    disps = array([trace(H_sq@dm).real for dm in dms_cop]) - expecs**2
    
    return expecs, disps/np.array(dms_negSq)**2, H


import numpy as np

def partial_trace(rho, dims, subsystem):
    """
    Compute the partial trace of a density matrix.
    
    Parameters:
    rho (np.ndarray): Density matrix of the composite system (shape (d, d)).
    dims (list): Dimensions of the subsystems [dA, dB].
    subsystem (int): Index of subsystem to trace over (0 for A, 1 for B).
    
    Returns:
    np.ndarray: Reduced density matrix after partial trace.
    """
    dA, dB = dims
    d = dA * dB
    
    # Validate input dimensions
    if rho.shape != (d, d):
        raise ValueError(f"rho must be {d}x{d} matrix")
    
    # Reshape to tensor with indices: i (A), j (B), k (A), l (B)
    tensor = rho.reshape((dA, dB, dA, dB))
    
    if subsystem == 0:  # Trace out subsystem A (index 0)
        # Contract indices: i (A) and k (A) set equal (trace), keep j (B) and l (B)
        return np.einsum('ijil->jl', tensor)
    elif subsystem == 1:  # Trace out subsystem B (index 1)
        # Contract indices: j (B) and l (B) set equal (trace), keep i (A) and k (A)
        return np.einsum('ijkj->ik', tensor)
    else:
        raise ValueError("subsystem must be 0 (A) or 1 (B)")


def negSq_dm(rho):
    return two_subsys_negativity(rho)**2
def dm_pure(negSq, d=2):
    neg=math.sqrt(negSq)
    U=np.kron(rand_unitary_haar(2),rand_unitary_haar(2))
    c1=math.sqrt(1+math.sqrt(1-negSq))
    c2=math.sqrt(1-math.sqrt(1-negSq))
    phi=U@(1/math.sqrt(2)*(c1*np.array([1,0,0,0])+c2*np.array([0,0,0,1])))
    dm=np.outer(phi,phi.conjugate())
    return dm
def dm_pure_negSq_channel(dm, negSq):
    new_dm=dm_pure(negSq)
    return new_dm

def dm_iso(negSq):
    Neg=math.sqrt(negSq)
    d=2
    return (2*Neg+1)/3*rhoBell+(1-Neg)/6*np.eye(d**2)

rhoBell=1/2*np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])
def dm_iso(negSq):
    Neg=math.sqrt(negSq)
    d=2
    return (2*Neg+1)/3*rhoBell+(1-Neg)/6*np.eye(d**2)

def dm_iso_negSq_channel(dm, negSq):
    new_dm=dm_iso(negSq)
    return new_dm