import numpy as np
from numpy import outer, trace, dot, vdot, pi, log2, exp, sin, cos, sqrt, diag, linspace, arange, array, inf, zeros, eye, mean, std, concatenate, kron, sign, ceil, log, unique
from numpy.random import uniform, normal, randint, choice
from numpy.linalg import svd, norm, eigh, inv, pinv, lstsq
from scipy.linalg import expm, sqrtm, pinvh
from scipy.optimize import minimize
from scipy.stats import unitary_group, gaussian_kde
from scipy import sparse
from functools import reduce, partial
from itertools import product
import qutip as qp
from multiprocessing import Pool
from time import time

P0 = np.array([[1., 0.],
               [0., 0.]])
P1 = np.array([[0., 0.],
               [0., 1.]])
X = np.array([[0.,1.],
              [1.,0.]])
Y = np.array([[0.,-1.j],
              [1.j, 0.]])
Z = np.array([[1., 0.],
              [0.,-1.]])
I = np.array([[1.,0.],
              [0.,1.]])

I_sp = sparse.csr_array(I)
X_sp = sparse.csr_array(X)
Y_sp = sparse.csr_array(Y)
Z_sp = sparse.csr_array(Z)


### Auxiliary functions ###

def kron_A_I(A, N): # fast kron(A, eye(N))
    m, n = A.shape
    out = zeros((m, N, n, N), dtype=A.dtype)
    r = arange(N)
    out[:, r, :, r] = A
    out.shape = (m*N, n*N)
    return out
    
def kron_U_A(A, N): # fast kron(eye(N), A)
    m, n = A.shape
    out = zeros((N, m, N, n), dtype=A.dtype)
    r = np.arange(N)
    out[r, :, r, :] = A
    out.shape = (m*N, n*N)
    return out

def kron_A_I_diag(A, N): # same, but "diagonal"
    m = len(A)
    out = zeros((N, m), dtype=A.dtype)
    out[arange(N)] = A
    out = out.T.reshape(m*N)
    return out

def kron_I_A_diag(A, N):  # same, but "diagonal"
    m = len(A)
    out = zeros((N, m), dtype=A.dtype)
    out[arange(N)] = A
    out = out.reshape(m*N)
    return out

def fkron(A, B):
    """ A faster kronecker product taken from https://stackoverflow.com/a/56067827 """
    s = len(A)*len(B)
    return (A[:, None, :, None]*B[None, :, None, :]).reshape(s, s)

def fkron_vec(A, B):
    """ A faster (?) kronecker product for vectors. Taken from https://stackoverflow.com/a/56067827 """
    s = len(A)*len(B)
    return (A[:, None]*B[None, :]).reshape(s)

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



### Main functions ###

def compute_L(dms, ansatz, n_copies=1):
    """ Computes only the L matrix with expected values. For density matrices. """ 
    n_train = len(dms)
    n_pars = len(ansatz)
    L = zeros((n_train, n_pars))
    time_start = time()
    for i in range(n_train):
        print("Computing L: i=%d | time passed: %.2f s" %(i, time() - time_start), end="\r")
        dm_c = reduce(fkron, [dms[i]]*n_copies)
        for j in range(n_pars):
            op_loc = dm_c@ansatz[j]
            L[i][j] = trace(op_loc).real
    time_finish = time() - time_start
    print("Computing L: finished in %.2f s" %time_finish, end="\r")
    return L

def compute_LS(dms, ansatz, n_copies=1):
    """ Computes the L and S matrices with expected values. For density matrices. """ 
    n_train = len(dms)
    n_pars = len(ansatz)
    L = zeros((n_train, n_pars))
    S = zeros((n_train, n_pars, n_pars))
    time_start = time()
    for i in range(n_train):
        print("Computing L and S: i=%d | time passed: %.2f s" %(i, time() - time_start), end="\r")
        dm_c = reduce(fkron, [dms[i]]*n_copies)
        for j in range(n_pars):
            op_loc = dm_c@ansatz[j]
            L[i][j] = trace(op_loc).real
            for k in range(n_pars):
                S[i][j][k] = trace(op_loc@ansatz[k]).real
    time_finish = time() - time_start
    print("Computing L and S: finished in %.2f s" %time_finish, end="\r")
    return L, S

def compute_L_pure(svs, ansatz, n_copies=1):
    """ Computes only the L matrix with expected values. For pure states. """ 
    n_train = len(svs)
    n_pars = len(ansatz)
    L = zeros((n_train, n_pars))
    time_start = time()
    for i in range(n_train):
        print("Computing L: i=%d | time passed: %.2f s" %(i, time() - time_start), end="\r")
        sv_c = reduce(fkron_vec, [svs[i]]*n_copies)
        for j in range(n_pars):
            op_sv_c = sv_c.conj()@ansatz[j]
            L[i][j] = (op_sv_c@sv_c).real
    time_finish = time() - time_start
    print("Computing L: finished in %.2f s" %time_finish, end="\r")
    return L

def compute_LS_pure(svs, ansatz, n_copies=1):
    """ Computes the L and S matrices with expected values. For density matrices. """ 
    n_train = len(svs)
    n_pars = len(ansatz)
    L = zeros((n_train, n_pars))
    S = zeros((n_train, n_pars, n_pars))
    time_start = time()
    for i in range(n_train):
        print("Computing L and S: i=%d | time passed: %.2f s" %(i, time() - time_start), end="\r")
        sv_c = reduce(fkron_vec, [svs[i]]*n_copies)
        for j in range(n_pars):
            op_sv_c = sv_c.conj()@ansatz[j]
            L[i][j] = (op_sv_c@sv_c).real
            for k in range(n_pars):
                S[i][j][k] = (op_sv_c@ansatz[k]@sv_c).real
    time_finish = time() - time_start
    print("Computing L and S: finished in %.2f s" %time_finish, end="\r")
    return L, S
    

def get_pars(labels, L, S=None, w_ls=1., w_var=1e-4, method="pinv"):
    k = w_var/w_ls
    B = L.T@L
    A = (1 - k)*B
    if S is not None:
        C = np.sum(S + np.transpose(S, axes=(0, 2, 1)), axis=0)
        A = A + k/2*C
    b = L.T@labels
    if method == "pinv":
        pars = pinv(A, hermitian=True)@b
    elif method == "lstsq":
        pars = lstsq(A, b, rcond=None)[0]
    return pars


def train_obs_L(labels, ansatz, L, method="Newton-CG", x0=None, options={}):
    """ 
        Finds the coefficients of the linear combination of observables in ansatz. 
        Uses only the L matrix (no variance regularization).
    """
    
    n_pars = len(ansatz)
    n_train = len(labels)

    ### precalcs ###
    f_1 = labels@labels
    f_2 = -2*labels@L
    f_3 = L.T@L
    g_1 = f_2.T
    g_2 = 2*L.T

    fval_cont = []

    def fun(x):
        f = f_1 + f_2@x + x@f_3@x
        fval_cont.append(f)
        return f
        
    def jac(x):
        return g_1 + g_2@x
        
    def hess(x):
        return g_2
    
    time_loc = time()
    
    def callback(x):
        print("\t\t\tIteration: %d | Cost: %.8f | Time passed: %d s" %(len(fval_cont), fval_cont[-1], time() - time_loc), end="\r")
        return None
    
    if method in ["Nelder-Mead", "L-BFGS-B", "SLSQP", "TNC", "Powell", "COBYLA", "COBYQA"]:
        bounds = [(-100, 100)]*n_pars # some values 
    else:
        bounds = None
    if x0 is None:
        x0 = normal(0, 1e-5, n_pars)
    
    optimization_result = minimize(fun=fun, jac=jac, hess=hess, x0=x0, bounds=bounds, method=method, callback=callback, options=options)
    
    print("\n", optimization_result.message)

    return optimization_result.x


def train_obs_LS(labels, ansatz, L, S, method="Newton-CG", w_ls=1e0, w_var=1e-4, x0=None, options={}):
    
    n_pars = len(ansatz)
    n_train = len(labels)

    ### precalcs ###
    S_sum = np.sum(S, axis=0)
    f_1 = w_ls*labels@labels
    f_2 = -2*w_ls*labels@L
    f_3 = w_var*S_sum + (w_ls - w_var)*L.T@L
    g_1 = -2*w_ls*L.T@labels
    g_2 = w_var*(S_sum + S_sum.T) + 2*(w_ls - w_var)*L.T@L
    
    fval_cont = []

    def fun(x):
        f = f_1 + f_2@x + x@f_3@x
        fval_cont.append(f)
        return f

    def jac(x):
        return g_1 + g_2@x

    def hess(x):
        return g_2
        
    time_loc = time()
    
    def callback(x):
        print("\t\t\tIteration: %d | Cost: %.8f | Time passed: %d s" %(len(fval_cont), fval_cont[-1], time() - time_loc), end="\r")
        return None
    
    if method in ["Nelder-Mead", "L-BFGS-B", "SLSQP", "TNC", "Powell", "COBYLA", "COBYQA"]:
        bounds = [(-100, 100)]*n_pars # some arbitrary values 
    else:
        bounds = None
    if x0 is None:
        x0 = normal(0, 1e-2, n_pars)
    
    optimization_result = minimize(fun=fun, x0=x0, jac=jac, hess=hess, bounds=bounds, method=method, callback=callback, options=options) # "maxiter": int(1e10)
    
    print("\n", optimization_result.message)

    return optimization_result.x



def aux_info(dms, labels, pars, ansatz, n_copies=1):

    n_tot = int(log2(len(dms[0])))*n_copies
    n_labels = len(labels)

    H = zeros([2**n_tot, 2**n_tot], dtype=complex)
    for par, op in zip(pars, ansatz):
        H += par*op
    H_sq = H@H

    expecs = zeros(n_labels, dtype=float)
    disps = zeros(n_labels, dtype=float)
    for i in range(n_labels):
        print("i = %d" %i, end="\r")
        dm_c = reduce(fkron, [dms[i]]*n_copies)
        expecs[i] = trace(H@dm_c).real
        disps[i] = trace(H_sq@dm_c).real
    disps = disps - expecs**2
    
    return expecs, disps, H
    

def aux_info_pure(svs, labels, pars, ansatz, n_copies=1):

    n_tot = int(log2(len(svs[0])))*n_copies
    n_labels = len(labels)

    H = zeros([2**n_tot, 2**n_tot], dtype=complex)
    for par, op in zip(pars, ansatz):
        H += par*op
    H_sq = H@H

    expecs = zeros(n_labels, dtype=float)
    disps = zeros(n_labels, dtype=float)
    for i in range(n_labels):
        print("i = %d" %i, end="\r")
        sv_c = reduce(fkron_vec, [svs[i]]*n_copies)
        H_sv_c = H@sv_c
        expecs[i] = (sv_c.conj()@H_sv_c).real
        disps[i] = (sv_c.conj()@H@H_sv_c).real
    disps = disps - expecs**2
    
    return expecs, disps, H


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
    labels = zeros(n, dtype=float)
    if mixed == True:
        states = zeros([n, d, d], dtype=complex)
    else:
        states = zeros([n, d], dtype=complex)
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
                states[count] = state
                labels[count] = ent
                count += 1
            
    return array(states), array(labels)



# Fisher informations #

def cfi_state(state_func, p, state_args, povm, n_copies=1, n_ext=0, dp=1e-5):
    """ Computes classical Fisher information. """
    dm_ext = diag([1] + [0]*(2**n_ext - 1)) # extension
    dm_n = reduce(kron, [state_func(p, *state_args)]*n_copies + [dm_ext])
    dm_n_p = reduce(kron, [state_func(p + dp, *state_args)]*n_copies + [dm_ext])
    dm_n_m = reduce(kron, [state_func(p - dp, *state_args)]*n_copies + [dm_ext])
    fi = 0
    for op in povm:
        prob = trace(dot(dm_n, op)).real
        if prob > 0:
            prob_p = trace(dot(dm_n_p, op)).real
            prob_m = trace(dot(dm_n_m, op)).real
            der = (prob_p - prob_m)/(2*dp)
            fi += der**2/prob
    return fi

def qfi_state(state_func, p, state_args=[], n_copies=1, n_ext=0, dp=1e-5, mode="right"):
    """ Computes quantum Fisher information via the fidelity. """
    dm_ext = diag([1] + [0]*(2**n_ext - 1)) # extension
    if mode == "left":
        dm_n_l = reduce(kron, [state_func(p-dp, *state_args)]*n_copies + [dm_ext])
        dm_n_r = reduce(kron, [state_func(p, *state_args)]*n_copies + [dm_ext])
        denom = 1
    if mode == "central":
        dm_n_l = reduce(kron, [state_func(p-dp, *state_args)]*n_copies + [dm_ext])
        dm_n_r = reduce(kron, [state_func(p+dp, *state_args)]*n_copies + [dm_ext])
        denom = 4
    if mode == "right":
        dm_n_l = reduce(kron, [state_func(p, *state_args)]*n_copies + [dm_ext])
        dm_n_r = reduce(kron, [state_func(p+dp, *state_args)]*n_copies + [dm_ext])
        denom = 1
    fi = 8*(1 - sqrt(fidelity(dm_n_l, dm_n_r)))/dp**2/denom
    return fi

def sup_qfi_state(state_func, p, state_args=[], n_copies=1, n_ext=0, dp=1e-5, mode="right"):
    """ Computes quantum Fisher information via the fidelity. """
    dm_ext = diag([1] + [0]*(2**n_ext - 1)) # extension
    if mode == "left":
        dm_n_r = reduce(kron, [state_func(p, *state_args)]*n_copies + [dm_ext])
        denom = 1
    if mode == "central":
        dm_n_l = reduce(kron, [state_func(p-dp, *state_args)]*n_copies + [dm_ext])
        dm_n_r = reduce(kron, [state_func(p+dp, *state_args)]*n_copies + [dm_ext])
        denom = 4
    if mode == "right":
        dm_n_l = reduce(kron, [state_func(p, *state_args)]*n_copies + [dm_ext])
        dm_n_r = reduce(kron, [state_func(p+dp, *state_args)]*n_copies + [dm_ext])
        denom = 1
    fi = 8*(1 - sqrt(sup_fidelity(dm_n_l, dm_n_r)))/dp**2/denom
    return fi


def sld(state_func, p, state_args=[], n_copies=1, n_ext=0, dp=1e-5, return_fi=False):
    """
        Numerically finds the SLD operator L.
        Optionally returns the classical and quantum Fisher informations.
    """
    
    n_inp = int(log2(len(dm_ini)))*n_copies
    n_tot = n_inp + n_ext
    d = 2**n_tot
        
    dm_ext = diag([1] + [0]*(2**(n_ext) - 1))
    dm_n = reduce(kron, [state_func(p, *state_args)]*n_copies + [dm_ext])
    dm_n_p = reduce(kron, [state_func(p + dp, *state_args)]*n_copies + [dm_ext])
    dm_n_m = reduce(kron, [state_func(p - dp, *state_args)]*n_copies + [dm_ext])
    dm_n_der = (dm_n_p - dm_n_m)/(2*dp)
    
    evals, evecs = eigh(dm_n)
    evecs = evecs.T
    
    L = zeros([d, d], dtype=complex)
    for i in range(d):
        for j in range(d):
            denom = evals[i] + evals[j]
            if denom > 1e-5:
                numer = evecs[i].conj().T@dm_n_der@evecs[j]
                oper = outer(evecs[i], evecs[j].conj().T)
                L += 2*numer/denom*oper
    # print("Conformity with the definition of SLD:", norm( (L@dm_n + dm_n@L)/2 - dm_n_der ))
    
    if return_fi == True:
        evecs_L = eigh(L)[1].T
        projs_L = [outer(vec, vec.conj().T) for vec in evecs_L]
        CFI = cfi(channel_func, dm_ini, channel_par, channel_args, projs_L, n_copies=n_copies, n_ext=n_ext, dp=dp)
        QFI = trace(L@L@dm_n).real
        return L, CFI, QFI
    else:
        return L
        





### Hamiltonians ###

def ising_ham(n_qubits, h, J=1, bc="closed"):
    d = 2**n_qubits
    Hx = zeros((d, d), dtype=complex)
    for q in range(n_qubits):
        X_op = [eye(2**q)] + [X] + [eye(2**(n_qubits - q - 1))]
        Hx = Hx + reduce(kron, X_op)
    Hzz = zeros((d, d), dtype=complex)
    for q in range(n_qubits - 1):
        Hzz = Hzz + reduce(kron, [eye(2**q)] + [Z, Z] + [eye(2**(n_qubits - q - 2))])
    if bc == "closed" and n_qubits > 2:
        Hzz = Hzz + reduce(kron, [Z] + [eye(2**(n_qubits - 2))] + [Z])
    if n_qubits == 1: # lame
        Hzz = 1*Z
    return -J*(Hzz + h*Hx)

def ising_ham_sparse(n, h, J=1, bc="closed"):
    d = 2**n
    Hx = sparse.csr_array(zeros([d, d], dtype=complex))
    for q in range(n):
        Hx = Hx + reduce(sparse.kron, [sparse.eye(2**q), X_sp, sparse.eye(2**(n - q - 1))])
    Hzz = sparse.csr_array(zeros([d, d], dtype=complex))
    for q in range(n - 1):
        Hzz = Hzz + reduce(sparse.kron, [sparse.eye(2**q), Z_sp, Z_sp, sparse.eye(2**(n - q - 2))])
    if bc == "closed" and n > 2:
        Hzz = Hzz + reduce(sparse.kron, [Z_sp, sparse.eye(2**(n - 2)), Z_sp])
    if n == 1: # lame
        Hzz = 1*Z_sp
    return -J*(Hzz + h*Hx)
    

def schwinger_ham(N, m, w=1, g=1, e0=0):
    d = 2**N
    sp = (X + 1j*Y)/2
    sm = (X - 1j*Y)/2
    term_1 = zeros([d, d], dtype=complex)
    for j in range(N - 1):
        op = reduce(kron, [I]*j + [sp, sm] + [I]*(N - j - 2)) # optimizable
        term_1 += op + op.conj().T
    term_2 = zeros([d, d], dtype=complex)
    for j in range(N):
        op = reduce(kron, [I]*j + [Z] + [I]*(N - j - 1))
        term_2 += (-1)**(j + 1)*op
    term_3 = zeros([d, d], dtype=complex)
    for j in range(N):
        L_j = zeros([d, d], dtype=complex)
        for l in range(j + 1):
            op = Z + (-1)**(l + 1)*I
            op = reduce(kron, [I]*l + [op] + [I]*(N - l - 1))
            L_j += op
        L_j = e0 - L_j/2
        term_3 += L_j@L_j
    return w*term_1 + m/2*term_2 + g*term_3

def schwinger_ham_sparse(N, m, w=1, g=1, e0=0):
    d = 2**N
    sp = (X_sp + 1j*Y_sp)/2
    sm = (X_sp - 1j*Y_sp)/2
    term_1 = sparse.csr_array(zeros([d, d], dtype=complex))
    for j in range(N - 1):
        op = reduce(sparse.kron, [sparse.eye(2**j)] + [sp, sm] + [sparse.eye(2**(N - j - 2))]) # optimizable
        term_1 += op + op.conj().T
    term_2 = sparse.csr_array(zeros([d, d], dtype=complex))
    for j in range(N):
        op = reduce(sparse.kron, [sparse.eye(2**j)] + [Z_sp] + [sparse.eye(2**(N - j - 1))])
        term_2 += (-1)**(j + 1)*op
    term_3 = sparse.csr_array(zeros([d, d], dtype=complex))
    for j in range(N):
        L_j = sparse.csr_array(zeros([d, d], dtype=complex))
        for l in range(j + 1):
            op = Z + (-1)**(l + 1)*I
            op = reduce(sparse.kron, [sparse.eye(2**l)] + [op] + [sparse.eye(2**(N - l - 1))])
            L_j += op
        L_j = e0 - L_j/2
        term_3 += L_j@L_j
    return w*term_1 + m/2*term_2 + g*term_3


def sk_ham(N, h=0, J=None):
    d = 2**N
    Hzz = zeros([d, d])
    for i in range(N):
        for j in range(i + 1, N):
            op = [I]*N
            op[i] = [Z]
            op[j] = [Z]
            Hzz = Hzz + normal(0, 1)*reduce(kron, op)
    Hx = zeros([d, d])
    for q in range(N):
        Hx = Hx + reduce(kron, [eye(2**q), X, eye(2**(N - q - 1))])
    return Hzz + h*Hx

def sk_ham_sparse(N, h=0, J=None):
    d = 2**N
    Hzz = sparse.csr_array(zeros([d, d]))
    for i in range(N):
        for j in range(i + 1, N):
            op = sparse.kron(sparse.eye(2**i), Z_sp)
            op = sparse.kron(op, sparse.eye(2**(j - i - 1)))
            op = sparse.kron(op, Z_sp)
            op = sparse.kron(op, sparse.eye(2**(N - j - 1)))
            Hzz = Hzz + normal(0, 1)*op
    Hx = sparse.csr_array(zeros([d, d]))
    for q in range(N):
        Hx = Hx + reduce(sparse.kron, [sparse.eye(2**q), X_sp, sparse.eye(2**(N - q - 1))])
    return Hzz + h*Hx

### Swaps ###

def swap_matrix(n_qudits, q1, q2, dim=2):
    p1, p2 = sorted([q1, q2])
    SW = zeros([dim**n_qudits, dim**n_qudits])
    SW_ph = zeros([dim, dim])
    for i in range(dim):
        for j in range(dim):
            SW_ph[i, j] = 1
            SW_ij = array(SW_ph)
            SW_ph[i, j] = 0
            SW_ph[j, i] = 1
            SW_ji = array(SW_ph)
            SW_ph[j, i] = 0
            op = reduce(kron, [eye(dim**p1), SW_ij, eye(dim**(p2 - p1 - 1)), SW_ji, eye(dim**(n_qudits - p2 - 1))])
            SW += op    
    return SW

def swap_matrix_sparse(n_qudits, q1, q2, dim=2):
    p1, p2 = sorted([q1, q2])
    SW = sparse.csr_array(zeros([dim**n_qudits, dim**n_qudits]))
    SW_ph = zeros([dim, dim])
    for i in range(dim):
        for j in range(dim):
            SW_ph[i, j] = 1
            SW_ij = array(SW_ph)
            SW_ph[i, j] = 0
            SW_ph[j, i] = 1
            SW_ji = array(SW_ph)
            SW_ph[j, i] = 0
            op = reduce(sparse.kron, [sparse.eye(dim**p1), SW_ij, sparse.eye(dim**(p2 - p1 - 1)), SW_ji, sparse.eye(dim**(n_qudits - p2 - 1))])
            SW += op    
    return SW


    
# measure #    

def measure_z_counts(dm, n_shots):
    probs = dm.diagonal().real
    d = len(probs)
    measurements = choice(arange(d), size=n_shots, p=probs)
    measurements = unique(measurements, return_counts=True)
    counts = zeros(d)
    counts[measurements[0]] = measurements[1]
    return counts

def measure_povm_counts(dm, n_shots, povm):
    probs = [trace(dm@el).real for el in povm]
    d = len(probs)
    measurements = choice(arange(d), size=n_shots, p=probs)
    measurements = unique(measurements, return_counts=True)
    counts = zeros(d)
    counts[measurements[0]] = measurements[1]
    return counts



### channels ###

def hw_channel(dm, p):
    """ Holevo-Werner channel """
    d = len(dm)
    return ((d - p)*eye(d) + (d*p - 1)*dm.T)/(d**2 - 1)
    
def depolarizing_channel(dm, p):
    d = len(dm)
    return (1 - p)*dm + p/d*eye(d)

def generalized_amplitude_damping_channel(dm, g, N, target_qubit):
    
    n_qubits = int(log2(len(dm)))
    dl = 2**target_qubit
    dr = 2**(n_qubits - target_qubit - 1)
    
    K1 = array([[1,           0],
                [0, sqrt(1 - g)]])*sqrt(1 - N)
    K1 = reduce(kron, [eye(dl), K1, eye(dr)]) # inefficient
    
    K2 = array([[0, sqrt(g*(1 - N))],
                [0,               0]])
    K2 = reduce(kron, [eye(dl), K2, eye(dr)])
    
    K3 = array([[sqrt(1 - g), 0],
                [          0, 1]])*sqrt(N)
    K3 = reduce(kron, [eye(dl), K3, eye(dr)])
            
    K4 = array([[0,         0],
                [sqrt(g*N), 0]])
    K4 = reduce(kron, [eye(dl), K4, eye(dr)])
    
    dm1 = reduce(dot, [K1, dm, K1.conj().T])
    dm2 = reduce(dot, [K2, dm, K2.conj().T])
    dm3 = reduce(dot, [K3, dm, K3.conj().T])
    dm4 = reduce(dot, [K4, dm, K4.conj().T])
        
    return dm1 + dm2 + dm3 + dm4


def another_generalized_amplitude_damping_channel(dm, g, N, target_qubit):
    """ Adapted from https://journals.aps.org/pra/abstract/10.1103/PhysRevA.70.012317 """
    
    n_qubits = int(log2(len(dm)))
    dl = 2**target_qubit
    dr = 2**(n_qubits - target_qubit - 1)
    
    K1 = array([[1,           0],
                [0,     sqrt(g)]])*sqrt(N)
    K1 = reduce(kron, [eye(dl), K1, eye(dr)]) # inefficient
    
    K2 = array([[0, sqrt((1 - g))],
                [0,             0]])*sqrt(N)
    K2 = reduce(kron, [eye(dl), K2, eye(dr)])
    
    K3 = array([[sqrt(g), 0],
                [      0, 1]])*sqrt(1 - N)
    K3 = reduce(kron, [eye(dl), K3, eye(dr)])
            
    K4 = array([[0,           0],
                [sqrt(1 - g), 0]])*sqrt(1 - N)
    K4 = reduce(kron, [eye(dl), K4, eye(dr)])
    
    dm1 = reduce(dot, [K1, dm, K1.conj().T])
    dm2 = reduce(dot, [K2, dm, K2.conj().T])
    dm3 = reduce(dot, [K3, dm, K3.conj().T])
    dm4 = reduce(dot, [K4, dm, K4.conj().T])
        
    return dm1 + dm2 + dm3 + dm4

    
def X_rotations(dm, p):
    n_qubits = int(log2(len(dm)))
    op = reduce(kron, [X]*n_qubits)
    U = expm(-1j*p*op)
    return reduce(dot, [U, dm, U.conj().T])

def Y_rotations(dm, p):
    n_qubits = int(log2(len(dm)))
    op = reduce(kron, [Y]*n_qubits)
    U = expm(-1j*p*op)
    return reduce(dot, [U, dm, U.conj().T])

def Z_rotations(dm, p):
    n_qubits = int(log2(len(dm)))
    op = reduce(kron, [Z]*n_qubits)
    U = expm(-1j*p*op)
    return reduce(dot, [U, dm, U.conj().T])

def z_rot(dm, p, target_qubit):
    """
        z-rotation of the specified qubit.
        Mind the division by two!
    """
    n_qubits = int(log2(len(dm)))
    dl = 2**target_qubit
    dr = 2**(n_qubits - target_qubit - 1)
    U = reduce(kron, [eye(dl), expm(-1j*p/2*Z), eye(dr)]) # inefficient
    return U@dm@U.conj().T


def random_channel(dm, p, pars, p_index, pauli_basis=None):
    """
        Random single-parametrized channel.
        Attaches to a given n-qubit state dm a 2n-qubit pure state,
        applies to the joint state a unitary with (4^(3n) - 1) parameters pars, one of which is the parameter p in question with the index p_index.        
    """
    d = len(dm)
    n_inp = int(log2(d))
    n_ext = 2*n_inp
    n_tot = n_inp + n_ext
    pars_conc = concatenate([pars[:p_index], [p], + pars[p_index:]])
    V = su2n(n_tot, pars_conc, pauli_basis=pauli_basis)
    dm_ext = diag([1] + [0]*(2**n_ext - 1))
    dm_n = V@kron(dm, dm_ext)@V.conj().T
    dm_n = trace(dm_n.reshape(2**n_inp, 2**n_ext, 2**n_inp, 2**n_ext), axis1=1, axis2=3) # partial trace with respect to the extension
    return dm_n