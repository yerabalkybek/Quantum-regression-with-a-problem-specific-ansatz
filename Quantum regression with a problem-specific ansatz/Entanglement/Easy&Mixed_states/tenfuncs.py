import numpy as np
from itertools import permutations
from qpe import *
from scipy.stats import unitary_group, gaussian_kde

import math
from itertools import permutations

import scipy.sparse as sps

import random
rhoBell=1/2*sps.csr_array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])

from time import time
from multiprocessing import Pool, cpu_count

import numpy.linalg as la
UNITS = sps.csr_array([1, 1j])

def randnz(shape, norm=1 / np.sqrt(2), seed=None):
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
    #SW = (SW + SW.conj().T)/2
    return SW
def sparse_swap_matrix(n_qudits, q1, q2, dim=2):
    p1, p2 = sorted([q1, q2])
    SW = sps.csr_array(zeros([dim**n_qudits, dim**n_qudits]))
    SW_ph = zeros([dim, dim])
    for i in range(dim):
        for j in range(dim):
            SW_ph[i, j] = 1
            SW_ij = array(SW_ph)
            SW_ph[i, j] = 0
            SW_ph[j, i] = 1
            SW_ji = array(SW_ph)
            SW_ph[j, i] = 0
            op = reduce(sps.kron, [sps.eye(dim**p1), SW_ij, sps.eye(dim**(p2 - p1 - 1)), SW_ji, sps.eye(dim**(n_qudits - p2 - 1))])
            SW += op    
    #SW = (SW + SW.conj().T)/2
    return SW

def generate_permutation_matrix(perm):
    """Generate a permutation matrix from a permutation tuple."""
    n = len(perm)
    P = np.zeros((n, n), dtype=int)
    for i, j in enumerate(perm):
        P[i, j] = 1
    return P

def permutation_to_swaps(perm):
    """Decompose a permutation into a sequence of transpositions (swaps)."""
    perm = list(perm)
    swaps = []
    for i in range(len(perm)):
        if perm[i] != i:
            # Find where i is currently located
            j = perm.index(i)
            # Swap the elements at positions i and j
            perm[i], perm[j] = perm[j], perm[i]
            swaps.append((i, j))
    return swaps

def swaps_to_matrix(swaps, n_qudits):
    Perm=np.eye(2**n_qudits)
    for i in range(len(swaps)):
        swap=swaps[i]
        Perm=swap_matrix(n_qudits, swap[0], swap[1], dim=2)@Perm
    return Perm
def swaps_AB_matrix(swaps, A, n_qudits, PermHerm=False):
    Perm=sps.csr_array(np.eye(2**(n_qudits)))
    for i in range(len(swaps)):
        swap=swaps[i]
        if A:
            Perm=sparse_swap_matrix(n_qudits, int(2*swap[0]), int(2*swap[1]), dim=2)@Perm
        else:
            Perm=sparse_swap_matrix(n_qudits, int(2*swap[0]+1), int(2*swap[1]+1), dim=2)@Perm
    if PermHerm:
        Perm = (Perm + Perm.conj().T)/2
    return Perm
def equal_matrices(A,B):
    diffs=A-B
    equal=True
    for diff in diffs:
        if diff.any():
            equal=False
            #print()#break
    return equal
def are_matrices_linearly_independent(matrices, printRanks=False):
    stacked = np.vstack([m.flatten() for m in matrices])
    rank = np.linalg.matrix_rank(stacked)
    if printRanks:
        print(rank,len(matrices)) 
    return rank == len(matrices)
def get_swaps_all(n_qudits=4, indep=True, herm=False, get_indep_swaps=False, hermitize=False, GS=None):
    
    if GS is None:
        G_s = [sps.eye(2**(2*n_qudits))]
        indep_perms=[np.eye(2**(2*n_qudits))]
        indep_swaps=[[[], []]]
    else:
        G_s = GS[0].copy()
        indep_perms=[G.toarray() for G in G_s]
        indep_swaps=GS[1].copy()
    
    all_perms = list(permutations(range(n_qudits)))
    all_swaps = []
    # Print each permutation matrix and its corresponding swaps
    for p1 in all_perms:
        swaps1 = permutation_to_swaps(p1)
        for p2 in all_perms:
            swaps2 = permutation_to_swaps(p2)
            sw=[swaps1, swaps2]
            if sw not in indep_swaps:
                all_swaps.append(sw)
    
    for i in range(len(all_swaps)):
        swaps1, swaps2=all_swaps[i]
        perm1 = swaps_AB_matrix(swaps1, True, n_qudits=2*n_qudits, PermHerm=False)
        perm2 = swaps_AB_matrix(swaps2, False, n_qudits=2*n_qudits, PermHerm=False)
        if hermitize:
            G=(perm1 @ perm2 + (perm1 @ perm2).T.conj())/2
        else:
            G=perm1 @ perm2
        perm_matrix=G.toarray()
        if herm:
            if not (perm_matrix == perm_matrix.T).all():
                continue
        stacked = np.vstack([m.flatten() for m in indep_perms + [perm_matrix]])
        if np.linalg.matrix_rank(stacked) > len(indep_perms):
            indep_swaps.append([swaps1, swaps2])
            indep_perms.append(perm_matrix)
            #print("_________Adding +1")
            G_s.append(G)
            print(len(G_s),end="\r")
    if get_indep_swaps:
        return G_s, indep_swaps
    return G_s
def get_swaps(n_qudits=4, indep=True, herm=False, get_indep_swaps=False, half=False):
    if indep:
        all_perms = list(permutations(range(n_qudits)))
        all_swaps = []
        # Print each permutation matrix and its corresponding swaps
        for p in all_perms:
            swaps = permutation_to_swaps(p)
            all_swaps.append(swaps)
    
        indep_perms = [np.eye(2**n_qudits)] ############################################################################
        indep_swaps = [[]]
        for swaps in all_swaps:
            perm_matrix = swaps_to_matrix(swaps, n_qudits)
            if herm:
                if not (perm_matrix == perm_matrix.T).all():
                    continue
            # Stack all matrices to check linear independence
            stacked = np.vstack([m.flatten() for m in indep_perms + [perm_matrix]])
            if np.linalg.matrix_rank(stacked) > len(indep_perms):
                indep_swaps.append(swaps)
                indep_perms.append(perm_matrix)
        G_s = []
        for i in range(len(indep_swaps)):
            swaps1=indep_swaps[i]
            perm1 = swaps_AB_matrix(swaps1, True, n_qudits=2*n_qudits, PermHerm=False)
            if half:
                for j in range(i, len(indep_swaps)):
                    swaps2=indep_swaps[j]
                    perm2 = swaps_AB_matrix(swaps2, False, n_qudits=2*n_qudits, PermHerm=False)
                    G=perm1 @ perm2
                    G_s.append((G+G.T.conj())/2)
            else:
                for j in range(len(indep_swaps)):
                    swaps2=indep_swaps[j]
                    perm2 = swaps_AB_matrix(swaps2, False, n_qudits=2*n_qudits, PermHerm=False)
                    G=perm1 @ perm2
                    G_s.append((G+G.T.conj())/2)
        if get_indep_swaps:
            return G_s, indep_swaps
        return G_s

def dm_pure(Neg, d=2):
    c1=math.sqrt((1+math.sqrt(1-Neg**2))/2)
    c2=math.sqrt((1-math.sqrt(1-Neg**2))/2)
    
    U = np.kron(rand_unitary_haar(2), rand_unitary_haar(2))
    
    # Build psi as dense then convert
    psi = np.zeros(4, dtype=np.complex128)
    psi[0] = c1
    psi[3] = c2
    
    phi = U @ psi
    return sps.csr_array(np.outer(phi, phi.conj()))
def dm_iso(Neg):
    d=2
    return (2*Neg+1)/3*rhoBell+(1-Neg)/6*sps.eye(d**2)
def get_data_train(points=100, both=False, pure_states=False, iso_states=False):
    if pure_states or both:
        labels_train_negSq_pure = np.linspace(0.0001, 0.9999, points)
        dms_train_pure=np.array([dm_pure(math.sqrt(negSq)) for negSq in labels_train_negSq_pure])
        labels_train_pure=labels_train_negSq_pure
        if not both:
            return labels_train_pure, dms_train_pure
    if iso_states or both:
        labels_train_negSq_iso = np.linspace(0.0001, 0.9999, points)
        dms_train_iso=[dm_iso(math.sqrt(negSq)) for negSq in labels_train_negSq_iso]
        labels_train_iso = labels_train_negSq_iso
        if not both:
            return labels_train_iso, dms_train_iso
    if not (pure_states or iso_states or both):
        return np.array([]),np.array([])
    labels_train=np.concatenate((labels_train_iso, labels_train_pure))
    dms_train=np.concatenate((dms_train_iso, dms_train_pure))
    return dms_train,labels_train 

def aux_info(dms, labels, pars, ansatz, n_copies=1):
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

def gen_even_ent_data(n, n_inp=2, mixed=True, marks="neg", n_chunks=100, eps=0):
    """ Generates a data set of states with evenly distributed entanglements. """
    
    d = 2**n_inp
    
    if marks == "neg":
        ent_measure_func = two_subsys_negativity
    elif marks == "negSq":
        ent_measure_func = two_subsys_negativity
    elif marks == "con":
        ent_measure_func = concurrence 
    if marks == "tangle":
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
            if marks == "tangle" or marks == "negSq":
                ent = ent_measure_func(state)**2
            else:
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


def get_dict_pars_swaps(pars,printing=True, return_dict_list=False,swapsA_all=None, swapsB_all=None, G_swaps=None, joined=False):
    # Small perturbation by i*1e-14 for better distinguishability of repeating pars
    pars_upd=[pars[i]+i*1e-14 for i in range(len(pars))]
    pars_sorted=sorted(pars_upd, key=lambda x: abs(x), reverse=True)
    # listing the sorted items' indexes that are from the items=pars_upd:
    # i.e. if orignal = [1.2, 3.4, 2.2, 0.9], sorted = [0.9, 1.2, 2.2, 3.4], pars_sorted=[3,0,2,1]
    inds=[]
    for i in range(len(pars_sorted)):
        inds.append(list(pars_upd).index(pars_sorted[i]))
    if G_swaps is None:
        if swapsA_all is None or swapsB_all is None:
            if joined:
                G_swaps = [[a, b] for a, b in zip(indep_swaps,indep_swaps)]
            else:
                G_swaps = [[a, b] for a in indep_swaps for b in indep_swaps]
        else:
            if joined:
                G_swaps = [[a, b] for a,b in zip(swapsA_all,swapsB_all)]
            else:
                G_swaps = [[a, b] for a in swapsA_all for b in swapsB_all]
    #print(len(G_swaps))
    dict_list=[]
    for ind in inds:
        par = pars_upd[ind]
        swapsA, swapsB = G_swaps[ind]
        dict_list.append([par, swapsA, swapsB])
        if printing:
            # Format par with 1 decimal place and fixed width
            par_str = f"{par:>8.4f}"
    
            # Format swap lists with consistent indentation
            swapsA_str = "[" + ", ".join(f"{pair}" for pair in swapsA) + "]"
            swapsB_str = "[" + ", ".join(f"{pair}" for pair in swapsB) + "]"
    
            print(f"θ = {par_str} | Swaps A: {swapsA_str:<30} | Swaps B: {swapsB_str}")
    if return_dict_list:
        return dict_list
def get_dict_pars_Gswaps(pars,printing=True, return_dict_list=False,G_swaps=None):
    # Small perturbation by i*1e-14 for better distinguishability of repeating pars
    pars_upd=[pars[i]+i*1e-14 for i in range(len(pars))]
    pars_sorted=sorted(pars_upd, key=lambda x: abs(x), reverse=True)
    # listing the sorted items' indexes that are from the items=pars_upd:
    # i.e. if orignal = [1.2, 3.4, 2.2, 0.9], sorted = [0.9, 1.2, 2.2, 3.4], pars_sorted=[3,0,2,1]
    inds=[]
    for i in range(len(pars_sorted)):
        inds.append(list(pars_upd).index(pars_sorted[i]))
    print(len(G_swaps))
    dict_list=[]
    for ind in inds:
        par = pars_upd[ind]
        swapsA, swapsB = G_swaps[ind]
        dict_list.append([par, swapsA, swapsB])
        if printing:
            # Format par with 1 decimal place and fixed width
            par_str = f"{par:>8.4f}"
    
            # Format swap lists with consistent indentation
            swapsA_str = "[" + ", ".join(f"{pair}" for pair in swapsA) + "]"
            swapsB_str = "[" + ", ".join(f"{pair}" for pair in swapsB) + "]"
    
            print(f"θ = {par_str} | Swaps A: {swapsA_str:<30} | Swaps B: {swapsB_str}")
    if return_dict_list:
        return dict_list
#get_dict_pars_swaps(pars_4_numcust, printing=False, return_dict_list=False)
def fkron(a, b):  # Make sure this is defined
    return sps.kron(a, b)

def compute_LS_worker(args):
    i, dms_i, ansatz, n_copies = args
    dm_c = reduce(fkron, [dms_i] * n_copies)
    n_pars = len(ansatz)
    L_i = zeros(n_pars)
    S_i = zeros((n_pars, n_pars))
    
    for j in range(n_pars):
        op_loc = dm_c @ ansatz[j]
        L_i[j] = op_loc.diagonal().sum().real#sps.trace(op_loc).real
        for k in range(n_pars):
            S_i[j][k] = (op_loc @ ansatz[k]).diagonal().sum().real#sps.trace(op_loc @ ansatz[k]).real
    
    return i, L_i, S_i

from multiprocessing import Pool, cpu_count

def parallel_compute_LS(dms, ansatz, n_copies=1, n_processes=None):
    n_train = len(dms)
    n_pars = len(ansatz)
    L = np.zeros((n_train, n_pars))
    S = np.zeros((n_train, n_pars, n_pars))
    
    if n_processes is None:
        n_processes = cpu_count()
    
    time_start = time()
    
    args = [(i, dms[i], ansatz, n_copies) for i in range(n_train)]
    
    with Pool(processes=n_processes) as pool:
        results = pool.imap_unordered(compute_LS_worker, args)
        
        for count, (i, L_i, S_i) in enumerate(results, 1):
            L[i] = L_i
            S[i] = S_i
            elapsed = time() - time_start
            print(f"Progress: {count}/{n_train} | Time: {elapsed:.2f}s", end="\r")
    
    time_finish = time() - time_start
    print(f"Completed in {time_finish:.2f}s".ljust(50))
    return L, S

def merging_terms(sorted_pars_Gswaps=None, n_qudits=4, distance=1e-3, return_Gs=False):
    if sorted_pars_Gswaps is None:
        return None
    merged_dict_list=[[sorted_pars_Gswaps[0][0], [[sorted_pars_Gswaps[0][1],sorted_pars_Gswaps[0][2]]]]]
    merged_dict_Gs=[swaps_AB_matrix(swaps=sorted_pars_Gswaps[0][1], A=True, n_qudits=2*n_qudits)@swaps_AB_matrix(swaps=sorted_pars_Gswaps[0][2], A=False, n_qudits=2*n_qudits)]
    for i in range(1,len(sorted_pars_Gswaps)):
        par=sorted_pars_Gswaps[i][0]
        if abs(par-merged_dict_list[-1][0])<distance:
            merged_dict_list[-1][1].append([sorted_pars_Gswaps[i][1],sorted_pars_Gswaps[i][2]])
            if return_Gs:
                merged_dict_Gs[-1]+=swaps_AB_matrix(swaps=sorted_pars_Gswaps[i][1], A=True, n_qudits=2*n_qudits)@swaps_AB_matrix(swaps=sorted_pars_Gswaps[i][2], A=False, n_qudits=2*n_qudits)
        else:
            new_merge=[sorted_pars_Gswaps[i][0], [[sorted_pars_Gswaps[i][1],sorted_pars_Gswaps[i][2]]]]
            merged_dict_list.append(new_merge)
            if return_Gs:
                merged_dict_Gs.append(swaps_AB_matrix(swaps=sorted_pars_Gswaps[i][1], A=True, n_qudits=2*n_qudits)@swaps_AB_matrix(swaps=sorted_pars_Gswaps[i][2], A=False, n_qudits=2*n_qudits))
    if return_Gs:
        return merged_dict_Gs
    return merged_dict_list


n_copies=4

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams['font.family']='serif'
matplotlib.rcParams['axes.unicode_minus']=False
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rc('text', usetex=False)
def plot_prediction(pars, G_s,n_copies=n_copies, dms_test=[], labels_test=[], results_only=False, save_plot=None, titleMSEVar=True, label_p=True):
    ansatz=[G.toarray() for G in G_s]
    expecs_test, disps_test, H = aux_info(dms_test, labels_test, pars, ansatz, n_copies=n_copies)
    total_mse=sum([(e1-e2)**2 for e1, e2 in zip(expecs_test,labels_test)])/len(labels_test)
    total_var=sum(disps_test)/len(labels_test)
    if results_only:
        return total_mse, total_var
    purities_test = [trace(dm@dm).real for dm in dms_test]
    # Create colormaps (truncated to 0.25-1.0 range)
    cmap1 = plt.get_cmap("Purples")
    colors1 = cmap1(np.linspace(0.25, 1., 256))  # Use 256 for smooth gradient
    cmap1 = LinearSegmentedColormap.from_list("truncated_Purples", colors1)
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))  # Wider figure for colorbars
    plt.rcParams['axes.axisbelow'] = True
    
    # Scatter plots (assign to variables)
    sc1 = plt.scatter(labels_test, expecs_test, c=purities_test, cmap=cmap1, s=20,edgecolor='none', label="Numerics")
    # if label_p:
    #     sc1 = plt.scatter(labels_test, expecs_test, c=purities_test, cmap=cmap1, s=20,edgecolor='none', label="Numerics")
    # else:
    #     sc1 = plt.scatter(labels_test, expecs_test, c=purities_test, cmap=cmap1, s=20,edgecolor='none')

    print(f"MSE=%.8f"%(total_mse))
    plt.plot(labels_test, labels_test, color="black", lw=1)

    if titleMSEVar:
        plt.title(f"MSE=%.8f"%(total_mse))
    plt.xlabel(r"$N^2$")
    plt.ylabel(r"$\mathsf{N}^2$")
    plt.grid()
    # plt.tight_layout()
    if label_p:
        plt.legend()
    # # Add colorbars with labels
    plt.colorbar()
    # Set consistent color limits for both
    plt.clim(0.25, 1)
    
    #plt.tight_layout()
    if save_plot is not None:
        plt.savefig(save_plot[0], bbox_inches='tight')
    plt.show()
    
    
    
    print(f"Var=%.8f"%(total_var))
    plt.figure(figsize=(10, 6))
    plt.rcParams['axes.axisbelow'] = True
    # Create figure and axis
    #plt.figure(figsize=(10, 6))  # Wider figure for colorbars
    #plt.rcParams['axes.axisbelow'] = True
    plt.scatter(labels_test, disps_test, c=purities_test, cmap=cmap1, s=20, edgecolor='none', label="Numerics")
    if titleMSEVar:
        plt.title(f"Total Var=%.8f"%(total_var))
    plt.xlabel(r"$N^2$")
    plt.ylabel(r"$\Delta^2 H$")
    plt.colorbar()
    plt.clim(0.25, 1)
    plt.grid()
    # plt.tight_layout()
    if label_p:
        plt.legend()
    if save_plot is not None:
        plt.savefig(save_plot[1], bbox_inches='tight')

    plt.show()