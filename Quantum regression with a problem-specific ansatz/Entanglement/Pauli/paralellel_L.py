from functools import reduce
from numpy import zeros, trace
from time import time
from multiprocessing import Pool, cpu_count, Array
import numpy as np
import ctypes
def fkron(A, B):
    """ A faster kronecker product taken from https://stackoverflow.com/a/56067827 """
    s = len(A)*len(B)
    return (A[:, None, :, None]*B[None, :, None, :]).reshape(s, s)
def compute_L_worker(args):
    i, dms_i, ansatz, n_copies = args
    dm_c = reduce(fkron, [dms_i] * n_copies)
    n_pars = len(ansatz)
    L_i = zeros(n_pars)
    
    for j in range(n_pars):
        op_loc = dm_c @ ansatz[j]
        L_i[j] = op_loc.diagonal().sum().real#sps.trace(op_loc).real
        
    return i, L_i

import scipy.sparse as sps
from multiprocessing import Pool, cpu_count
from time import time

def parallel_compute_L(dms, ansatz, n_copies=1, n_processes=None):
    n_train = len(dms)
    n_pars = len(ansatz)
    L = np.zeros((n_train, n_pars))
    
    if n_processes is None:
        n_processes = cpu_count()
    
    time_start = time()
    
    args = [(i, dms[i], ansatz, n_copies) for i in range(n_train)]
    
    with Pool(processes=n_processes) as pool:
        results = pool.imap_unordered(compute_L_worker, args)
        
        for count, (i, L_i) in enumerate(results, 1):
            L[i] = L_i
            elapsed = time() - time_start
            print(f"Progress: {count}/{n_train} | Time: {elapsed:.2f}s", end="\r")
    
    time_finish = time() - time_start
    print(f"Completed in {time_finish:.2f}s".ljust(50))
    return L

# def _worker_func(args):
#     """
#     Worker function for parallel processing.
#     Must be defined at top level for pickling.
#     """
#     i, dms_i, ansatz, n_copies, n_pars, dtype = args
#     # Convert shared array back to numpy array for computation
#     ansatz_np = np.array(ansatz, dtype=dtype)
#     dm_c = reduce(np.kron, [dms_i] * n_copies)
#     row = np.zeros(n_pars, dtype=np.float64)
#     for j in range(n_pars):
#         op_loc = dm_c @ ansatz_np[j]
#         row[j] = np.trace(op_loc).real
#     return i, row.tolist()


# def compute_L_parallel(dms, ansatz, n_copies=1, n_processes=None):
#     """
#     Parallel version of compute_L that distributes computation across multiple cores.
#     """
#     n_train = len(dms)
#     n_pars = len(ansatz)
#     L = zeros((n_train, n_pars))
    
#     if n_processes is None:
#         n_processes = min(cpu_count(), n_train)
    
#     time_start = time()
    
#     # Prepare data for parallel processing
#     # Create tasks - each task processes one index
#     tasks = []
#     for i in range(n_train):
#         # We need to pass the density matrix and ansatz for this task
#         # Note: This creates some memory overhead but allows parallel processing
#         tasks.append((i, dms[i], ansatz, n_copies, n_pars, dms[0].dtype))
    
#     print(f"Computing L using {n_processes} processes...")
    
#     # Process in parallel
#     with Pool(processes=n_processes) as pool:
#         results = []
#         # Use imap_unordered for better performance
#         for idx, result in enumerate(pool.imap_unordered(_worker_func, tasks), 1):
#             i, row = result
#             results.append((i, row))
            
#             # Print progress update
#             progress = len(results) / n_train * 100
#             time_elapsed = time() - time_start
#             if idx % 10 == 0:  # Update progress less frequently
#                 print(f"Progress: {progress:.1f}% ({len(results)}/{n_train}) | "
#                       f"Time: {time_elapsed:.2f}s", end="\r")
    
#     # Sort results by index and assemble into L matrix
#     results.sort(key=lambda x: x[0])
#     for i, row in results:
#         L[i] = row
    
#     time_finish = time() - time_start
#     print(f"\nComputing L: finished in {time_finish:.2f} s using {n_processes} processes")
#     return L


# # Simpler version using concurrent.futures (often easier)
# def compute_L_parallel_futures(dms, ansatz, n_copies=1, max_workers=None):
#     """
#     Parallel version using concurrent.futures.
#     This is often more robust than multiprocessing.Pool.
#     """
#     from concurrent.futures import ProcessPoolExecutor, as_completed
    
#     n_train = len(dms)
#     n_pars = len(ansatz)
#     L = zeros((n_train, n_pars))
    
#     if max_workers is None:
#         max_workers = min(cpu_count(), n_train)
    
#     time_start = time()
    
#     # Define a wrapper function for the worker
#     def process_index(i):
#         dm_c = reduce(np.kron, [dms[i]] * n_copies)
#         row = np.zeros(n_pars, dtype=np.float64)
#         for j in range(n_pars):
#             op_loc = dm_c @ ansatz[j]
#             row[j] = np.trace(op_loc).real
#         return i, row
    
#     print(f"Computing L using {max_workers} workers...")
    
#     # Submit all tasks
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         # Submit all tasks
#         futures = {executor.submit(process_index, i): i for i in range(n_train)}
        
#         # Collect results as they complete
#         completed = 0
#         for future in as_completed(futures):
#             i, row = future.result()
#             L[i] = row
#             completed += 1
            
#             # Update progress every 10 completions
#             if completed % 10 == 0:
#                 progress = completed / n_train * 100
#                 time_elapsed = time() - time_start
#                 print(f"Progress: {progress:.1f}% ({completed}/{n_train}) | "
#                       f"Time: {time_elapsed:.2f}s", end="\r")
    
#     time_finish = time() - time_start
#     print(f"\nComputing L: finished in {time_finish:.2f} s using {max_workers} workers")
#     return L


# # Alternative: Use shared memory to avoid copying large arrays
# def compute_L_parallel_shared(dms, ansatz, n_copies=1, n_processes=None):
#     """
#     Parallel version using shared memory arrays to avoid copying.
#     This is more memory efficient for large arrays.
#     """
#     from multiprocessing import shared_memory
#     import multiprocessing as mp
    
#     n_train = len(dms)
#     n_pars = len(ansatz)
    
#     # Get array shapes and dtypes
#     dm_shape = dms[0].shape
#     dm_dtype = dms[0].dtype
#     ansatz_shape = ansatz[0].shape
#     ansatz_dtype = ansatz[0].dtype
    
#     # Flatten arrays for shared memory
#     dms_flat = np.concatenate([dm.flatten() for dm in dms])
#     ansatz_flat = np.concatenate([a.flatten() for a in ansatz])
    
#     # Create shared memory
#     shm_dms = shared_memory.SharedMemory(create=True, size=dms_flat.nbytes)
#     shm_ansatz = shared_memory.SharedMemory(create=True, size=ansatz_flat.nbytes)
    
#     # Copy data to shared memory
#     dms_shared = np.ndarray(dms_flat.shape, dtype=dm_dtype, buffer=shm_dms.buf)
#     ansatz_shared = np.ndarray(ansatz_flat.shape, dtype=ansatz_dtype, buffer=shm_ansatz.buf)
#     np.copyto(dms_shared, dms_flat)
#     np.copyto(ansatz_shared, ansatz_flat)
    
#     L = zeros((n_train, n_pars))
    
#     if n_processes is None:
#         n_processes = min(cpu_count(), n_train)
    
#     time_start = time()
    
#     # Worker function that accesses shared memory
#     def worker_func_shared(indices):
#         # Attach to existing shared memory
#         shm_dms_local = shared_memory.SharedMemory(name=shm_dms.name)
#         shm_ansatz_local = shared_memory.SharedMemory(name=shm_ansatz.name)
        
#         # Create numpy arrays from shared memory
#         dms_local = np.ndarray(dms_flat.shape, dtype=dm_dtype, buffer=shm_dms_local.buf)
#         ansatz_local = np.ndarray(ansatz_flat.shape, dtype=ansatz_dtype, buffer=shm_ansatz_local.buf)
        
#         # Reshape to original structure
#         dms_reshaped = []
#         dm_size = dm_shape[0] * dm_shape[1]
#         for i in range(n_train):
#             start = i * dm_size
#             end = (i + 1) * dm_size
#             dms_reshaped.append(dms_local[start:end].reshape(dm_shape))
        
#         ansatz_reshaped = []
#         ansatz_size = ansatz_shape[0] * ansatz_shape[1]
#         for j in range(n_pars):
#             start = j * ansatz_size
#             end = (j + 1) * ansatz_size
#             ansatz_reshaped.append(ansatz_local[start:end].reshape(ansatz_shape))
        
#         # Compute for assigned indices
#         results = []
#         for i in indices:
#             dm_c = reduce(np.kron, [dms_reshaped[i]] * n_copies)
#             row = np.zeros(n_pars, dtype=np.float64)
#             for j in range(n_pars):
#                 op_loc = dm_c @ ansatz_reshaped[j]
#                 row[j] = np.trace(op_loc).real
#             results.append((i, row))
        
#         # Clean up
#         shm_dms_local.close()
#         shm_ansatz_local.close()
        
#         return results
    
#     # Split indices for parallel processing
#     indices = list(range(n_train))
#     chunk_size = max(1, n_train // n_processes)
#     chunks = [indices[i:i + chunk_size] for i in range(0, n_train, chunk_size)]
    
#     print(f"Computing L using {n_processes} processes with shared memory...")
    
#     # Process chunks in parallel
#     with mp.Pool(processes=n_processes, initializer=lambda: None) as pool:
#         all_results = []
#         for chunk_results in pool.imap_unordered(worker_func_shared, chunks):
#             all_results.extend(chunk_results)
            
#             # Update progress
#             progress = len(all_results) / n_train * 100
#             time_elapsed = time() - time_start
#             print(f"Progress: {progress:.1f}% ({len(all_results)}/{n_train}) | "
#                   f"Time: {time_elapsed:.2f}s", end="\r")
    
#     # Assemble results
#     all_results.sort(key=lambda x: x[0])
#     for i, row in all_results:
#         L[i] = row
    
#     # Clean up shared memory
#     shm_dms.close()
#     shm_dms.unlink()
#     shm_ansatz.close()
#     shm_ansatz.unlink()
    
#     time_finish = time() - time_start
#     print(f"\nComputing L: finished in {time_finish:.2f} s")
#     return L


# # Simple wrapper to choose the best method
# def compute_L(dms, ansatz, n_copies=1, method='futures', n_workers=None):
#     """
#     Enhanced version with multiple parallelization options.
    
#     Parameters:
#     - dms: List of density matrices
#     - ansatz: List of operators
#     - n_copies: Number of copies for Kronecker product
#     - method: 'futures' (default), 'multiprocessing', 'shared', or 'sequential'
#     - n_workers: Number of workers (None = auto)
#     """
#     if method == 'sequential':
#         # Original sequential version
#         n_train = len(dms)
#         n_pars = len(ansatz)
#         L = zeros((n_train, n_pars))
#         time_start = time()
#         for i in range(n_train):
#             print("Computing L: i=%d | time passed: %.2f s" % (i, time() - time_start), end="\r")
#             dm_c = reduce(np.kron, [dms[i]] * n_copies)
#             for j in range(n_pars):
#                 op_loc = dm_c @ ansatz[j]
#                 L[i][j] = trace(op_loc).real
#         time_finish = time() - time_start
#         print("\nComputing L: finished in %.2f s" % time_finish)
#         return L
    
#     elif method == 'multiprocessing':
#         return compute_L_parallel(dms, ansatz, n_copies, n_workers)
    
#     elif method == 'shared':
#         try:
#             return compute_L_parallel_shared(dms, ansatz, n_copies, n_workers)
#         except Exception as e:
#             print(f"Shared memory method failed: {e}. Falling back to futures method.")
#             return compute_L_parallel_futures(dms, ansatz, n_copies, n_workers)
    
#     else:  # 'futures' is default
#         return compute_L_parallel_futures(dms, ansatz, n_copies, n_workers)