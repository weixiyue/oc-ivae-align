import h5py, numpy as np

with h5py.File('data/raw/N-CMAPSS_DS02-006.h5', 'r') as f:
    A = f['A_dev'][:]
    A_test = f['A_test'][:]

print("A_var列含义: [unit, cycle, Fc, hs]")
print("dev units:", np.unique(A[:,0]).astype(int))
print("test units:", np.unique(A_test[:,0]).astype(int))

for u in np.unique(A[:,0]):
    n_cycles = len(np.unique(A[A[:,0]==u, 1]))
    print(f"  dev unit {int(u)}: {n_cycles} cycles")

for u in np.unique(A_test[:,0]):
    n_cycles = len(np.unique(A_test[A_test[:,0]==u, 1]))
    print(f"  test unit {int(u)}: {n_cycles} cycles")