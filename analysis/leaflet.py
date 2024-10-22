import MDAnalysis as mda
from sklearn.cluster import DBSCAN
import numpy as np

def detect_leaflets(top_file = 'wrap.gro', lipid_resname='ucer2', headgroup_name='O80', tailgroup_name='C71', eps=1, min_samples=10):
    # Load result files
    res = mda.Universe(top_file)

    # Selecting lipid headgroup and tailgroup atoms
    lipids = res.select_atoms(f'resname {lipid_resname}')
    headgroups = res.select_atoms(f'resname {lipid_resname} and name {headgroup_name}')
    tailgroups = res.select_atoms(f'resname {lipid_resname} and name {tailgroup_name}')

    # Extracting z-coordinates of atoms
    head_z_coords = headgroups.positions[:, 2]
    tail_z_coords = tailgroups.positions[:, 2]

    # Apply DBSCAN for both head and tail groups
    dbscan_head = DBSCAN(eps=eps, min_samples=min_samples)
    head_clusters = dbscan_head.fit_predict(head_z_coords.reshape(-1, 1))
    
    dbscan_tail = DBSCAN(eps=eps, min_samples=min_samples)
    tail_clusters = dbscan_tail.fit_predict(tail_z_coords.reshape(-1, 1))

    # Count number of leaflets
    n_leaflets = len(set(tail_clusters)) - (1 if -1 in tail_clusters else 0)

    # Verifying with headgroup clusters
    head_group_clusters = len(set(head_clusters)) - (1 if -1 in head_clusters else 0)
    expected_leaflets = (head_group_clusters * 2) - 2

    # Adjust eps if expected leaflets don't match
    loop = 0
    while expected_leaflets != n_leaflets and loop < 10:
        eps -= 0.05
        dbscan_tail_new = DBSCAN(eps=eps, min_samples=min_samples)
        tail_clusters = dbscan_tail_new.fit_predict(tail_z_coords.reshape(-1, 1))
        n_leaflets = len(set(tail_clusters)) - (1 if -1 in tail_clusters else 0)
        loop += 1
        if expected_leaflets == n_leaflets:
            break

    return n_leaflets
