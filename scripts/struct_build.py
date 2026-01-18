import sys
from ase import Atoms
import numpy as np
from pathlib import Path
import os
from ase.build import bulk
from ase.optimize import BFGS
from ase.visualize import view
from ase.build import graphene
from ase.build import bulk
from ase.calculators.emt import EMT
import json
from ase.io import read, write 
from ase.build import make_supercell
from ase import Atoms
import warnings 
from ase.phonons import Phonons
import matplotlib.pyplot as plt
import shutil
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)


# This script just simply constructs our unrelaxed atoms models. 
# ———————————— PARAMETERS
material = 'hbn_c2' # could be diamond, nv_diamond, hbn, hbn_c2
# note that hbn and hbn_c2 are both bulk.

# ————————————————————————

def place_nv_diamond(atoms):
    # 1. Grab scaled positions and symbols
    frac = atoms.get_scaled_positions(wrap=True)
    symbols = np.array(atoms.get_chemical_symbols())
    c_inds = np.where(symbols == 'C')[0]
    
    # 2. Identify the Carbon closest to the center to be our Nitrogen
    dm = frac[c_inds] - 0.5
    dm -= np.round(dm)
    n_idx = c_inds[np.argmin(np.linalg.norm(dm, axis=1))]
    
    # 3. Find its nearest neighbor to become the Vacancy
    # get_distances is much cleaner than manual norm for bulk
    dists = atoms.get_distances(n_idx, c_inds, mic=True)
    
    # Ignore the atom itself (distance 0) to find the neighbor
    dists[c_inds == n_idx] = np.inf
    v_idx = c_inds[np.argmin(dists)]
    
    # 4. Apply changes
    atoms[n_idx].symbol = 'N'
    # Use pop for the vacancy so the index shifts are handled by ASE
    atoms.pop(v_idx)
    
    print(f"NV Center: N placed at {n_idx}, Vacancy created at {v_idx}")
    return atoms

def place_cc_dimer_hbn(atoms):
    
    frac = atoms.get_scaled_positions(wrap=True)
    symbols = np.array(atoms.get_chemical_symbols())
    
    # Find the Boron closest to the fractional center (0.5, 0.5, 0.5)
    b_inds = np.where(symbols == 'B')[0]
    dm = frac[b_inds] - 0.5
    dm -= np.round(dm) # Minimum Image Convention 
    iB = b_inds[np.argmin(np.linalg.norm(dm, axis=1))]
    
    # 3. Find the closest Nitrogen in the same plane
    n_inds = np.where(symbols == 'N')[0]
    # Filter for same z-layer
    same_layer = n_inds[np.isclose(frac[n_inds, 2], frac[iB, 2], atol=1e-4)]
    
    # Find bond partner using MIC distances in Cartesian space
    d_bn = frac[same_layer] - frac[iB]
    d_bn -= np.round(d_bn)
    d_cart = d_bn @ atoms.cell
    iN = same_layer[np.argmin(np.linalg.norm(d_cart, axis=1))]
    
    # 4. Swap to Carbon
    atoms[iB].symbol = 'C'
    atoms[iN].symbol = 'C'
    
    print(f"hBN Dimer: Replaced B[{iB}] and N[{iN}]")
    return atoms

def get_material():
    # note: the current version of materials project struggles to output the primitive cell. AFLOW requires the primitive cell 
    # for its k-paths to make sense. this means that we cant rely on the MP exported poscars since they are going to be a 
    # conventional cell's unit cell, as opposed to the primitive cell. we thus construct the primitive cell in python here, for 
    # both diamond and h-BN.
    if material == 'diamond':
        atoms = bulk('C', 'diamond', a=3.567) # Pure diamond reference
        # we have a poscar file for this however it is inbuilt.
        #atoms = read(r"C:\Users\rnpla\Desktop\2026\mlip_phonons\input_files\C.poscar")
        print(atoms)
        supercell_matrix = np.diag([2,2,2])
        supercell = make_supercell(atoms, supercell_matrix)
    elif material == 'nv_diamond':
        atoms = bulk('C', 'diamond', a=3.567)
        supercell_matrix = np.diag([2,2,2])
        supercell = make_supercell(atoms, supercell_matrix)
        supercell = place_nv_diamond(supercell)
        supercell = supercell[np.argsort(supercell.get_chemical_symbols())]
    elif material == 'hbn':
        # load the unit cell from materials projet:
        # the BN.poscar was found here: 
        # https://next-gen.materialsproject.org/materials/mp-984?formula=BN
        # the materials project defaults to exporting .poscar to be read by vasp 
        atoms = read(r"/home/rnpla/projects/mlip_phonons/input_structure/BN.poscar")
        supercell_matrix = np.diag([2,2,2])
        supercell = make_supercell(atoms, supercell_matrix)
        supercell = supercell[np.argsort(supercell.get_chemical_symbols())]
    elif material == 'hbn_c2':
        atoms = read(r"/home/rnpla/projects/mlip_phonons/input_structure/BN.poscar")
        supercell_matrix = np.diag([2,2,2])
        supercell = make_supercell(atoms, supercell_matrix)
        supercell = place_cc_dimer_hbn(supercell)
        supercell = supercell[np.argsort(supercell.get_chemical_symbols())]
        # if we are dealing with monolayer version of hBN we will need the vacuum gap:
        #supercell.center(vacuum=20, axis=2) # 20 angstroms is far enough.
    else:
        raise ValueError("material not recognised")
    view(supercell)
    return atoms, supercell, supercell_matrix

atoms, supercell, supercell_matrix = get_material()

view(supercell)

if material == 'nv_diamond':
    pure_supercell = make_supercell(atoms, np.diag([2,2,2]))
    write("/home/rnpla/projects/mlip_phonons/structures/diamond_nv.poscar", supercell)
    write("/home/rnpla/projects/mlip_phonons/structures/diamond_primitive.poscar", atoms)
    write("/home/rnpla/projects/mlip_phonons/structures/diamond_super.poscar", pure_supercell)
elif material == 'hbn_c2':
    write("/home/rnpla/projects/mlip_phonons/structures/hbn_c2.poscar", supercell)
    write("/home/rnpla/projects/mlip_phonons/structures/hbn_primitive.poscar", atoms)
    write("/home/rnpla/projects/mlip_phonons/structures/hbn_super.poscar", pure_supercell)

