# ————————————————————————
from ase.filters import ExpCellFilter

def get_kpath():
    """
    returns the k-points and path as a string. no breaks to dentoe 
    jumps required on input. to add another material provide path and k-coordinate details
    """
    if material == 'diamond' or material == 'nv_diamond': 
        # k-point path for structure
        # AFLOW Standard Paths, taken from W. Setyawan and S. Curtarolo https://www.sciencedirect.com/science/article/pii/S0927025610002697 
        # Diamond (fcc)
        # G-X-W-K-G-L-U-W-L-K; U-X (';' is a jump)
        points = {
        'G': [0.0, 0.0, 0.0],
        'X': [0.5, 0.0, 0.5],
        'W': [0.5, 0.25, 0.75],
        'K': [0.375, 0.375, 0.75],
        'L': [0.5, 0.5, 0.5],
        'U': [0.625, 0.25, 0.625]
        } 
        k_path = "GXWKGLUWLK" # diamond
    elif material == 'hbn' or material == 'hbn_c2':
        # hBN (HEX)
        # G-M-K-G-A-L-H-A
        # this can be used for the monolayer, as that is simply GMKG
        points = {
            'G': [0.0, 0.0, 0.0],
            'M': [0.5, 0.0, 0.0],
            'K': [1/3, 1/3, 0.0],
            'A': [0.0, 0.0, 0.5],
            'L': [0.5, 0.0, 0.5],
            'H': [1/3, 1/3, 0.5]
        }
        #k_path = "GMKG" # hbn monolayer
        k_path = "GMKGALHA" # hbn bulk 
    else:
        raise ValueError
    return points, k_path

points, k_path = get_kpath()

def get_phonons():
    import numpy as np
    # clear the name from cache so 'Phonons' does not get confused.
    if os.path.exists(material):
        shutil.rmtree(material)
    print(len(supercell))
    # supercell sizing has been handled already. we want the phonons of the supercell so pass 'supercell' in. (note it is relaxed)
    # even for pure crystals, we still need a supercell because the supercell is what determines 'how big does this crystal be for which 
    # you are comfortable saying long range interactinos become insignificant?' hence we require a supercell for both pure crystals and defects.
    
    if not (supercell_matrix == np.eye(3)).all():
        size = (1, 1, 1)
    else:
        size = (2,2,2)
    mlip_calc.set(device='cuda') 
    atoms.calc = mlip_calc
    
    #ecf = ExpCellFilter(atoms)          # relax cell + positions
    #opt = BFGS(ecf, trajectory="cell_relax.traj")
    #opt.run(fmax=0.01)
    ph = Phonons(supercell, mlip_calc, supercell=size, delta=0.01, name = material)
    ph.run() # phonons object
    ph.read(acoustic=True)
    ph.clean()

    # define the k-point path. this should be relative to the unit_cell, which we stored as atoms.
    path = atoms.cell.bandpath(k_path, npoints=200, special_points=points) # bandpath 
    bs = ph.get_band_structure(path) # band_structure 
    dos = ph.get_dos(kpts=(12, 12, 12)).sample_grid(npts=400, width=1e-3) # density of states
    import numpy as np

    emin_ev = np.min(np.asarray(bs.energies))
    print("Finished getting band structure and DOS. Details:")
    print("Minimum band energy (eV):", emin_ev)
    print("Minimum frequency (THz):", emin_ev * 241.79893)

    print('Obtained DOS')
    E = np.array(dos.get_energies()) # eV
    g = np.array(dos.get_weights()) # states / eV

    modes = np.trapz(g, E) # number of modes
    print("∫DOS dE =", modes)
    print("Expected =", 3 * len(supercell)) # supercell is the atoms you passed to Phonons
    
    return ph, bs, dos, path
ph, bs, dos, path = get_phonons() # = phonons, band structure, density of states, bandpath 
