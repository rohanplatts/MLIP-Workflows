from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from ase.io import read, write
from ase import Atoms
import numpy as np 
from numpy import ndarray
from pathlib import Path
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from ase.dft.kpoints import parse_path_string



def ap(struct_ase: Atoms): 
    """Convert an ASE Atoms object to a PhonopyAtoms object.

    Args:
        struct_ase (Atoms): ASE Atoms object.

    Returns:
        PhonopyAtoms: Phonopy-compatible atoms object.
    """
    numbers_a = struct_ase.get_atomic_numbers() 
    cell_a = struct_ase.get_cell()[:]
    pos_a = struct_ase.get_positions()
    return PhonopyAtoms(cell = cell_a, positions = pos_a, numbers = numbers_a)

def pa(struct_phon: PhonopyAtoms):
    """Convert a PhonopyAtoms object to an ASE Atoms object.

    Args:
        struct_phon (PhonopyAtoms): Phonopy atoms object.

    Returns:
        Atoms: ASE Atoms object.
    """
    numbers_p = struct_phon.numbers 
    cell_p = struct_phon.cell
    pos_p = struct_phon.positions
    return Atoms(numbers = numbers_p, cell = cell_p, positions = pos_p, pbc = True)

def get_primitive_matrix(unitcell: Atoms, primitive_m_a: ndarray | Atoms):
    """Compute the primitive matrix that maps unitcell to primitive cell.

    Args:
        unitcell (Atoms): ASE Atoms unit cell.
        primitive_m_a (ndarray | Atoms): Primitive matrix or primitive cell Atoms.

    Returns:
        ndarray: 3x3 primitive matrix.
    """
    if isinstance(primitive_m_a, Atoms):
        primitive_cell_lvs = primitive_m_a.get_cell()[:].T # lattice vecs are stored as row vecs so .T
        unitcell_lvs = unitcell.get_cell()[:].T 
        prim_matrix = np.linalg.inv(unitcell_lvs) @ primitive_cell_lvs
    else: 
        prim_matrix = primitive_m_a
    return prim_matrix

def get_phonons(
    unitcell_a: Atoms | list[Atoms], 
    mlip_calc, 
    supercell_m: ndarray | tuple[int, int, int] | list[int] = np.eye(3), 
    primitive_m_a: ndarray | Atoms | list[Atoms] = np.eye(3), 
    delta: float = 0.01,
    outdir: Path | str | None = None,
): 
    """Build a Phonopy object by computing forces on displaced supercells.

    Args:
        unitcell_a (Atoms | list[Atoms]): Relaxed unit cell as ASE Atoms.
        mlip_calc (Any): ASE-compatible calculator used for forces.
        supercell_m (ndarray | tuple[int, int, int] | list[int]): Supercell matrix.
        primitive_m_a (ndarray | Atoms | list[Atoms]): Primitive matrix or primitive cell.
        delta (float): Displacement distance in angstrom.
        outdir (Path | str | None): Output path for serialized phonon data.

    Returns:
        Phonopy: Phonopy object with force constants computed.
    """
    supercell_m = np.array(supercell_m)
    supercell_m = supercell_m if supercell_m.shape == (3, 3) else np.diag(supercell_m)
    prim_matrix = get_primitive_matrix(unitcell_a, primitive_m_a)
    unitcell_p = ap(unitcell_a) # PhonopyAtoms object
    phonon = Phonopy(unitcell_p, supercell_m, prim_matrix)
    phonon.generate_displacements(distance = delta)
    supercells = phonon.supercells_with_displacements # list of phonopyAtoms objects

    num_sc = len(supercells) 
    num_atoms = len(supercells[0]) 
    num_dof = 3 #fx, fy, fz
    sets_of_forces = np.zeros((num_sc, num_atoms, num_dof)) #initialise phonopy acessible force constants 
    
    # phonopy accepts forces of this form: 
    # [ [ [ f_1x, f_1y, f_1z ], [ f_2x, f_2y, f_2z ], ... ], # first supercell
    # [ [ f_1x, f_1y, f_1z ], [ f_2x, f_2y, f_2z ], ... ] ]  # second supercell
    if num_sc > 50:  
        print(f"Number of displaced supercells to calculate is: {num_sc}, time factor: {num_sc/10} ")            

        for i, supercell_p in enumerate(supercells):
            supercell_a = pa(supercell_p)
            supercell_a.calc = mlip_calc
            
            sets_of_forces[i,:,:] = supercell_a.get_forces()
            if i % 10 == 0 or i == num_sc:
                print(f"forces for {i} / {num_sc} supercells.")
    else:
        for i, supercell_p in enumerate(supercells):
            supercell_a = pa(supercell_p)
            supercell_a.calc = mlip_calc
            sets_of_forces[i,:,:] = supercell_a.get_forces()

    print("beginning force constant calculation:")
    phonon.forces = sets_of_forces
    phonon.produce_force_constants()
    if outdir is not None:
        outpath = Path(outdir) if isinstance(outdir, str) else outdir
        outpath.parent.mkdir(parents=True, exist_ok=True)
        phonon.save(settings={"force_constants": True}, filename=str(outpath)) 
    return phonon


def get_phonopy_kpath_ase(
    unitcell_a: Atoms,
    eps: float = 2e-4,
    override_path: str | None = None,
):
    """Build a k-point path and labels using ASE's AFLOW conventions.

    Args:
        unitcell_a (Atoms): ASE Atoms object for path detection.
        eps (float): Symmetry tolerance for bandpath detection.
        override_path (str | None): Optional explicit path string.

    Returns:
        tuple[list[list[list[float]]], list[str]]: Path segments and flattened labels.
    """
    # ASE chooses the default SC/AFLOW path for the inferred Bravais lattice if override_path is None. 
    bp = unitcell_a.cell.bandpath(path=override_path, eps=eps)
    print('bp is', bp)
    # Parse e.g. "GXWKGLUWLK,UX" -> [["G","X","W","K","G","L","U","W","L","K"], ["U","X"]] 
    sections = parse_path_string(bp.path) # takes the form of (diamond as an example): 
    #[['G', 'X', 'W', 'K', 'G', 'L', 'U', 'W', 'L', 'K'], ['U', 'X']]
    sp = bp.special_points# takes the form of (daimond as an example): 
    #{'G': array([0., 0., 0.]), 'K': array([0.375, 0.375, 0.75 ]), 'L': array([0.5, 0.5, 0.5]), 'U': array([0.625, 0.25 , 0.625]), 'W': array([0.5 , 0.25, 0.75]), 'X': array([0.5, 0. , 0.5])}
    
    # require the path to be a list exactly like sections, but with the coordinates instead
    # of the letters. We also then require labels to be exacly like sections except just a 
    # continuous string of the ordered labels, with no delimiters to signify jumps, as that 
    # information is retained by path list[list[listfloat]] 

    path = [seg.copy() for seg in sections] 
    labels = []

    for i, seg in enumerate(sections):
        for j, lbl in enumerate(seg): 
            if lbl not in sp:
                raise KeyError(
                    f"ASE bandpath label {lbl!r} not found in special_points. "
                    f"Available: {sp.keys()}"
                )
            coord = sp[lbl]
            path[i][j] = coord.tolist()
            labels.append(lbl) if lbl not in ("g", "G","Γ") else labels.append("$\\Gamma$")
            
    return path, labels

#TODO: figure out if it is worth accounting for getting the band structure of a defect. particular question is whether if i do get the band path of the defect by using a pristing primitive cell, will the band structure calculator account for the slight deviations from those coordaintes due to relaxation?

import numpy as np
from pathlib import Path

def write_gamma_band_yaml_for_plumipy(phonon, outpath: str | Path):
    """Write a single-Gamma band.yaml file for plumipy.

    Args:
        phonon (Phonopy): Phonopy object with force constants.
        outpath (str | Path): Output path for band.yaml.

    Returns:
        None
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # One "segment" with a single q-point at Gamma.
    qpoints = [np.array([[0.0, 0.0, 0.0]], dtype=float)]
    path_connections = [False]

    phonon.run_band_structure(
        qpoints,
        path_connections=path_connections,
        with_eigenvectors=True,
    )
    phonon.write_yaml_band_structure(filename=str(outpath))

def get_band_structure(
    unitcell_NOTrelaxed: Atoms | list[Atoms],
    phonon: Phonopy,
    npts: int,
    outdir_bandyaml: Path | str | None = None,
):
    """Compute phonon band structure along an AFLOW k-path.

    Args:
        unitcell_NOTrelaxed (Atoms | list[Atoms]): Unrelaxed unit cell for k-path detection.
        phonon (Phonopy): Phonopy object with force constants.
        npts (int): Number of points along each path segment.
        outdir_bandyaml (Path | str | None): Output path for band.yaml.

    Returns:
        Phonopy: Phonopy object with band structure attached.
    """
    path, labels = get_phonopy_kpath_ase(unitcell_NOTrelaxed)
    # from phonopy documentation:
    qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=npts)
    phonon.run_band_structure(qpoints, path_connections=connections, labels=labels, with_eigenvectors = True)

    if outdir_bandyaml is not None:
        outdir_bandyaml = Path(outdir_bandyaml) if isinstance(outdir_bandyaml, str) else outdir_bandyaml
        outdir_bandyaml.parent.mkdir(parents=True, exist_ok=True)
        phonon.write_yaml_band_structure(filename=str(outdir_bandyaml))
    return phonon

def get_dos(
    phonon: Phonopy,
    kpts_mesh: list[int] = [12, 12, 12],
    outdir: Path | str | None = None, 
):
    #TODO: add phonopy native saving of Dos raw data. 
    """Compute the total phonon density of states.

    Args:
        phonon (Phonopy): Phonopy object with force constants.
        kpts_mesh (list[int]): Monkhorst-Pack mesh for DOS.
        outdir (Path | str | None): Unused placeholder for future output.

    Returns:
        Phonopy: Phonopy object with DOS attached.
    """
    phonon.run_mesh(kpts_mesh)
    phonon.run_total_dos()
    return phonon





        

