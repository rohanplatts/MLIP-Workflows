from ase import Atoms
from pathlib import Path
import numpy as np 

def write_contcar_for_plumipy(atoms: Atoms, path: str | Path, comment: str = "CONTCAR_GS (ASE/MLIP)"):
    """Write a VASP CONTCAR matching plumipy parser expectations.

    Args:
        atoms (Atoms): ASE Atoms object to write.
        path (str | Path): Output path for the CONTCAR file.
        comment (str): Header comment line for the CONTCAR.

    Returns:
        None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cell = np.asarray(atoms.cell)  # 3x3, row vectors
    frac = atoms.get_scaled_positions(wrap=True)

    # Preserve atom order; just compute species + counts in first-appearance order.
    syms = atoms.get_chemical_symbols()
    species_order: list[str] = []
    counts: list[int] = []
    idx = {}
    for s in syms:
        if s not in idx:
            idx[s] = len(species_order)
            species_order.append(s)
            counts.append(0)
        counts[idx[s]] += 1

    with path.open("w", encoding="utf-8") as f:
        f.write(f"{comment}\n")
        f.write("1.0\n")
        for v in cell:
            f.write(f" {v[0]: .16f} {v[1]: .16f} {v[2]: .16f}\n")
        f.write(" ".join(species_order) + "\n")
        f.write(" ".join(str(c) for c in counts) + "\n")
        f.write("Direct\n")
        for r in frac:
            f.write(f" {r[0]: .16f} {r[1]: .16f} {r[2]: .16f}\n")

def write_minimal_outcar_for_plumipy(atoms: Atoms, path: str | Path, header: str = "OUTCAR_GS (synthetic; ASE/MLIP)"):
    """Write a minimal OUTCAR that plumipy can parse for forces.

    Args:
        atoms (Atoms): ASE Atoms object providing positions and forces.
        path (str | Path): Output path for the OUTCAR file.
        header (str): Header line for the OUTCAR file.

    Returns:
        None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pos = atoms.get_positions()        # Angstrom
    frc = atoms.get_forces()           # eV/Angstrom

    with path.open("w", encoding="utf-8") as f:
        f.write(f"{header}\n")
        f.write(" POSITION                                          TOTAL-FORCE (eV/Angst)\n")
        f.write(" -----------------------------------------------------------------------------------\n")
        for (x, y, z), (fx, fy, fz) in zip(pos, frc):
            # 6  slices columns 4..6 fo the forces 
            f.write(f" {x:16.10f} {y:16.10f} {z:16.10f} {fx:16.10f} {fy:16.10f} {fz:16.10f}\n")
        f.write(" -----------------------------------------------------------------------------------\n")
        f.write(" total drift:  0.000000  0.000000  0.000000\n")
    
