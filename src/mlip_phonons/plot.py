from __future__ import annotations
from phonopy import Phonopy
import phonopy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def _auto_title_from_phonon(phonon: Phonopy) -> str | None:
    """Build a descriptive plot title from a Phonopy object.

    Args:
        phonon (Phonopy): Phonopy object with unitcell/primitive/supercell info.

    Returns:
        str | None: Title based on formula and supercell size, or None if unavailable.
    """
    cell = None
    formula = None
    for attr in ("primitive", "unitcell", "supercell"):
        cell = getattr(phonon, attr, None)
        if cell is None:
            continue
        if hasattr(cell, "get_chemical_formula"):
            formula = cell.get_chemical_formula()
        if formula:
            break

    if not formula or cell is None:
        return None

    supercell_matrix = getattr(phonon, "supercell_matrix", None)
    if supercell_matrix is None:
        return formula

    matrix = np.asarray(supercell_matrix)
    if matrix.shape == (3, 3) and np.allclose(matrix, np.diag(np.diag(matrix))):
        dims = [int(round(v)) for v in np.diag(matrix)]
        if all(v > 0 for v in dims):
            return f"{formula} ({dims[0]}x{dims[1]}x{dims[2]})"

    return formula



def obj_plot_band(
    phonon: Phonopy | Path | str, 
    outdir: Path | str | None = None,
    title: str | None = None,
    auto_title: bool = True,
):
    """Plot the phonon band structure and optionally save to disk.

    Args:
        phonon (Phonopy | Path | str): Phonopy object or path to a phonopy YAML file.
        outdir (Path | str | None): Output file path for the plot image.
        title (str | None): Explicit plot title to use.
        auto_title (bool): If True, derive a title from the phonon object.

    Returns:
        None
    """
    if not isinstance(phonon, Phonopy):
        pho_path = Path(phonon)
        phonon = phonopy.load(pho_path)

    fig = phonon.plot_band_structure()
    if title is None and auto_title:
        title = _auto_title_from_phonon(phonon)
    if title:
        fig.suptitle(title)

    if outdir is not None: 
        outdir = Path(outdir)
        fig.savefig(fname = outdir)
    return


def obj_plot_band_dos(
    phonon: Phonopy | Path | str, 
    outdir: Path | str | None = None,
    title: str | None = None,
    auto_title: bool = True,
):
    """Plot band structure and density of states together.

    Args:
        phonon (Phonopy | Path | str): Phonopy object or path to a phonopy YAML file.
        outdir (Path | str | None): Output file path for the plot image.
        title (str | None): Explicit plot title to use.
        auto_title (bool): If True, derive a title from the phonon object.

    Returns:
        None
    """
    if not isinstance(phonon, Phonopy):
        pho_path = Path(phonon)
        phonon = phonopy.load(pho_path)

    fig = phonon.plot_band_structure_and_dos()
    if title is None and auto_title:
        title = _auto_title_from_phonon(phonon)
    if title:
        fig.suptitle(title)

    if outdir is not None: 
        outdir = Path(outdir)
        fig.savefig(fname = outdir)
    return 


def obj_plot_dos(
    phonon: Phonopy | Path | str, 
    outdir: Path | str | None = None, 
    title: str | None = None,
    auto_title: bool = True,
):
    """Plot phonon density of states.

    Args:
        phonon (Phonopy | Path | str): Phonopy object or path to a phonopy YAML file.
        outdir (Path | str | None): Output file path for the plot image.
        title (str | None): Explicit plot title to use.
        auto_title (bool): If True, derive a title from the phonon object.

    Returns:
        None
    """
    if not isinstance(phonon, Phonopy):
        pho_path = Path(phonon)
        phonon = phonopy.load(pho_path)
        
    fig = phonon.plot_total_dos()
    if title is None and auto_title:
        title = _auto_title_from_phonon(phonon)
    if title:
        fig.suptitle(title)

    if outdir is not None: 
        outdir = Path(outdir)
        fig.savefig(fname = outdir)
    return 
