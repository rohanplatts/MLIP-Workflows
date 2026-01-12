"""
Organised ASE + MACE phonons workflow
- Builds a system (diamond / hBN)
- Optionally inserts a defect (NV / C2 dimer)
- Relaxes
- Computes phonons
- Plots: (bands + DOS) for perfect crystals, (DOS only) for defects

You only need to edit:
  CONFIG["system"] and CONFIG["defect"] and sizes.
"""

# ===========================
# 0) Imports + warnings
# ===========================
import os
import sys
import shutil
import warnings
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from ase.io import read
from ase.build import bulk, make_supercell
from ase.optimize import BFGS
from ase.phonons import Phonons

from mace.calculators import mace_mp

# Optional (turn off on HPC/headless)
try:
    from ase.visualize import view
    HAVE_VIEW = True
except Exception:
    HAVE_VIEW = False

print("PYTHON:", sys.executable)

warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)

EV_TO_THz = 241.79893  # 1 eV = 241.79893 THz


# ===========================
# 1) Configuration
# ===========================
CONFIG = {
    # --- choose what you are running ---
    "system": "diamond",      # "diamond" or "hbn_bulk"
    "defect": None,           # None, "nv", or "c2"

    # --- structure sizes ---
    # For diamond: primitive has 2 atoms. 4x4x4 => 128 atoms.
    "diamond_cell_repeat": (4, 4, 4),

    # For hBN bulk: you'll usually read a POSCAR and then repeat it.
    "hbn_poscar": r"C:\Users\rnpla\Desktop\2026\mlip_phonons\input_files\BN.poscar",
    "hbn_cell_repeat": (4, 4, 2),  # example; tune to your use case

    # --- relaxation ---
    "relax_fmax": 0.01,
    "relax_traj": "relax.traj",
    "show_view": False,   # set True only if local + GUI

    # --- MACE calculator ---
    "mace_model": "small-omat-0",
    "device": "cuda",
    "dtype": "float64",

    # --- phonons ---
    # Perfect crystals: use ph_supercell like (4,4,4) starting from primitive cell.
    # Defects: treat defect supercell as the "primitive" => use (1,1,1).
    "delta": 0.01,

    # --- DOS ---
    "dos_npts": 4000,
    "dos_width_thz": 0.25,      # 0.2–0.5 good range
    "dos_kpts_perfect": None,   # if None, auto-chosen below
    "dos_kpts_defect": None,    # if None, auto-chosen below
}


# ===========================
# 2) Defect constructors
# ===========================
def place_nv_diamond(atoms):
    """Replace center-most C with N and remove its nearest neighbor to create a vacancy."""
    atoms = atoms.copy()

    frac = atoms.get_scaled_positions(wrap=True)
    symbols = np.array(atoms.get_chemical_symbols())
    c_inds = np.where(symbols == "C")[0]

    dm = frac[c_inds] - 0.5
    dm -= np.round(dm)
    n_idx = c_inds[np.argmin(np.linalg.norm(dm, axis=1))]

    dists = atoms.get_distances(n_idx, c_inds, mic=True)
    dists[c_inds == n_idx] = np.inf
    v_idx = c_inds[np.argmin(dists)]

    atoms[n_idx].symbol = "N"
    atoms.pop(v_idx)

    print(f"NV Center: N placed at {n_idx}, Vacancy created at (old index) {v_idx}")
    return atoms


def place_cc_dimer_hbn(atoms):
    """Replace center-most B and its nearest in-plane N with C-C (a C2 dimer substitution)."""
    atoms = atoms.copy()

    frac = atoms.get_scaled_positions(wrap=True)
    symbols = np.array(atoms.get_chemical_symbols())

    b_inds = np.where(symbols == "B")[0]
    dm = frac[b_inds] - 0.5
    dm -= np.round(dm)
    iB = b_inds[np.argmin(np.linalg.norm(dm, axis=1))]

    n_inds = np.where(symbols == "N")[0]
    same_layer = n_inds[np.isclose(frac[n_inds, 2], frac[iB, 2], atol=1e-4)]

    d_bn = frac[same_layer] - frac[iB]
    d_bn -= np.round(d_bn)
    d_cart = d_bn @ atoms.cell
    iN = same_layer[np.argmin(np.linalg.norm(d_cart, axis=1))]

    atoms[iB].symbol = "C"
    atoms[iN].symbol = "C"

    print(f"hBN C2: Replaced B[{iB}] and N[{iN}] with C,C")
    return atoms


# ===========================
# 3) Build structure
# ===========================
def build_system(cfg):
    sys_name = cfg["system"]
    defect = cfg["defect"]

    if sys_name == "diamond":
        prim = bulk("C", "diamond", a=3.567)
        S = np.diag(cfg["diamond_cell_repeat"])
        atoms = make_supercell(prim, S)

        if defect == "nv":
            atoms = place_nv_diamond(atoms)

    elif sys_name == "hbn_bulk":
        # read POSCAR from Materials Project / VASP export
        atoms0 = read(cfg["hbn_poscar"])
        S = np.diag(cfg["hbn_cell_repeat"])
        atoms = make_supercell(atoms0, S)

        if defect == "c2":
            atoms = place_cc_dimer_hbn(atoms)

    else:
        raise ValueError(f"Unknown system '{sys_name}'")

    return atoms


# ===========================
# 4) Calculator + relaxation
# ===========================
def make_calc(cfg):
    return mace_mp(
        model=cfg["mace_model"],
        device=cfg["device"],
        default_dtype=cfg["dtype"],
    )


def relax(atoms, cfg):
    print("Starting relaxation...")
    opt = BFGS(atoms, trajectory=cfg["relax_traj"])
    opt.run(fmax=cfg["relax_fmax"])
    print("Relaxation complete.")
    if cfg["show_view"] and HAVE_VIEW:
        view(atoms)
    return atoms


# ===========================
# 5) Plot helpers
# ===========================
def auto_emax_ev(bs=None, dos=None, pad=0.03):
    chunks = []
    if bs is not None and hasattr(bs, "energies"):
        chunks.append(np.ravel(np.asarray(bs.energies)))
    if dos is not None:
        chunks.append(np.ravel(np.asarray(dos.get_energies())))
    if not chunks:
        return None
    x = np.concatenate(chunks)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0:
        return None
    return float(x.max() * (1.0 + pad))


def plot_vertical_dos_only(dos, out_png, title=None, dpi=450):
    freqs_thz = np.asarray(dos.get_energies()) * EV_TO_THz
    g_thz = np.asarray(dos.get_weights()) / EV_TO_THz

    emax_thz = float(freqs_thz.max() * 1.02) if freqs_thz.size else 1.0

    fig, ax = plt.subplots(figsize=(4.8, 6.2), constrained_layout=True)
    ax.plot(g_thz, freqs_thz, linewidth=1.2)
    ax.fill_betweenx(freqs_thz, 0.0, g_thz, alpha=0.25)

    ax.set_xlabel("Phonon DOS (states / THz)")
    ax.set_ylabel("Frequency (THz)")
    ax.set_ylim(0.0, emax_thz)
    ax.set_xlim(left=0.0)
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="in", top=True, right=True)

    if title:
        ax.set_title(title)

    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def plot_bands_plus_vertical_dos(bs, dos, out_png, title=None, dpi=450):
    emax_ev = auto_emax_ev(bs=bs, dos=dos, pad=0.03)

    fig = plt.figure(figsize=(9, 5))

    ax = fig.add_axes([0.12, 0.1, 0.60, 0.8])
    bs.plot(ax=ax, emin=0.0, emax=emax_ev)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * EV_TO_THz:.0f}"))
    ax.set_ylabel("Frequency (THz)", fontsize=14)

    dosax = fig.add_axes([0.75, 0.1, 0.20, 0.8])
    freqs_thz = np.asarray(dos.get_energies()) * EV_TO_THz
    g_thz = np.asarray(dos.get_weights()) / EV_TO_THz
    dosax.plot(g_thz, freqs_thz, linewidth=1.0)
    dosax.fill_betweenx(freqs_thz, 0.0, g_thz, alpha=0.25)

    dosax.set_ylim(0.0, emax_ev * EV_TO_THz)
    dosax.set_xlim(left=0.0)
    dosax.set_yticks([])
    dosax.set_xlabel("DOS", fontsize=14)

    if title:
        fig.suptitle(title, y=0.98)

    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


# ===========================
# 6) Phonons config (paths + defaults)
# ===========================
def get_bandpath_config(system):
    if system == "diamond":
        special_points = {
            "G": [0.0, 0.0, 0.0],
            "X": [0.5, 0.0, 0.5],
            "W": [0.5, 0.25, 0.75],
            "K": [0.375, 0.375, 0.75],
            "L": [0.5, 0.5, 0.5],
            "U": [0.625, 0.25, 0.625],
        }
        k_path = "GXWKGLUWLK"
        return special_points, k_path

    if system == "hbn_bulk":
        special_points = {
            "G": [0.0, 0.0, 0.0],
            "M": [0.5, 0.0, 0.0],
            "K": [1 / 3, 1 / 3, 0.0],
            "A": [0.0, 0.0, 0.5],
            "L": [0.5, 0.0, 0.5],
            "H": [1 / 3, 1 / 3, 0.5],
        }
        k_path = "GMKGALHA"
        return special_points, k_path

    raise ValueError(f"No bandpath config for system '{system}'")


def choose_defaults(cfg):
    """Reasonable defaults if user left kpts=None."""
    is_defect = cfg["defect"] is not None
    if is_defect:
        if cfg["dos_kpts_defect"] is None:
            cfg["dos_kpts_defect"] = (6, 6, 6) if cfg["system"] == "diamond" else (6, 6, 3)
    else:
        if cfg["dos_kpts_perfect"] is None:
            cfg["dos_kpts_perfect"] = (16, 16, 16) if cfg["system"] == "diamond" else (16, 16, 8)

    return cfg


# ===========================
# 7) Main run
# ===========================
def main():
    cfg = choose_defaults(CONFIG)

    # build + calc + relax
    atoms = build_system(cfg)
    calc = make_calc(cfg)
    atoms.calc = calc
    atoms = relax(atoms, cfg)

    # name + cache clearing
    tag = f"{cfg['system']}" + (f"_{cfg['defect']}" if cfg["defect"] else "")
    workdir = tag
    if os.path.exists(workdir):
        shutil.rmtree(workdir)

    # phonons supercell choice
    # - perfect crystals: use larger ph_supercell (e.g., 4,4,4) starting from primitive OR moderate supercell
    # - defects: treat your defect supercell as primitive => (1,1,1)
    ph_supercell = (1, 1, 1) if cfg["defect"] else (4, 4, 4)

    ph = Phonons(atoms, calc, supercell=ph_supercell, delta=cfg["delta"], name=workdir)
    ph.run()
    ph.read(acoustic=True)
    ph.clean()

    # DOS
    width_ev = cfg["dos_width_thz"] / EV_TO_THz
    dos_kpts = cfg["dos_kpts_defect"] if cfg["defect"] else cfg["dos_kpts_perfect"]
    dos = ph.get_dos(kpts=dos_kpts).sample_grid(npts=cfg["dos_npts"], width=width_ev)

    # Bands + DOS for perfect crystals; DOS only for defects
    if cfg["defect"]:
        plot_vertical_dos_only(dos, out_png=f"{tag}_DOS.png", title=f"{tag} DOS")
        print(f"Wrote {tag}_DOS.png")
    else:
        sp, kp = get_bandpath_config(cfg["system"])
        path = atoms.cell.bandpath(kp, npoints=400, special_points=sp)
        bs = ph.get_band_structure(path)
        plot_bands_plus_vertical_dos(bs, dos, out_png=f"{tag}_Phonon.png", title=tag)
        print(f"Wrote {tag}_Phonon.png")


if __name__ == "__main__":
    main()
