#1. `CONTCAR` (structure) & `OUTCAR` (forces) for the ground state
#1. `CONTCAR` (structure) & `OUTCAR` (forces) for the excited state
#1. `band.yaml` from Phonopy (if phonopy) or DFPT/FD `OUTCAR` from VASP

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Optional, Union, Any

import numpy as np
import matplotlib.pyplot as plt



@dataclass(frozen=True)
class PLResult:
    name: str
    positions_gs: np.ndarray
    positions_es: np.ndarray
    qk: Any
    energy_k: np.ndarray
    partial_hr: np.ndarray
    energy_mev_positive: np.ndarray
    specfun_energy: np.ndarray
    t_fs: np.ndarray
    s_t: np.ndarray
    s_t_exact: np.ndarray
    g_t: np.ndarray
    energy_mev: np.ndarray
    a_e: np.ndarray
    l_e: np.ndarray
    ipr: np.ndarray

    @staticmethod
    def from_tuple(t: tuple) -> "PLResult":
        """
        Expects:
        (
            name,
            positions_gs,
            positions_es,
            qk,
            (energy_k, partial_hr),
            (energy_mev_positive, specfun_energy),
            (t_fs, s_t, s_t_exact),
            (g_t),
            (energy_mev, a_e),
            (l_e),
            ipr,
        )
        """
        if len(t) != 11:
            raise ValueError(f"Expected 11 items in result tuple, got {len(t)}")

        (
            name,
            positions_gs,
            positions_es,
            qk,
            (energy_k, partial_hr),
            (energy_mev_positive, specfun_energy),
            (t_fs, s_t, s_t_exact),
            g_t,
            (energy_mev, a_e),
            l_e,
            ipr,
        ) = t

        def _arr(x, *, name_: str) -> np.ndarray:
            a = np.asarray(x)
            if a.ndim == 0:
                raise ValueError(f"{name} -> {name_} is scalar; expected array")
            return a

        return PLResult(
            name=str(name),
            positions_gs=_arr(positions_gs, name_="positions_gs"),
            positions_es=_arr(positions_es, name_="positions_es"),
            qk=qk,
            energy_k=_arr(energy_k, name_="energy_k"),
            partial_hr=_arr(partial_hr, name_="partial_hr"),
            energy_mev_positive=_arr(energy_mev_positive, name_="energy_mev_positive"),
            specfun_energy=_arr(specfun_energy, name_="specfun_energy"),
            t_fs=_arr(t_fs, name_="t_fs"),
            s_t=_arr(s_t, name_="s_t"),
            s_t_exact=_arr(s_t_exact, name_="s_t_exact"),
            g_t=_arr(g_t, name_="g_t"),
            energy_mev=_arr(energy_mev, name_="energy_mev"),
            a_e=_arr(a_e, name_="a_e"),
            l_e=_arr(l_e, name_="l_e"),
            ipr=_arr(ipr, name_="ipr"),
        )


def plot_pl_comparison(
    *results: tuple,
    save_dir: Optional[Union[str, Path]] = None,
    prefix: str = "pl_compare",
    show: bool = True,
    scatter_alpha: float = 0.7,
    line_alpha: float = 0.9,
    eps_log: float = 1e-30,
) -> list[plt.Figure]:
    """
    Compare any number of PL results returned from calculate_spectrum.
    - Each input is the big tuple with leading 'name' for model defined.
    - Creates separate figures 
    - Optional saving: pass save_dir to write PNGs.
    Returns list of matplotlib Figure objects.
    """
    runs: list[PLResult] = [PLResult.from_tuple(r) for r in results]
    if len(runs) == 0:
        raise ValueError("No results provided.")

    save_path: Optional[Path] = None
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

    figs: list[plt.Figure] = []

    def _save(fig: plt.Figure, fname: str):
        if save_path is not None:
            fig.savefig(save_path / fname, dpi=200, bbox_inches="tight")
    # these are from core.py of plumipy
    fig = plt.figure()
    for r in runs:
        S_tot = float(np.nansum(r.partial_hr))
        plt.scatter(
            r.energy_k,
            r.partial_hr,
            s=10,
            marker="s",
            alpha=scatter_alpha,
            label=f"{r.name} (ΣS={S_tot:.3g})",
        )
    plt.xlabel("Phonon Energy (meV)")
    plt.ylabel(r"$S_k$")
    plt.title("Partial Huang–Rhys factors")
    plt.legend()
    figs.append(fig)
    _save(fig, f"{prefix}_partial_hr.png")

    fig = plt.figure()
    for r in runs:
        plt.plot(
            r.energy_mev_positive,
            r.specfun_energy,
            alpha=line_alpha,
            label=r.name,
        )
    plt.xlabel("Phonon Energy (meV)")
    plt.ylabel("S(E)")
    plt.title("Spectral function")
    plt.legend()
    figs.append(fig)
    _save(fig, f"{prefix}_specfun_SE.png")

    fig = plt.figure()
    for r in runs:
        plt.plot(r.t_fs, np.real(r.s_t), alpha=line_alpha, label=f"{r.name} Re[S(t)]")
        plt.plot(r.t_fs, np.imag(r.s_t), alpha=line_alpha, linestyle="--", label=f"{r.name} Im[S(t)]")
    plt.xlabel("Time (fs)")
    plt.ylabel("S(t)")
    plt.title("S(t)")
    plt.legend()
    figs.append(fig)
    _save(fig, f"{prefix}_S_of_t.png")

    fig = plt.figure()
    for r in runs:
        plt.plot(r.t_fs, np.real(r.g_t), alpha=line_alpha, label=f"{r.name} Re[G(t)]")
        plt.plot(r.t_fs, np.imag(r.g_t), alpha=line_alpha, linestyle="--", label=f"{r.name} Im[G(t)]")
    plt.xlabel("Time (fs)")
    plt.ylabel("G(t)")
    plt.title("G(t)")
    plt.legend()
    figs.append(fig)
    _save(fig, f"{prefix}_G_of_t.png")
    try:
        fig = plt.figure()
        for r in runs:
            plt.plot(r.energy_mev, r.a_e, alpha=line_alpha, label=r.name)
        plt.xlabel("Photon Energy (meV)")
        plt.ylabel("a(E)")
        plt.title("a(E)")
        plt.legend()
        figs.append(fig)
        _save(fig, f"{prefix}_a_of_E.png")
    except:
        print("plotting a_of_E did not work LOL")
        pass

    plt.figure()
    for r in runs:
        y = (np.abs(r.l_e) + eps_log)
        plt.plot(r.energy_mev, y, alpha=line_alpha, label=r.name)
    plt.yscale("log")
    plt.xlabel("Photon Energy (meV)")
    plt.ylabel("log(|PL|)")
    plt.title("PL spectrum (log amplitude)")
    plt.legend()
    figs.append(fig)
    _save(fig, f"{prefix}_PL_logabs.png")

    plt.figure()
    for r in runs:
        y = (np.abs(r.l_e) + eps_log)
        plt.plot(r.energy_mev, y, alpha=line_alpha, label=r.name)
    plt.xlabel("Photon Energy (meV)")
    plt.ylabel("linear |PL|)")
    plt.title("PL spectrum (linear amplitude)")
    plt.legend()
    figs.append(fig)
    _save(fig, f"{prefix}_PL_linear.png")

    fig = plt.figure()
    for r in runs:
        plt.scatter(
            r.energy_k,
            r.ipr,
            s=10,
            marker="s",
            alpha=scatter_alpha,
            label=r.name,
        )
    plt.xlabel("Phonon Energy (meV)")
    plt.ylabel("IPR")
    plt.title("Inverse Participation Ratio")
    plt.legend()
    figs.append(fig)
    _save(fig, f"{prefix}_ipr.png")

    if show:
        plt.show()

    return figs

from plumipy import calculate_spectrum
def get_plinf(roots, name = None):
    PL_inf = []
    for i, root in enumerate(roots):
        root = Path(root)
        raw = root.parent
        material_dir = raw.parent
        model_dir = material_dir.parent
        
        if root.stem == 'CBVN':
            model = 'DFT'
        else:
            model = model_dir.stem
        if name:
            model = name

        print(f'calculating spectrum for {model}')
        contcar_gs_ = Path(root / "CONTCAR_GS")
        band_ = Path(root / "band.yaml")
        outcar_gs_ = Path(root / "OUTCAR_GS")

        old = calculate_spectrum( 
            gs_structure_path=contcar_gs_,
            es_structure_path=contcar_es_t,
            phonon_band_path=band_,
            phonons_source="Phonopy",
            temperature=0,
            zpl=3339, # doesnt matter
            tmax=2000, # 
            gamma=10, # two broadenings 1 for the ZPL and one for the phonons. sigma in the 'spectral function' is defaulted
            # actually no leave sigma at 6. 
           forces= (outcar_es_t, outcar_gs_), 
            plot = True
        )
        new = (model, ) + old 
        PL_inf.append(new)
    return PL_inf


# plot_pl_comparison(res_dft, res_mace, save_dir="./compare_plots", prefix="dft_vs_mace", show=True)


contcar_gs_t = "/home/rnpla/projects/mlip_phonons/test/CBVN/CONTCAR_GS"
outcar_gs_t = "/home/rnpla/projects/mlip_phonons/test/CBVN/OUTCAR_GS"
band_t = "/home/rnpla/projects/mlip_phonons/test/CBVN/band.yaml"

contcar_es_t = "/home/rnpla/projects/mlip_phonons/test/CBVN/CONTCAR_ES"
outcar_es_t = "/home/rnpla/projects/mlip_phonons/test/CBVN/OUTCAR_ES"

# first entry in roots is the DFT one.
# roots is list of base dirs of the plumipy stats to compare. 
#roots = ["/home/rnpla/projects/mlip_phonons/test/CBVN", "/home/rnpla/projects/mlip_phonons/results/small-omat-0/cbvn/raw/plumipy/band.yaml"]
#root_roots = [roots] # put other combinations in. 

#contcar_gs_ = "/home/rnpla/projects/mlip_phonons/test/plumipy_inputs/CONTCAR_GS"
#outcar_gs_ = "/home/rnpla/projects/mlip_phonons/test/plumipy_inputs/OUTCAR_GS"
#band_ = "/home/rnpla/projects/mlip_phonons/test/plumipy_inputs/band.yaml"
#ontcar_es_ = "/home/rnpla/projects/mlip_phonons/test/CBVN/CONTCAR_ES"
#outcar_es_ = "/home/rnpla/projects/mlip_phonons/test/CBVN/OUTCAR_ES_0"

#DFT = get_plinf()

#PL_infs = []
#for root in root_roots:
#    t = get_plinf(root)
#    PL_infs.append(t)

#for PL_inf in PL_infs:
#    plot_pl_comparison(*PL_inf, save_dir="/home/rnpla/projects/mlip_phonons/test/plumipy_outputs")

from pathlib import Path

repo = Path("/home/rnpla/projects/mlip_phonons")
dft_root = repo / "test" / "CBVN"

dft_results = get_plinf([dft_root])  # list with 1 tuple


for model_dir in Path("/home/rnpla/projects/mlip_phonons/results").iterdir():
    plum_dir = model_dir / "cbvn" / "raw" / "Plumipy_Files"
    if not plum_dir.is_dir():
        continue

    model_results = get_plinf([plum_dir], name = model_dir.stem)
    out_dir = model_dir / "cbvn" / "plot" / "Plumipy_plots"

    plot_pl_comparison(
        *(dft_results + model_results),
        save_dir=out_dir,
        prefix=f"dft_vs_{model_dir.name}",
        show=False,
    )

#MLIP: 
# /home/rnpla/projects/mlip_phonons/results/small-omat-0/cbvn/raw/Plumipy_Files/band.yaml
# /home/rnpla/projects/mlip_phonons/results/small-omat-0/cbvn/raw/Plumipy_Files/CONTCAR_GS
# /home/rnpla/projects/mlip_phonons/results/small-omat-0/cbvn/raw/Plumipy_Files/OUTCAR_GS
#DFT:
# /home/rnpla/projects/mlip_phonons/test/CBVN/CONTCAR_GS
# /home/rnpla/projects/mlip_phonons/test/CBVN/OUTCAR_GS
# /home/rnpla/projects/mlip_phonons/test/CBVN/band.yaml


# Run from notebook root directory.
# cd main/test_notebooks

