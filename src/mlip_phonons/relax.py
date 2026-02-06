from pathlib import Path
from ase import Atoms
from ase.optimize import BFGS

#TODO: if file was saved, append to the "Relaxation complete" string "{filename}" was saved to <directory>
def relax(structure: Atoms, 
          fmax: float = 0.01, 
          outdir: Path | None = None, 
          filename: str | None = None):
    """Relax an ASE Atoms structure with BFGS.

    Args:
        structure (Atoms): ASE Atoms object to relax (modified in place).
        fmax (float): Maximum force criterion for convergence.
        outdir (Path | None): Directory to write a trajectory file into.
        filename (str | None): Trajectory filename to use when outdir is provided.

    Returns:
        Atoms: Relaxed atoms object.
    """
    saved_outdir = None

    if outdir is not None:
        if filename is not None:
            outdir = Path(outdir)
            outdir.mkdir(parents = True, exist_ok = True)

            name = str(filename)
            trajectory = outdir / name
            saved_outdir = outdir

            opt = BFGS(structure, trajectory=str(trajectory))
        else:
            pass
    else: 
        opt = BFGS(structure)

    opt.run(fmax=fmax)

    message = "Relaxation complete."
    if saved_outdir is not None:
        message += f' "{filename}" was saved to {saved_outdir}'
    print(message)

    return structure
