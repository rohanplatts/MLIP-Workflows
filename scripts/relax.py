config = { # we assume that they provide the supercell to be analysed,
    'calculators': ['chgnet', 'pet-mad', 'orb', 'small-omat-0'], # the list of calculators to perform calculations on. #TODO: supply a full comprehensive list of every single calculator that is possible for calculations (like including various mace models as an example), this is for the user, then let the user comment out the calculators they dont want. 
    'diamond': { 
        'filepath': r'/home/rnpla/projects/mlip_phonons/structures/diamond_super.poscar',
        'relaxed': {
            'is_file_relaxed': False, # if the input file is not relaxed supply following config, if not, leave as None
            'fmax': 0.01, # maximum intermolecular force for relaxed structure
        },
        'delta': 0.01 # how far (in Å) each atom is nudged when ASE estimates second derivatives of the potential.
    },
    'nv_diamond': {
        'filepath': r'/home/rnpla/projects/mlip_phonons/structures/diamond_nv.poscar',
        'relaxed': {
            'is_file_relaxed': False, 
            'fmax': 0.01,
        },
        'delta': 0.01,
    }
    'hbn': {
        'filepath: r''
    }
    
        
    }
}



# ———————————— PARAMETERS
calculator = "chgnet" #"pet-mad" #"orb" #"small-omat-0" # from mace_mp
"""c
import ase
from ase.build import bulk

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

device="cpu" # or device="cuda"
# or choose another model using ORB_PRETRAINED_MODELS[model_name]()
orbff = pretrained.orb_v3_conservative_inf_omat(
  device=device,
  precision="float32-high",   # or "float32-highest" / "float64
)
calc = ORBCalculator(orbff, device=device)
atoms = bulk('Cu', 'fcc', a=3.58, cubic=True)

atoms.calc = calc
atoms.get_potential_energy()
"""

# ————————————————————————

def get_calc_object():
    if calculator == "small-omat-0":
        from mace.calculators import mace_mp
        calc = mace_mp(model=calculator, device="cuda", default_dtype="float64")
    if calculator == "mattersim":
        from mattersim.forcefield import MatterSimCalculator
        calc = MatterSimCalculator(device="cuda")
    if calculator == "orb": 
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        device="cuda"
        # or choose another model using ORB_PRETRAINED_MODELS[model_name]()
        orbff = pretrained.orb_v3_conservative_inf_omat(
        device=device,
        precision="float32-high",   # or "float32-highest" / "float64
        )
        calc = ORBCalculator(orbff, device=device)
        print("using Orb calculator")
    if calculator == "pet-mad":
        from upet.calculator import UPETCalculator
        calc = UPETCalculator(model="pet-mad-s", version="1.1.0", device="cpu", non_conservative=True)
        print("using pet-mad calculator")
    if calculator == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator
        calc = CHGNetCalculator()
        print("using chgnet calc")
    return calc

mlip_calc = get_calc_object()
supercell.calc = mlip_calc

opt = BFGS(supercell, trajectory='defect_relax.traj') # trajectory will store the path of the optimisation if you want 
# to store that.
print(type(opt))
opt.run(fmax=0.01)  # fmax=0.01 is a good standard for phonons
print("Relaxation complete")