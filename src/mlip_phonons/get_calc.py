from __future__ import annotations
import __main__
from pathlib import Path
from typing import Any




"""
Help(load_model):
load_model(path: 'str | Path', **kwargs)
    Convenience method to load a model from a directory or name.

    Args:
        path (str|path): Path to saved model or name of pre-trained model. The search order is path, followed by
            download from PRETRAINED_MODELS_BASE_URL (with caching).
        **kwargs: Additional kwargs passed to RemoteFile class. E.g., a useful one might be force_download if you
            want to update the model.

    Returns:
        Returns: model_object if include_json is false. (model_object, dict) if include_json is True.
"""
#get_calc_object("CHGNet-MatPES-PBE-2025.2.10-2.7M-PES")





from pathlib import Path
from typing import Callable, Any, Dict

def _mace_builder(model_rel: str) -> Callable[[Path, str, str], Any]:
    def _build(models_root: Path, device: str, dtype: str) -> Any:
        from mace.calculators import MACECalculator
        model_path = models_root / "mace" / model_rel
        return MACECalculator(model_path=str(model_path), device=device, default_dtype=dtype)
    return _build

def _mattersim_builder(model_rel: str) -> Callable[[Path, str, str], Any]:
    def _build(models_root: Path, device: str, dtype: str) -> Any:
        from mattersim.forcefield import MatterSimCalculator
        model_path = models_root / "mattersim" / model_rel
        return MatterSimCalculator(from_checkpoint=str(model_path), device=device)
    return _build

def _matgl_builder(model_dir: str) -> Callable[[Path, str, str], Any]:
    def _build(models_root: Path, device: str, dtype: str) -> Any:
        import os
        os.environ["MATGL_BACKEND"] = "DGL"
        import matgl
        matgl.set_backend("DGL")
        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_path = models_root / "matgl" / "pretrained_models" / model_dir
        pot = load_model(model_path, force_download=False)
        pot.cuda()
        return PESCalculator(pot)
    return _build

def _pet_builder(model_rel: str) -> Callable[[Path, str, str], Any]:
    def _build(models_root: Path, device: str, dtype: str) -> Any:
        from metatomic.torch.ase_calculator import MetatomicCalculator
        model_path = models_root / "petmad" / "upet" / model_rel
        return MetatomicCalculator(str(model_path), device=device, non_conservative=True)
    return _build

def _orb_builder(model_rel: str, precision: str) -> Callable[[Path, str, str], Any]:
    def _build(models_root: Path, device: str, dtype: str) -> Any:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        model_path = models_root / "orb" / model_rel
        orbff = pretrained.orb_v3_direct_inf_omat(weights_path=str(model_path), device=device)
        return ORBCalculator(orbff, device=device)
    return _build

MODEL_BUILDERS: Dict[str, Callable[[Path, str, str], Any]] = {
    "small-omat-0": _mace_builder("mace-omat-0-small.model"),
    "mace-omat-0-medium": _mace_builder("mace-omat-0-medium.model"),
    "mace-mpa-0-medium": _mace_builder("mace-mpa-0-medium.model"),
    "mace-matpes-pbe-omat-ft": _mace_builder("MACE-matpes-pbe-omat-ft.model"),
    "mace-matpes-r2scan-omat-ft": _mace_builder("MACE-matpes-r2scan-omat-ft.model"),

    "mattersim-v1.0.0-1M": _mattersim_builder("mattersim-v1.0.0-1M.pth"),
    "mattersim-v1.0.0-5M": _mattersim_builder("mattersim-v1.0.0-5M.pth"),

    "orb-v3-direct-omat": _orb_builder("orb-v3-direct-inf-omat.ckpt", precision="float32-high"),
    "orb-d3-sm-v2": _orb_builder("orb-d3-sm-v2.ckpt", precision="float64"),
    "orb-v3-conservative-inf-omat": _orb_builder("orb-v3-conservative-inf-omat.ckpt", precision="float64"),

    "pet-mad-s-v1.1.0": _pet_builder("pet-mad-s-v1.1.0.pt"),
    "pet-omad-l-v0.1.0": _pet_builder("pet-omad-l-v0.1.0.pt"),
    "pet-omad-s-v1.0.0": _pet_builder("pet-omad-s-v1.0.0.pt"),
    "pet-omat-l-v1.0.0": _pet_builder("pet-omat-l-v1.0.0.pt"),
    "pet-omat-m-v1.0.0": _pet_builder("pet-omat-m-v1.0.0.pt"),
    "pet-omat-xl-v1.0.0": _pet_builder("pet-omat-xl-v1.0.0.pt"),

    "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES": _matgl_builder("CHGNet-MatPES-PBE-2025.2.10-2.7M-PES"),
    "CHGNet-MPtrj-2023.12.1-2.7M-PES": _matgl_builder("CHGNet-MPtrj-2023.12.1-2.7M-PES"),
    "CHGNet-MPtrj-2024.2.13-11M-PES": _matgl_builder("CHGNet-MPtrj-2024.2.13-11M-PES"),
    "M3GNet-ANI-1x-Subset-PES": _matgl_builder("M3GNet-ANI-1x-Subset-PES"),
    "M3GNet-MatPES-PBE-v2025.1-PES": _matgl_builder("M3GNet-MatPES-PBE-v2025.1-PES"),
    "M3GNet-MatPES-r2SCAN-v2025.1-PES": _matgl_builder("M3GNet-MatPES-r2SCAN-v2025.1-PES"),
    "M3GNet-MP-2021.2.8-DIRECT-PES": _matgl_builder("M3GNet-MP-2021.2.8-DIRECT-PES"),
    "M3GNet-MP-2021.2.8-PES": _matgl_builder("M3GNet-MP-2021.2.8-PES"),
    "TensorNetDGL-MatPES-PBE-v2025.1-PES": _matgl_builder("TensorNetDGL-MatPES-PBE-v2025.1-PES"),
    "TensorNetDGL-MatPES-r2SCAN-v2025.1-PES": _matgl_builder("TensorNetDGL-MatPES-r2SCAN-v2025.1-PES"),
}

def get_calc_object(model: str, models_root: str | Path | None = None,
                    device: str = "cuda", dtype: str = "float32") -> Any:
    models_root = Path(models_root) if models_root is not None else Path("assets") / "models"
    try:
        return MODEL_BUILDERS[model](models_root, device, dtype)
    except KeyError:
        raise ValueError(f"Unknown model '{model}'")


mace_env = [
    "small-omat-0",
    "mace-omat-0-medium",
    "mace-mpa-0-medium",
    "mace-matpes-pbe-omat-ft",
    "mace-matpes-r2scan-omat-ft",
    "orb-v3-direct-omat",
    "orb-d3-sm-v2",
    "orb-v3-conservative-inf-omat",
    "pet-mad-s-v1.1.0",
    "pet-omad-l-v0.1.0",
    "pet-omad-s-v1.0.0",
    "pet-omat-l-v1.0.0",
    "pet-omat-m-v1.0.0",
    "pet-omat-xl-v1.0.0",
]

matgl_env = [
    "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES",
    "CHGNet-MPtrj-2023.12.1-2.7M-PES",
    "CHGNet-MPtrj-2024.2.13-11M-PES",
    "M3GNet-ANI-1x-Subset-PES",
    "M3GNet-MatPES-PBE-v2025.1-PES",
    "M3GNet-MatPES-r2SCAN-v2025.1-PES",
    "M3GNet-MP-2021.2.8-DIRECT-PES",
    "M3GNet-MP-2021.2.8-PES",
    "TensorNetDGL-MatPES-PBE-v2025.1-PES",
    "TensorNetDGL-MatPES-r2SCAN-v2025.1-PES",
]

mattersim_env = [
    "mattersim-v1.0.0-1M",
    "mattersim-v1.0.0-5M",
]

        
    
if __name__ == "__main__":
    for model in matgl_env: #mattersim_env: #mace_env: #matgl_env: #mattersim_env: #matgl_env: #mace_env: #+ matgl_env + mattersim_env:
        try:
            calc = get_calc_object2(model)
            print(type(calc))
            print(f"Successfully loaded calculator for model: {model}")
            #if hasattr(calc, "element_types"):
            #    print("model element_types:", calc.element_types)
            if hasattr(calc, "model") and hasattr(calc.model, "element_types"):
                print("model element_types:", calc.model.element_types)  
        except Exception as e:
            print(f"Failed to load calculator for model: {model}. Error: {e}")



# Old version. havnt fully tested the implementtion  of the new one 
#( should be fine) but wanted to keep this just in case i stuffed up smth.

"""
def get_calc_object2(model: str, models_root: str | Path | None = None) -> Any:
    #Construct a calculator by model name using local model paths.

    #Args:
    #    model (str): Model name key corresponding to an on-disk checkpoint.
    #    models_root (str | Path | None): folder containing the folders containing the models (default: ./assets/models).

    #Returns:
    #    Any: ASE-compatible calculator instance for the requested model.
    
    models_root = Path(models_root) if models_root is not None else Path("assets") / "models"

    calc: Any | None = None

    if model == "small-omat-0":
        from mace.calculators import MACECalculator
        model_path = models_root / "mace" / "mace-omat-0-small.model"
        calc = MACECalculator(model_path=str(model_path), device="cuda", default_dtype="float32")
        print(f"using {model} calculator")

    elif model == "mace-omat-0-medium":
        from mace.calculators import MACECalculator
        model_path = models_root / "mace" / "mace-omat-0-medium.model"
        calc = MACECalculator(
            model_path=str(model_path),
            device="cuda",
            default_dtype="float32",
        )
        print(f"using {model} calculator")

    elif model == "mace-mpa-0-medium":
        from mace.calculators import MACECalculator
        model_path = models_root / "mace" / "mace-mpa-0-medium.model"
        calc = MACECalculator(
            model_path=str(model_path),
            device="cuda",
            default_dtype="float32",
        )
        print(f"using {model} calculator")

    elif model == "mace-matpes-pbe-omat-ft":
        from mace.calculators import MACECalculator
        model_path = models_root / "mace" / "MACE-matpes-pbe-omat-ft.model"
        calc = MACECalculator(
            model_path=str(model_path),
            device="cuda",
            default_dtype="float32",
        )
        print(f"using {model} calculator")

    elif model == "mace-matpes-r2scan-omat-ft":
        from mace.calculators import MACECalculator
        model_path = models_root / "mace" / "MACE-matpes-r2scan-omat-ft.model"
        calc = MACECalculator(
            model_path=str(model_path),
            device="cuda",
            default_dtype="float32",
        )
        print(f"using {model} calculator")

    elif model == "mattersim-v1.0.0-1M":
        from mattersim.forcefield import MatterSimCalculator
        checkpoint_path = models_root / "mattersim" / "mattersim-v1.0.0-1M.pth"
        calc = MatterSimCalculator(from_checkpoint=str(checkpoint_path), device="cuda")
        print(f"using {model} calculator")

    elif model == "mattersim-v1.0.0-5M":
        from mattersim.forcefield import MatterSimCalculator
        checkpoint_path = models_root / "mattersim" / "mattersim-v1.0.0-5M.pth"
        calc = MatterSimCalculator(from_checkpoint=str(checkpoint_path), device="cuda")
        print(f"using {model} calculator")

    elif model == "orb-v3-direct-omat":  # ORB models are not working.
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        model_path_ = models_root / "orb" / "orb-v3-direct-inf-omat.ckpt"
        device_ = "cuda"
        precision_ = "float32-high" # or "float32-highest" / "float64"

        orbff = pretrained.orb_v3_direct_inf_omat(
            weights_path=str(model_path_),
            device=device_,
            #precision=precision_,
        )
        calc = ORBCalculator(orbff, device=device_)
        print(f"using {model} calculator")

    elif model == "orb-d3-sm-v2":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        model_path_ = models_root / "orb" / "orb-d3-sm-v2.ckpt"
        device_ = "cuda"
        precision_ = "float64" # or "float32-highest" / "float64"

        orbff = pretrained.orb_d3_sm_v2(
            weights_path=str(model_path_),
            device=device_,
            #precision=precision_,
        )
        calc = ORBCalculator(orbff, device=device_)
        print(f"using {model} calculator")

    elif model == "orb-v3-conservative-inf-omat":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        model_path_ = models_root / "orb" / "orb-v3-conservative-inf-omat.ckpt"
        device_ = "cuda"
        precision_ = "float64" # or "float32-highest" / "float64"

        orbff = pretrained.orb_v3_conservative_inf_omat(
            weights_path=str(model_path_),
            device=device_,
            precision=precision_,
        )
        calc = ORBCalculator(orbff, device=device_)
        print(f"using {model} calculator")

    elif model == "pet-mad-s-v1.1.0":
        
        #petmad requires downloading files from their initialisation and moving them out
        #of the cache directory (do not moove the hugging face files those are symmlinks to the
        #upet cache), or the use of the the upet.save_upet() function.

        #models can then be loaded using metatomic.torch.ase_calculator and the path to the file.
        #the file should be a torch script file (.pt), or a .chpt file, a sanity check is whether the
        #file is in bytes.
        
        checkpoint_path_ = models_root / "petmad" / "upet" / "pet-mad-s-v1.1.0.pt"
        from metatomic.torch.ase_calculator import MetatomicCalculator
        calc = MetatomicCalculator(str(checkpoint_path_), device="cuda", non_conservative=True)
        print(f"using {model} calculator")

    elif model == "pet-omad-l-v0.1.0":
        checkpoint_path_ = models_root / "petmad" / "upet" / "pet-omad-l-v0.1.0.pt"
        from metatomic.torch.ase_calculator import MetatomicCalculator
        calc = MetatomicCalculator(str(checkpoint_path_), device="cuda", non_conservative=True)
        print(f"using {model} calculator")

    elif model == "pet-omad-s-v1.0.0":
        checkpoint_path_ = models_root / "petmad" / "upet" / "pet-omad-s-v1.0.0.pt"
        from metatomic.torch.ase_calculator import MetatomicCalculator
        calc = MetatomicCalculator(str(checkpoint_path_), device="cuda", non_conservative=True)
        print(f"using {model} calculator")

    elif model == "pet-omat-l-v1.0.0":
        checkpoint_path_ = models_root / "petmad" / "upet" / "pet-omat-l-v1.0.0.pt"
        from metatomic.torch.ase_calculator import MetatomicCalculator
        calc = MetatomicCalculator(str(checkpoint_path_), device="cuda", non_conservative=True)
        print(f"using {model} calculator")

    elif model == "pet-omat-m-v1.0.0":
        checkpoint_path_ = models_root / "petmad" / "upet" / "pet-omat-m-v1.0.0.pt"
        from metatomic.torch.ase_calculator import MetatomicCalculator
        calc = MetatomicCalculator(str(checkpoint_path_), device="cuda", non_conservative=True)
        print(f"using {model} calculator")

    elif model == "pet-omat-xl-v1.0.0":
        checkpoint_path_ = models_root / "petmad" / "upet" / "pet-omat-xl-v1.0.0.pt"
        from metatomic.torch.ase_calculator import MetatomicCalculator
        calc = MetatomicCalculator(str(checkpoint_path_), device="cuda", non_conservative=True)
        print(f"using {model} calculator")

    ###########################

    elif model == "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES":
        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")

        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_dir = models_root / "matgl" / "pretrained_models" / "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES"
        pot = load_model(model_dir, force_download=False)
        print(type(pot))
        pot.cuda()
        calc = PESCalculator(pot)
        print(type(calc))
        print(f"using {model} calculator")

    elif model == "CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES":
        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")

        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_dir = models_root / "matgl" / "pretrained_models" / "CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES"
        pot = load_model(model_dir, force_download=False)
        pot.cuda()
        calc = PESCalculator(pot)
        print(f"using {model} calculator")

    elif model == "CHGNet-MPtrj-2023.12.1-2.7M-PES":
        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")

        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_dir = models_root / "matgl" / "pretrained_models" / "CHGNet-MPtrj-2023.12.1-2.7M-PES"
        pot = load_model(model_dir, force_download=False)
        pot.cuda()
        calc = PESCalculator(pot)
        print(f"using {model} calculator")

    elif model == "CHGNet-MPtrj-2024.2.13-11M-PES":
        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")

        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_dir = models_root / "matgl" / "pretrained_models" / "CHGNet-MPtrj-2024.2.13-11M-PES"
        pot = load_model(model_dir, force_download=False)
        pot.cuda()
        calc = PESCalculator(pot)
        print(f"using {model} calculator")

    elif model == "M3GNet-ANI-1x-Subset-PES":
        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")
        
        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator


        model_dir = models_root / "matgl" / "pretrained_models" / "M3GNet-ANI-1x-Subset-PES"
        pot = load_model(model_dir, force_download=False)


        pot.cuda()

        calc = PESCalculator(pot)
        print(f"using {model} calculator")

    elif model == "M3GNet-MatPES-PBE-v2025.1-PES":

        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")

        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_dir = models_root / "matgl" / "pretrained_models" / "M3GNet-ANI-1x-Subset-PES"
        pot = load_model(model_dir, force_download=False)
        pot.cuda()
        calc = PESCalculator(pot)
        print(f"using {model} calculator")

    elif model == "M3GNet-MatPES-r2SCAN-v2025.1-PES":

        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")

        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_dir = models_root / "matgl" / "pretrained_models" / "M3GNet-MatPES-r2SCAN-v2025.1-PES"
        pot = load_model(model_dir, force_download=False)
        pot.cuda()
        calc = PESCalculator(pot)
        print(f"using {model} calculator")

    elif model == "M3GNet-MP-2021.2.8-DIRECT-PES":

        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")

        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_dir = models_root / "matgl" / "pretrained_models" / "M3GNet-MP-2021.2.8-DIRECT-PES"
        pot = load_model(model_dir, force_download=False)
        pot.cuda()
        calc = PESCalculator(pot)
        print(f"using {model} calculator")

    elif model == "M3GNet-MP-2021.2.8-PES":

        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")

        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_dir = models_root / "matgl" / "pretrained_models" / "M3GNet-MP-2021.2.8-PES"
        pot = load_model(model_dir, force_download=False)
        print(type(pot))
        pot.cuda()
        calc = PESCalculator(pot)
        print(f"using {model} calculator")

    elif model == "TensorNetDGL-MatPES-PBE-v2025.1-PES":

        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")

        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_dir = models_root / "matgl" / "pretrained_models" / "TensorNetDGL-MatPES-PBE-v2025.1-PES"
        pot = load_model(model_dir, force_download=False)
        pot.cuda()
        calc = PESCalculator(pot)
        print(f"using {model} calculator")
    
    elif model == "TensorNetDGL-MatPES-r2SCAN-v2025.1-PES":
        
        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")

        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_dir = models_root / "matgl" / "pretrained_models" / "TensorNetDGL-MatPES-r2SCAN-v2025.1-PES"
        pot = load_model(model_dir, force_download=False)
        pot.cuda()
        calc = PESCalculator(pot)
        print(f"using {model} calculator")

    if calc:
        return calc
    raise ValueError(f"Unknown model '{model}'")

"""