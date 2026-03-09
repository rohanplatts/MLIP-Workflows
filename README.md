# mlip-phonons

This package is for benchmarking a variety of MLIPs wtih respect to their obtained Phonons, DOS, band structures, Pl/vibronic calculations, Coupling modes (of the calculated phonons) and MEP via NEB. Comparisons with DFT are especially used. 

Models that it currently compares are the Orb-materials, MACE, Mattersim, Upet, and matgl model families.

## Installation

### Important: D3 (vdW) Dependencies For NEB

`mlip-neb` can optionally add a D3 (vdW) correction during the `mlip_d3` refinement stage (and `config.yml` defaults `neb.defaults.include_vdw: true`).

This D3 backend comes from the `dftd3-python` + `simple-dftd3` packages and should be installed from **conda-forge**. Pip-only installs are not reliable here (these packages are not always available on pip, and builds without OpenMP can be extremely slow on HPC). If you run `mlip-neb` with vdW enabled but do not have these installed, the D3 stage will be unusably slow or fail.

Recommended install flow for any environment that will run NEB with vdW:

```bash
conda env create -f env/mace_env.yml
conda activate mace_env
python -m pip install -e .
```
Mace_env is the conda environment suited to run this package for mace, orb and petmad models. If you would like to use other models, use the other .yml files supplied in `./env/`. 

This allows vast acceleration of the D3 calculations: Before you run the mlip-neb pipeline, you set: 

```bash
export LD_PRELOAD="${CONDA_PREFIX}/lib/libgomp.so.1" 

export OMP_NUM_THREADS= <NUMBER OF CORES>
export MKL_NUM_THREADS= <NUMBER OF CORES>
export OPENBLAS_NUM_THREADS= <NUMBER OF CORES>
```
i have found that for <NUMBER OF CORES> to be 12 or 16 that D3 works marvellously, yet i have not tested for much more than that. 

If you do not need vdW/D3, you can disable it for NEB runs (or disable it in config.yml):

```bash
mlip-neb --no-include-vdw
```

### Standard installation

Clone the repository, create a Python environment, activate it, then install from the repo root.

For most users (especially if you plan to run `mlip-neb` with vdW/D3), the recommended route is to use the provided Conda environment YAML so DFTD3 is installed from conda-forge:

```bash
cd /scratch/user/$USER/mlip_phonons
git clone https://github.com/rohanplatts/phonons-mlip.git
cd phonons-mlip

conda env create -f env/mace_env.yml  # mace example. 
conda activate mace_env
python -m pip install -e .
```

Install additional model backends (if you are not using the provided env YAMLs), the options are:

```bash
python -m pip install -e '.[mace]'
python -m pip install -e '.[mattersim]'
python -m pip install -e '.[matgl]'
python -m pip install -e '.[pet]'
python -m pip install -e '.[orb]'
```

If you are planning on using several families of models, i would recomend using conda, it will save you pain. I would also recommend not using the editable install to obtain such a loadout, and instead create and install all of the necessary environments using `/env/ENVIRONMENTS.md`. After you finish with this, then for each environment you can install the editable for this package, and you will have CLI access but with greater freedom across all currently covered model families.
 
Notes:

* Python requirement: `>=3.10` (see `pyproject.toml`).
* Install only the extras you actually need.
* If an editable install with extras fails, see `env/ENVIRONMENTS.md`.
* A backup environment file is available at `env/mace_env.yml`.
* `src/mlip_phonons/main.py` currently assumes CUDA is available (`torch.cuda.is_available()` must be `True`). This should be relaxed in future.


---

### HPC installation

On HPC systems, do the installation on a **compute node**, not on the login node. Specifically, this is a CUDA-dependent workflow, so use a **GPU node with CUDA support**.

#### 1. Request an interactive compute node

Example Slurm request for a CUDA-capable GPU node:

```bash
salloc \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=16G \
  --job-name=environment_setup \
  --time=01:00:00 \
  --gres=gpu:1 \
  --partition=gpu_cuda \
  --qos=gpu \
  --account=ACCOUNT_STR \
  srun --export=PATH,TERM,HOME,LANG --pty /bin/bash -l
```

Notes:

* Replace `ACCOUNT_STR` with your Slurm account string.
* The exact partition, QoS, and GPU resource syntax may vary between clusters.
* An A100 is a tad overkill, was just the first example that came to mind. 

---

#### 2. Choose an environment manager

`miniforge` is recommended on HPC. 

##### Option A: Miniforge / Conda

Check available modules:

```bash
module avail anaconda miniconda miniforge
```

Load Miniforge (the below bash is something that you will use daily for miniforge usage):

```bash
module load miniforge/25.3.0-3
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"
```

Create a working directory in scratch, then clone the repository:

```bash
mkdir -p /scratch/user/$USER/mlip_phonons
cd /scratch/user/$USER/mlip_phonons

git clone https://github.com/rohanplatts/phonons-mlip.git

```

---

#### 3. Create the environment from the .yml in env/

Create and activate an environment:

```bash
cd phonons-mlip
conda env create -f env/mace_env.yml
conda activate mace_env
```

---

#### 4. Install the package

From the repository root:

```bash
python -m pip install -e .
```
Above we are installing the package into the active environment. 

If you plan to run NEB with `include_vdw: true`, prefer using `conda env create -f env/mace_env.yml` (or other env YAMLs) so `dftd3-python` and `simple-dftd3` come from conda-forge. (Pip does not have them)

---

#### 6. Daily environment activation on HPC

If you are using Miniforge / Conda, a typical daily setup, with d3 acceleration (via parallelising over cores):

```bash
module purge
module load miniforge/25.3.0-3
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"
conda activate mlip_phonons

export LD_PRELOAD="${CONDA_PREFIX}/lib/libgomp.so.1" 

export OMP_NUM_THREADS= <NUMBER OF CORES>
export MKL_NUM_THREADS= <NUMBER OF CORES>
export OPENBLAS_NUM_THREADS= <NUMBER OF CORES>
```

---

#### 5. CHECKS

Check that the package imports:

```bash
python -c "import mlip_phonons; print('mlip_phonons import successful')"
```

Check that PyTorch can see the GPU:

```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PY
```

---

## Configure
Next step is to configure this such that it works for you. First you need to obtain the model file, for `mace-mpa-0-medium` i have suppied this as an example in `assets/models/mace/mace-mpa-0-medium.model`, as is the example fine tuned model: `assets/models/mace/ivac0_neb_ft.model`. Any models that you would like to test should be copied into `assets/models/<model_family>/<your_model_weight_file>`. The currently supported models are listed in config.yml in the 'models' dictionary. 

Generally, if you wanted to add a model, then you will first have to resolve a way to obtain the ASE calculator object from that model file, and implement that workflow into `src/mlip_phonons/get_calc.py`. You will find though, that if this model is of the model families accomodataed by this script, it is very likely all you will have to do is:

* copy the model file into the correct family subfolder in `assets/models`, 
* then add a new dictionary item for that specific model in `src/mlip_phonons/get_calc.model_build`
* the key for this dictionary item is simply the name of the model, as for the value, you will be able to identify that easily when you are appending this.

Next, it is reccomended to edit config.yml and add the supported model to your list. Note that there is a structure key to be added under the model. This structure key is important when it comes to model sweeps of Phonon workflows, as in those cases, the config.yml file is read and each model is fed its own structure. If it does not appear to you that this would be useful (perhaps you are doing only NEB workflows) you can set this key to None.

Edit `config.yml`:
- `models`: model names (must match the keys supported by `src/mlip_phonons/get_calc.py`) and which structure/material they run on.
- `structures`: where structures live (`assets/structures/...`) plus phonon settings (supercell, displacement `delta`, DOS mesh, etc.).

See `config.yml` for the expected structure of inputs.

Model weights/checkpoints are expected under `assets/models/` 

## NEB Quickstart (poscar_i/poscar_f + MLIP)

Suppose you want the MEP from NEB between two endpoints with the MLIP 'mace-mpa-0-medium'.

1. Ensure you have mace_env set up (see above)
2. Obtain the 'mace-mpa-0-medium' model file (I HAVE ALREADY INCLUDED THIS FILE FOR CONVENIENCE...)
3. Put the model file in `assets/models/<model_family>/<model_file>` (I HAVE ALREADY DONE THIS...)
4. Prepare endpoints: `POSCAR_i` and `POSCAR_f` (or pass paths via flags). (AS AN EXAMPLE, I HAVE PUT SAMPLE POSCAR_i and POSCAR_f in `src/NEB` )
5. Check if the model file is supported by reading config.yml (IT IS), and if it isnt, add its calculator object to src/mlip_phonons/get_calc.py
6. Run:

```bash
mlip-neb "mace-mpa-0-medium" --poscar-i path/to/POSCAR_i --poscar-f path/to/POSCAR_f
```

Useful overrides:
- `--results-root`: where outputs go (default `resultsNEB/`).
- `--models-root`: where model weights live (default `assets/models/`).
- `--dft-neb-dat`: optional DFT reference `neb.dat` for comparisons.
- `--n-images`: number of images (otherwise inferred or defaulted).

Now if you want this to be even more efficient, you can prepare your `mlip-neb` command in config.yml by editting NEB defaults. Say i had the path to POSCAR_i, and path to POSCAR_f, say i wanted the results to be located in some obscure folder, that i wanted van-der-waals term correction on, that i wanted the endpoints to be relaxed by the MLIP, and that i want within-species remapping of the initial and final poscars, and that i wanted the final supplied MLIP MEP to be vasp loadable. then in config.yml, i would change NEB to: 

```text
neb:
  defaults:
    model_name: mace-mpa-0-medium # the model
    results_root: /some/obscure/folder/resultsNEB # where you want the results
    models_root: assets/models # where to look for the model 
    structures_dir: None # this is only really for a backup place to look for poscar_i or poscar_f, or neb.dat when trying to compare with DFT 

    poscar_i: /path/to/POSCAR_i
    poscar_f: /path/to/POSCAR_f

    dft_neb_dat: null # the neb.dat path when wanting to compare model performance with DFT predictions.
    vasp_inputs_dir: /path/to/vasp_inputs 
    relax_endpoints: true 
    remap_f_i: true
    include_vdw: true
    overwrite: false
    device: cuda
    dtype: float32
  settings:
    n_images_fallback: 9
    maxstep_mlip_guess: 0.05 
    fmax_mlip_guess: 0.03
    steps_mlip_guess: 3000
    k_spring_mlip: 0.6
    k_spring: 0.6
    maxstep_mlip_d3: 0.03
    fmax_mlip_d3: 0.03
    steps_mlip_d3: 1400
    maxstep_ci: 0.03
    fmax_ci: 0.03
    steps_ci: 1000
```

Then, all you will have to do is type into command line: 
```
mlip-neb
```
And it shall run according to your settings.

For more NEB details (including comparison and VASP export), see `src/NEB/README.md`.

## Run Phonon Workflow

The main entry point is the console script (defaults read from `config.yml` under `mlip_phonons.defaults`):

If you would like to obtain the phonons using a particular model, on a particular structure, then you may run either of the following:

```bash
mlip-phonons
mlip-phonons <model_name>
mlip-phonons --config config.yml
mlip-phonons <model_name> --structure <structure_key>
```
Structure_key is read from config.yml, under `structures`. See config.yml for setting this up for a given structure you have. 


For more details on phonon calculations see `src/mlip_phonons/README.md`
## Outputs

By default outputs are written under:

```text
results/<model_name>/<structure_key>/
  raw/
  plot/
```

Additionally, Plumipy-ready (a package for obtaining PL spectra) files are written under:

```text
results/<model_name>/<structure_key>/raw/Plumipy_Files/
```

Note: there are some TODOs in this repo about splitting subprojects (e.g., coupling, plumipy helpers).

## DFT vs ML Coupling/Ranking 

`src/coupling_modes/phonon_coupling.py` compares a DFT `band.yaml` against one or more MLIP `band.yaml` files using:
- a GS→ES displacement vector (from `CONTCAR_GS`/`CONTCAR_ES`) for projection weights,
- mode matching via overlap + Hungarian assignment,
- weighted errors `E_freq`, `E_vec`, and combined `Score = E_freq + alpha * E_vec`,
- frequency-cluster window diagnostics.

Run:

```bash
python src/coupling_modes/phonon_coupling.py --alpha 0.5 --weight_kind S
```

Reports are printed and also saved to `resultsPhonCoupling/phonon_coupling_report_<i>.txt`.

## PL Plot Comparison 

`src/plumipy_run/exploratory_script.py` contains helpers to call `plumipy.calculate_spectrum` and generate comparison plots (partial Huang–Rhys, spectral function, PL, IPR). Requires plumipy package installed.

## What Benchmarking Is Actually Done

The repository does two main kinds of implemented MLIP-vs-DFT benchmarking, plus one plotting-oriented comparison workflow:

1. **NEB / MEP benchmarking**
   `src/NEB/NEB_compare_all.py` (accessible via command line as `mlip-neb --compare`) compares each MLIP CI-NEB path against a DFT `neb.dat`.

   It reports:
   - `mlip_barrier_eV` and `mlip_deltaE_eV`
   - `dft_barrier_eV` and `dft_deltaE_eV`
   - `barrier_abs_err_eV` and `deltaE_abs_err_eV`
   - optional force errors on DFT geometries: `force_RMSE_eV_per_A`, `max_force_err_eV_per_A`
   - optional path quality on the final MLIP path: `max_F_perp_eV_per_A`
   - timing metrics parsed from ASE optimizer logs

   The final NEB ranking is sorted by lower `barrier_abs_err_eV`, then lower `deltaE_abs_err_eV`. Outputs are written under `resultsNEB/<model>/plot/` plus a combined `resultsNEB/rankings/rankings.txt`.

2. **Phonon-coupling / mode-matching benchmarking**
   `src/coupling_modes/phonon_coupling.py` compares MLIP `band.yaml` files against a DFT `band.yaml`, using the GS->ES displacement from `CONTCAR_GS` and `CONTCAR_ES` to decide which phonon modes matter most.

   It reports:
   - `E_freq`: weighted frequency error after DFT-to-ML mode matching
   - `E_vec`: weighted eigenvector mismatch
   - `Score = E_freq + alpha * E_vec`
   - `E_freq_rel`
   - `X_mean` for coupling-subspace agreement
   - cluster-window metrics for near-degenerate eigenspaces

   a more comprehensive description on what is computed WRT phonon-coupling is included in the src/phonon_coupling folder. 

   The final phonon-coupling ranking is sorted by lower `Score_mean`. Reports are written to `resultsPhonCoupling/phonon_coupling_report_<i>.txt`.

3. **PL / vibronic comparison**
   `src/plumipy_run/exploratory_script.py` uses the MLIP phonon outputs to generate comparison plots such as Huang-Rhys contributions, spectral functions, PL spectra, and IPR-style views. This is useful for side-by-side benchmarking, but it is currently a plotting workflow rather than a single scalar ranking.

So, in short: `mlip-phonons` produces the phonon data, `NEB_compare_all.py` benchmarks MEPs against DFT, and `phonon_coupling.py` benchmarks the displacement-relevant phonon modes against DFT.
