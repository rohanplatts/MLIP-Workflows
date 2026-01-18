
test_mace i on wsl i simply installed with:

conda create --name test_mace python=3.10 pip
pip install mace-torch
conda install ipykernel

let pip and conda handle all dependencies.


mattersim_env: 
conda create --name mattersim_eng python=3.10 pip -y
pip install orb-models 
pip install "pynanoflann@git+https://github.com/dwastberg/pynanoflann#egg=af434039ae14bedcbb838a7808924d6689274168"

every model is now working



