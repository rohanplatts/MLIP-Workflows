
Last update: 31/01/2026. 

Models are not always mutually compatible. However, ASE is the standard for atomic work, consequently most models have a described method (in their documentation) by which one can access the model as an ASE calculator object. Thus, models most often are supported for ASE. 

It is best to assume that the models are not compatible, however of course many of them are. MACE, orb-models, PET-mad are compatible, thus i have a dedicated environment to those. I have a second environment for MatterSim, and a separate environment for the MATGL models. 

It does not ultimately matter if a model is incompatible with another, as seen in the configuration file, for every new model you add to this project, you also point it to the name of the environment for which that model runs in. NOTE: ASE and PHONOPY must be installed into that environment as well. This should raise no dependency conflicts. 

For the models i have currently implemented, i have attached their .yml files. However, if these do not work for one reason or another, the exact installations i ran in terminal to obtain these environments are as follows: 

NOTE: install this package first. Clone the repository, cd to the project directory, and for each environment (for each model) you use,  run: 

```
pip install -e . 
```


mace_env installs:

```
conda create --name mace_env python=3.10 pip 
pip install mace-torch 
conda install -c conda-forge phonopy
pip install upet
pip install orb-models
pip install "pynanoflann@git+https://github.com/dwastberg/pynanoflann#egg=af434039ae14bedcbb838a7808924d6689274168"
```

matgl_env installs: 

```
conda create --name matgl_env python=3.10 pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
pip install matgl
pip install ase
pip install phonopy
```

mattersim_env installs: 

```
conda create --name mattersim_env python=3.10 pip 
pip install mattersim
pip install phonopy
```

NOTE: for the matgl installation, at that time DGL (a matgl backend that governed model tensor operation) was not supported and thus had to be manually installed since the matgl models had not been shifted from dgl to pytorch (PyG) yet. This was january 2026, and may have since changed. If you are getting some backend error with the installation, Read their updates here:
https://matgl.ai/#major-update-v200-nov-12-2025 you should be check for which backend (PyG or DGL) your model uses and install that. If the installation is messy and not working, i found these tips helpful:

1. install the desired version of torch first 
2. install the required backend for matgl first 
3. install matgl

NOTES FOR DEV: 
of 31/01/2026: Checked mace_environment.yml, matgl_env and it loaded all calc objects

For mace:
```    
conda create --name mace_env python=3.10 pip 
pip install mace-torch 
pip install phonopy
```

for mattersim:

```
conda create --name mattersim_env python=3.10 pip -y 
pip install mattersim 
```

for orb-models: 

```
conda create --name orb_env python=3.10 pip -y
pip install orb-models
pip install "pynanoflann@git+https://github.com/dwastberg/pynanoflann#egg=af434039ae14bedcbb838a7808924d6689274168"
```

for pet-mad:

```
conda create --name petmad_env python=3.10 pip -y
pip install upet 
```





