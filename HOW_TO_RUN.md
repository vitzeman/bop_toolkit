# Instalation
mamba/conda installation
```sh
mamba create -n bop python=3.7
mamba activate bop
pip install --upgrade pip setuptools
mamba install numpy
mamba install cython
pip install -r requirements.txt -e .
```

# New dataset
In [config.py](bop_toolkit_lib/config.py) setup:
- `dataset_path`  as path to the directory containing all BOP datasets.
- `result_path` path to folder containing the csv files.
\<`Method`>\_\<`DatasetName`>\_\<`DatasetSplit`>.csv
  - Names in \<`Method`>, <`DatasetName`> & <`DatasetSplit`> **should not contain** the underscores or dashes. As the BOP scripts rely on this during runtime. Use camelCase
  - <`DatasetSplit`> is normally `test`
- `eval_path` path to directory where the results of BOP metrics will be saved
- `RESULT_FILENAMES` list with the names of the .csv files in format 

In [dataset_params.py](bop_toolkit_lib/dataset_params.py) in function `get_model_params()` setup:
- In dictionary `obj_ids` define new dataset object ids in format:
  - "\<DatasetName>": list[int] # List of integers
- In dictionary `symmetric_obj_ids` define which object are symmetric:
  - "\<DatasetName>": list[int] # List should be subset of obj_ids


In [eval_bop19_pose.py](scripts/eval_bop19_pose.py) setup:
- in parameters `p` setup
  - `errors` types used for the mAR 
  - `targets_filename` path to evaluation targets

# Running the scripts