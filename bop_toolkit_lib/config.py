# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Configuration of the BOP Toolkit."""

import os


######## Basic ########

# Folder with the BOP datasets.
if 'BOP_PATH' in os.environ:
  datasets_path = os.environ['BOP_PATH']
else:
  datasets_path = r'/home/pmvanderburg/noetic-husky/datasets/bop_datasets'

# Folder with pose results to be evaluated.
results_path = r'/home/pmvanderburg/noetic-husky/datasets/bop_datasets/pose_results'

# Folder for the calculated pose errors and performance scores.
eval_path = r'/home/pmvanderburg/noetic-husky/datasets/bop_datasets/evaluation'

######## Extended ########

# Folder for outputs (e.g. visualizations).
#output_path = r'/home/pmvanderburg/noetic-husky/datasets/bop_datasets'
output_path = r'/media/pmvanderburg/T7/bop_datasets'
#output_path = r'/home/pmvanderburg/noetic-husky/datasets/bop_datasets_T7'

# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = r'/path/to/bop_renderer/build'

# Executable of the MeshLab server.
meshlab_server_path = r'/usr/bin/meshlabserver'

# Custom
#dataset_name = 'tudl'
dataset_name = 'husky'
#dataset_name = 'lmo'
#dataset_name = 'husky_devel'
#dataset_split = 'test'
#dataset_split = 'train_tmp_9000'
#dataset_split = 'train_tmp_8999'
dataset_split = os.environ['split_type']
