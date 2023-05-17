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


# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = r'/path/to/bop_renderer/build'

# Executable of the MeshLab server.
meshlab_server_path = r'/usr/bin/meshlabserver'

# Custom
#dataset_name = 'tudl'
dataset_name = 'husky'
#dataset_name = 'lmo'
#dataset_name = 'husky_devel'
# dataset_split = 'test' # os.environ['split_type']
# exp_type = 0
# exp_type = 1
exp_type = 2

# predictor     = 'MPPI'
# predictor     = 'ZP'
predictor     = 'ZPbaseline'

#dataset_split_num = 0
#dataset_split_num = 1
#dataset_split_num = 2
#dataset_split_num = 3
dataset_split_num = 4
#dataset_split_num = 5
#dataset_split_num = 6
#dataset_split_num = 7
#dataset_split_num = 8
# dataset_split_num = 10
# dataset_split_num = 11
# dataset_split_num = 12
# dataset_split_num = 13
# dataset_split_num = 14
# dataset_split_num = 15
# dataset_split_num = 16
# dataset_split_num = 17
# dataset_split_num = 18

dataset_split = 'experiment_'+'{:02d}'.format(dataset_split_num)  #'experiment'


'''
# for s in {18,25,26,28,29,2,3,12,13}; \
if exp_type == 0:
  target_nums = [
    # 'test_targets_bop19_WIP_00{:02d}18.json'.format(dataset_split_num),
    # 'test_targets_bop19_WIP_00{:02d}20.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}25.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}26.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}28.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}29.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}02.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}03.json'.format(dataset_split_num),
    # 'test_targets_bop19_WIP_00{:02d}12.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}13.json'.format(dataset_split_num),
    ]


# for s in {33,37,38,43,44,45,46,48,49,50,51,52,84}; \
elif exp_type == 1:
  target_nums = [
    'test_targets_bop19_WIP_00{:02d}33.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}37.json'.format(dataset_split_num),
    #'test_targets_bop19_WIP_00{:02d}38.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}43.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}44.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}45.json'.format(dataset_split_num),
    # 'test_targets_bop19_WIP_00{:02d}46.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}48.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}50.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}51.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}52.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}84.json'.format(dataset_split_num),
  ]
else:
  target_nums = [
    'test_targets_bop19_WIP_00{:02d}53.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}54.json'.format(dataset_split_num),
    # 'test_targets_bop19_WIP_00{:02d}56.json'.format(dataset_split_num), #out bc of sketchy run
    'test_targets_bop19_WIP_00{:02d}57.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}59.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}60.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}61.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}66.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}67.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}68.json'.format(dataset_split_num),
    # 'test_targets_bop19_WIP_00{:02d}74.json'.format(dataset_split_num), #very sketchy run, bc of mppi controller
    'test_targets_bop19_WIP_00{:02d}79.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}80.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}87.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}88.json'.format(dataset_split_num),
    'test_targets_bop19_WIP_00{:02d}91.json'.format(dataset_split_num),
  ]
  '''
  

# Folder with pose results to be evaluated.
results_path = os.path.join(datasets_path,dataset_name,'results') #r'/home/pmvanderburg/noetic-husky/datasets/bop_datasets/pose_results'

# Folder for the calculated pose errors and performance scores.
eval_path = os.path.join(datasets_path,dataset_name,'evaluation') # r'/home/pmvanderburg/noetic-husky/datasets/bop_datasets/evaluation'

# bag_path='/home/pmvanderburg/6dof_pose_experiments/20230503_pushing_experiments'
# bag_path='/home/pmvanderburg/6dof_pose_experiments/20230510_pushing_experiments'
bag_path='/media/pmvanderburg/T7/bagfiles/20230510_pushing_experiments'

######## Extended ########

# Folder for outputs (e.g. visualizations).
#output_path = r'/home/pmvanderburg/noetic-husky/datasets/bop_datasets'
output_path = os.path.join(datasets_path,dataset_name,'visualisation') # datasets_path #	r'/media/pmvanderburg/T7/bop_datasets'
#output_path = r'/home/pmvanderburg/noetic-husky/datasets/bop_datasets_T7'
