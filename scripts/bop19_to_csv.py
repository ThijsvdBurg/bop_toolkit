"""Evaluation script for the BOP Challenge 2019/2020."""

import os
import time
import argparse
import subprocess
import numpy as np

from bop_toolkit_lib import config
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc

import pandas as pd


# PARAMETERS (some can be overwritten by the command line arguments below).
################################################################################
p = {
  # Errors to calculate.
  'errors': [
    {
      'n_top': 1,
      'type': 'add',
      # 'type': 'vsd',
      'vsd_deltas': {
        'hb': 15,
        'icbin': 15,
        'icmi': 15,
        'itodd': 5,
        'lm': 15,
        'lmo': 15,
        'ruapc': 15,
        'tless': 15,
        'tudl': 15,
        'tyol': 15,
        'ycbv': 15,
        'hope': 15,
      },
      'vsd_taus': list(np.arange(0.05, 0.51, 0.05)),
      'vsd_normalized_by_diameter': False,
      'correct_th': [[th] for th in np.arange(0.01, 0.101, 0.01)]
    },
  ],

  # Minimum visible surface fraction of a valid GT pose.
  # -1 == k most visible GT poses will be considered, where k is given by
  # the "inst_count" item loaded from "targets_filename".
  'visib_gt_min': -1,

  # See misc.get_symmetry_transformations().
  'max_sym_disc_step': 0.01,

  # Type of the renderer (used for the VSD pose error function).
  'renderer_type': 'vispy',  # Options: 'vispy', 'cpp', 'python'.

  'predictor': config.predictor,
  # Names of files with results for which to calculate the errors (assumed to be
  # stored in folder p['results_path']). See docs/bop_challenge_2019.md for a
  # description of the format. Example results can be found at:
  # https://bop.felk.cvut.cz/media/data/bop_sample_results/bop_challenge_2019/
  'result_filenames': [
    'zebrapose_husky_experiment_13_obj07_30-74_20230426_ZP.csv',
    'zebrapose_husky_experiment_14_obj07_30-91_20230426_ZP.csv',
  ],

  'calc_errors': False,

  # Folder with results to be evaluated.
  'results_path': config.results_path,

  # Folder for the calculated pose errors and performance scores.
  'eval_path': config.eval_path,

  # File with a list of estimation targets to consider. The file is assumed to
  # be stored in the dataset folder.
  # 'targets_filenames': [
    # 'test_targets_bop19_WIP_000028.json',
    # 'test_targets_bop19_WIP_000029.json'
  # ]
  # 'targets_filename': 'test_targets_bop19.json',
}
################################################################################
for result_filename in p['result_filenames']:
  # Name of the result and the dataset.
  result_name = os.path.splitext(os.path.basename(result_filename))[0]
  dataset = str(result_name.split('_')[1].split('-')[0])
  split_num = result_name.split('_')[3]
  

# Save the final scores.
  final_scores_path = os.path.join(
    p['eval_path'], result_name, 'scores_bop19.json')
  final_scores=inout.load_json(final_scores_path)

  csv_path=r'/home/pmvanderburg/noetic-husky/bop_ros_ws/src/Husky_scripts/performance_metrics.csv'
  # Load the CSV file into a pandas dataframe
  df = pd.read_csv(csv_path)

  # Calculate performance metrics and store in variables
  metric_1 = final_scores['bop19_average_recall']
  metric_2 = final_scores['bop19_average_time_per_image']

  # Add the performance metrics to the correct cells in the dataframe
  df.loc[0, 'Average Recall'] = metric_1
  df.loc[0, 'Inference Time'] = metric_2

  # Save the updated dataframe back to the original CSV file
  df.to_csv('existing_file.csv', index=False)
