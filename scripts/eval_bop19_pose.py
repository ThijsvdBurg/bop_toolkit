# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Evaluation script for the BOP Challenge 2019/2020."""

import os
import time
import argparse
import subprocess
import numpy as np

from bop_toolkit_lib import config
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc


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
      # 'correct_th': [[th] for th in np.arange(0.01, 0.101, 0.01)]
      'correct_th': [[th] for th in np.arange(0.05, 0.101, 0.025)]
    },
    # {
      # 'n_top': -1,
      # 'type': 'mssd',
      # 'correct_th': [[th] for th in np.arange(0.05, 0.51, 0.05)]
    # },
    # {
      # 'n_top': -1,
      # 'type': 'mspd',
      # 'correct_th': [[th] for th in np.arange(5, 51, 5)]
    # },
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
    # 'zebrapose_husky_experiment_obj07_28_29.csv',
    # 'zebrapose_husky_experiment_obj07_28_29_20230419_2.csv',
    # 'zebrapose_husky_experiment0_obj07_26-32_20230421_{}.csv'.format(config.predictor)
    # 'zebrapose_husky_experiment08_obj07_33-38_20230425_{}.csv'.format(config.predictor),
    'zebrapose_husky_experiment18_obj07_29-43_20230425_{}.csv'.format(config.predictor)
  ],


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


# Command line arguments.
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--renderer_type', default=p['renderer_type'])
parser.add_argument('--result_filenames',
                    default=','.join(p['result_filenames']),
                    help='Comma-separated names of files with results.')
parser.add_argument('--results_path', default=p['results_path'])
parser.add_argument('--eval_path', default=p['eval_path'])
# parser.add_argument('--targets_filenames',
                    # default=','.join(p['targets_filenames']),
                    # help='Comma-separated names of files with results.')
                    #  default=p['targets_filenames'])
args = parser.parse_args()

p['renderer_type'] = str(args.renderer_type)
p['result_filenames'] = args.result_filenames.split(',')
# p['targets_filenames'] = args.targets_filenames.split(',')
p['results_path'] = str(args.results_path)
p['eval_path'] = str(args.eval_path)
# p['targets_filename'] = str(args.targets_filename)

# Evaluation.
# ------------------------------------------------------------------------------
for result_filename in p['result_filenames']:

  misc.log('===========')
  misc.log('EVALUATING: {}'.format(result_filename))
  misc.log('===========')

  time_start = time.time()

  # Volume under recall surface (VSD) / area under recall curve (MSSD, MSPD).
  average_recalls = {}
  average_IoU = {}
  average_score = {}

  # Name of the result and the dataset.
  result_name = os.path.splitext(os.path.basename(result_filename))[0]
  dataset = str(result_name.split('_')[1].split('-')[0])

  # Calculate the average estimation time per image.
  # ests = inout.load_bop_results(
    # os.path.join(p['results_path'], result_filename), version='bop19')
  ests = inout.load_bop_results(
    os.path.join(p['results_path'], result_filename), version='husky23')
  
  times_total = {}
  times_2DDet = {}
  times_PnP = {}
  times_available = True
  for est in ests:
    result_key = '{:06d}_{:06d}'.format(est['scene_id'], est['im_id'])
    if int(est['time'][2]) < 0:
      # All estimation times must be provided.
      times_available = False
      break
    elif result_key in times_total:
      if abs(times_total[result_key] - est['time'][2]) > 0.001:
        raise ValueError(
          'The running time for scene {} and image {} is not the same for '
          'all estimates.'.format(est['scene_id'], est['im_id']))
    else:
      times_total[result_key] = est['time'][2] # ms to sec
      times_2DDet[result_key] = est['time'][0] # ms to sec
      times_PnP[result_key] = est['time'][1] # ms to sec

  if times_available:
    average_total_time_per_image = np.mean(list(times_total.values()))
    average_pnp_time_per_image = np.mean(list(times_PnP.values()))
    average_2ddet_time_per_image = np.mean(list(times_2DDet.values()))
  else:
    average_total_time_per_image = -1.0

  # Evaluate the pose estimates.
  for error in p['errors']:
    '''
    # Calculate error of the pose estimates.
    calc_errors_cmd = [
      'python',
      os.path.join(os.path.dirname(os.path.realpath(__file__)), 'eval_calc_errors.py'),
      '--n_top={}'.format(error['n_top']),
      '--error_type={}'.format(error['type']),
      '--result_filenames={}'.format(result_filename),
      '--renderer_type={}'.format(p['renderer_type']),
      '--results_path={}'.format(p['results_path']),
      '--eval_path={}'.format(p['eval_path']),
      '--targets_filename={}'.format(p['targets_filename']),
      '--max_sym_disc_step={}'.format(p['max_sym_disc_step']),
      '--skip_missing=1',
    ]
    '''
    # Calculate error of the pose estimates.
    calc_errors_cmd = [
      'python',
      os.path.join(os.path.dirname(os.path.realpath(__file__)), 'eval_calc_errors_edit.py'),
      '--n_top={}'.format(error['n_top']),
      '--error_type={}'.format(error['type']),
      '--result_filenames={}'.format(result_filename),
      # '--result_filenames={}'.format(p['result_filenames']),
      '--renderer_type={}'.format(p['renderer_type']),
      '--results_path={}'.format(p['results_path']),
      '--eval_path={}'.format(p['eval_path']),
      # '--targets_filenames={}'.format(targets_filename),
      '--max_sym_disc_step={}'.format(p['max_sym_disc_step']),
      '--skip_missing=1',
    ]
    if error['type'] == 'vsd':
      vsd_deltas_str = \
        ','.join(['{}:{}'.format(k, v) for k, v in error['vsd_deltas'].items()])
      calc_errors_cmd += [
        '--vsd_deltas={}'.format(vsd_deltas_str),
        '--vsd_taus={}'.format(','.join(map(str, error['vsd_taus']))),
        '--vsd_normalized_by_diameter={}'.format(
          error['vsd_normalized_by_diameter'])
      ]

    misc.log('Running: ' + ' '.join(calc_errors_cmd))

    # if subprocess.call(calc_errors_cmd) != 0:
      # raise RuntimeError('Calculation of pose errors failed.')

    # Paths (rel. to p['eval_path']) to folders with calculated pose errors.
    # For VSD, there is one path for each setting of tau. For the other pose
    # error functions, there is only one path.
    error_dir_paths = {}
    if error['type'] == 'vsd':
      for vsd_tau in error['vsd_taus']:
        error_sign = misc.get_error_signature(
          error['type'], error['n_top'], vsd_delta=error['vsd_deltas'][dataset],
          vsd_tau=vsd_tau)
        error_dir_paths[error_sign] = os.path.join(result_name, error_sign)
    else:
      error_sign = misc.get_error_signature(error['type'], error['n_top'])
      error_dir_paths[error_sign] = os.path.join(result_name, error_sign)

    # Recall scores for all settings of the threshold of correctness (and also
    # of the misalignment tolerance tau in the case of VSD).
    recalls = []
    mIoU = []
    score = []

    # Calculate performance scores.
    for error_sign, error_dir_path in error_dir_paths.items():
      for correct_th in error['correct_th']:
        '''
        calc_scores_cmd = [
          'python',
          os.path.join(os.path.dirname(os.path.realpath(__file__)), 'eval_calc_scores.py'),
          '--error_dir_paths={}'.format(error_dir_path),
          '--eval_path={}'.format(p['eval_path']),
          '--targets_filename={}'.format(p['targets_filename']),
          '--visib_gt_min={}'.format(p['visib_gt_min'])
        ]
        '''
        calc_scores_cmd = [
          'python',
          os.path.join(os.path.dirname(os.path.realpath(__file__)), 'eval_calc_scores_edit.py'),
          '--error_dir_paths={}'.format(error_dir_path),
          '--eval_path={}'.format(p['eval_path']),
          # '--targets_filenames={}'.format(p['targets_filenames']),
          '--visib_gt_min={}'.format(p['visib_gt_min'])
        ]
        calc_scores_cmd += ['--correct_th_{}={}'.format(
          error['type'], ','.join(map(str, correct_th)))]

        misc.log('Running: ' + ' '.join(calc_scores_cmd))
        if subprocess.call(calc_scores_cmd) != 0:
          raise RuntimeError('Calculation of scores failed.')

        # Path to file with calculated scores.
        score_sign = misc.get_score_signature(correct_th, p['visib_gt_min'])

        scores_filename = 'scores_{}.json'.format(score_sign)
        scores_path = os.path.join(
          p['eval_path'], result_name, error_sign, scores_filename)

        # Load the scores.
        misc.log('Loading calculated scores from: {}'.format(scores_path))
        scores = inout.load_json(scores_path)
        recalls.append(scores['recall'])
        mIoU.append(scores['2D bbox IoU'])
        score.append(scores['mean abs score'])

    average_recalls[error['type']] = np.mean(recalls)
    average_score[error['type']] = np.mean(score)
    average_IoU[error['type']] = np.mean(mIoU)

    misc.log('Recall scores per threshold: {}'.format(' '.join(map(str, recalls))))
    misc.log('AUC of ADD 0.1d: {}'.format(average_recalls[error['type']]))
    misc.log('Mean abs ADD: {}'.format(average_score[error['type']]))
    misc.log('mIoU: {}'.format(average_IoU[error['type']]))

  time_total = time.time() - time_start
  misc.log('Evaluation of {} took {}s.'.format(result_filename, time_total))

  # Calculate the final scores.
  final_scores = {}
  final_scores['__config_number__'] = config.dataset_split_num
  for error in p['errors']:
    final_scores['bop19_average_recall_{}'.format(error['type'])] =\
      average_recalls[error['type']]

  # Final score for the given dataset.
  final_scores['bop19_average_recall'] = np.mean([
    average_recalls['add']]) #, average_recalls['mssd'], average_recalls['mspd']])

  # Average estimation time per image.
  final_scores['bop19_average_time_per_image'] = average_total_time_per_image
  final_scores['bop19_average_pnp_time_per_image'] = average_pnp_time_per_image
  final_scores['bop19_average_2D_detect_time_per_image'] = average_2ddet_time_per_image

  final_scores['husky23_mean_score_config_{}'.format(config.dataset_split_num)] = average_score[error['type']]
  final_scores['husky23_mean_IoU_config_{}'.format(config.dataset_split_num)] = average_IoU[error['type']]


  # Save the final scores.
  final_scores_path = os.path.join(
    p['eval_path'], result_name, 'scores_bop19.json')
  inout.save_json(final_scores_path, final_scores)

  # Print the final scores.
  misc.log('FINAL SCORES:')
  for score_name, score_value in final_scores.items():
    misc.log('- {}: {}'.format(score_name, score_value))

misc.log('Done.')
