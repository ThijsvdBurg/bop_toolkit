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

from bop_toolkit_lib import split_to_config_name


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
      'correct_th': [[th] for th in np.arange(0.02, 0.101, 0.01)] # correct one
      # 'correct_th': [[th] for th in np.arange(0.05, 0.101, 0.025)]
      # 'correct_th': [[th] for th in np.arange(0.1, 0.101, 0.025)] # for testing
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
    # 'zebrapose_husky_experiment_{:02d}_obj07_exp2_{}.csv'.format(config.dataset_split_num,config.predictor),    
    # 'zebrapose_husky_experiment_{:02d}_obj07_exp{}_{}.csv'.format(config.dataset_split_num,config.exp_type,config.predictor),    
    # 'zebrapose_husky_experiment_0{:01d}_obj07_exp{}_{}.csv'.format(config.dataset_split_num,config.exp_type,config.predictor),    
    # 'zebrapose_husky_experiment_1{:01d}_obj07_exp{}_{}.csv'.format(config.dataset_split_num,config.exp_type,config.predictor),    
    # 'zebrapose_husky_experiment_{:02d}_obj07_exp0_{}.csv'.format(config.dataset_split_num,config.predictor),    
    # 'zebrapose_husky_experiment_{:02d}_obj07_exp1_{}.csv'.format(config.dataset_split_num,config.predictor),
  'result_filenames': [
    # 'zebrapose_husky_experiment_00_obj07_exp2_ZP.csv',
    # 'zebrapose_husky_experiment_01_obj07_exp2_ZP.csv',
    # 'zebrapose_husky_experiment_01_obj07_exp2_ZP.csv',
    # 
    # 'zebrapose_husky_experiment_09_obj07_exp2_ZP.csv',
    # 'zebrapose_husky_experiment_19_obj07_exp2_ZP.csv',
    # 
    # 'zebrapose_husky_experiment_00_obj07_exp{}_{}.csv'.format(config.exp_type, config.predictor),
    # 'zebrapose_husky_experiment_01_obj07_exp{}_{}.csv'.format(config.exp_type, config.predictor),
    # 'zebrapose_husky_experiment_10_obj07_exp{}_{}.csv'.format(config.exp_type, config.predictor),
    'zebrapose_husky_experiment_11_obj07_exp{}_{}.csv'.format(config.exp_type, config.predictor),
    'zebrapose_husky_experiment_12_obj07_exp{}_{}.csv'.format(config.exp_type, config.predictor),
    'zebrapose_husky_experiment_13_obj07_exp{}_{}.csv'.format(config.exp_type, config.predictor),
    'zebrapose_husky_experiment_14_obj07_exp{}_{}.csv'.format(config.exp_type, config.predictor),
    # 'zebrapose_husky_experiment_15_obj07_exp{}_{}.csv'.format(config.exp_type, config.predictor),
    'zebrapose_husky_experiment_16_obj07_exp{}_{}.csv'.format(config.exp_type, config.predictor),
    'zebrapose_husky_experiment_17_obj07_exp{}_{}.csv'.format(config.exp_type, config.predictor),
    'zebrapose_husky_experiment_18_obj07_exp{}_{}.csv'.format(config.exp_type, config.predictor),
    # 
    # 'zebrapose_husky_experiment_00_obj07_exp{}_ZP.csv'.format(config.exp_type),
    # 'zebrapose_husky_experiment_00_obj07_exp{}_ZP.csv'.format(config.exp_type),
    # 'zebrapose_husky_experiment_10_obj07_exp{}_ZP.csv'.format(config.exp_type),
    # 
    # 'zebrapose_husky_experiment_01_obj07_exp{}_ZP.csv'.format(config.exp_type),
    # 'zebrapose_husky_experiment_02_obj07_exp{}_ZP.csv'.format(config.exp_type),
    # 'zebrapose_husky_experiment_03_obj07_exp{}_ZP.csv'.format(config.exp_type),
    # 'zebrapose_husky_experiment_04_obj07_exp{}_ZP.csv'.format(config.exp_type),
    # 'zebrapose_husky_experiment_05_obj07_exp{}_ZP.csv'.format(config.exp_type),
    # 'zebrapose_husky_experiment_06_obj07_exp{}_ZP.csv'.format(config.exp_type),
    # 'zebrapose_husky_experiment_07_obj07_exp{}_ZP.csv'.format(config.exp_type),
    # 'zebrapose_husky_experiment_08_obj07_exp{}_ZP.csv'.format(config.exp_type),
    # 'zebrapose_husky_experiment_11_obj07_exp{}_ZP.csv'.format(config.exp_type),
    # 'zebrapose_husky_experiment_12_obj07_exp{}_ZP.csv'.format(config.exp_type),
    # 'zebrapose_husky_experiment_13_obj07_exp{}_ZP.csv'.format(config.exp_type),
    # 'zebrapose_husky_experiment_14_obj07_exp{}_ZP.csv'.format(config.exp_type),
    # 'zebrapose_husky_experiment_15_obj07_exp{}_ZP.csv'.format(config.exp_type),
    # 'zebrapose_husky_experiment_16_obj07_exp{}_ZP.csv'.format(config.exp_type),
    # 'zebrapose_husky_experiment_17_obj07_exp{}_ZP.csv'.format(config.exp_type),
    # 'zebrapose_husky_experiment_18_obj07_exp{}_ZP.csv'.format(config.exp_type),
  ],
    # 'zebrapose_husky_experiment_00_obj07_exp0_ZP.csv',
    # 'zebrapose_husky_experiment_01_obj07_exp0_ZP.csv',
    # 'zebrapose_husky_experiment_02_obj07_exp0_ZP.csv',
    # 'zebrapose_husky_experiment_03_obj07_exp0_ZP.csv',
    # 'zebrapose_husky_experiment_04_obj07_exp0_ZP.csv',
    # 'zebrapose_husky_experiment_05_obj07_exp0_ZP.csv',
    # 'zebrapose_husky_experiment_06_obj07_exp0_ZP.csv',
    # 'zebrapose_husky_experiment_07_obj07_exp0_ZP.csv',
    # 'zebrapose_husky_experiment_08_obj07_exp0_ZP.csv',
    # 'zebrapose_husky_experiment_10_obj07_exp0_ZP.csv',
    # 'zebrapose_husky_experiment_11_obj07_exp0_ZP.csv',
    # 'zebrapose_husky_experiment_12_obj07_exp0_ZP.csv',
    # 'zebrapose_husky_experiment_13_obj07_exp0_ZP.csv',
    # 'zebrapose_husky_experiment_14_obj07_exp0_ZP.csv',
    # 'zebrapose_husky_experiment_15_obj07_exp0_ZP.csv',
    # 'zebrapose_husky_experiment_16_obj07_exp0_ZP.csv',
    # 'zebrapose_husky_experiment_17_obj07_exp0_ZP.csv',
    # 'zebrapose_husky_experiment_18_obj07_exp0_ZP.csv',
    # 'zebrapose_husky_experiment_02_obj07_exp2_ZP.csv',
    # 'zebrapose_husky_experiment_12_obj07_exp2_ZP.csv',
    # 'zebrapose_husky_experiment_03_obj07_exp2_ZP.csv',
    # 'zebrapose_husky_experiment_13_obj07_exp2_ZP.csv',
    # 
    # 'zebrapose_husky_experiment13_obj07_30-74_20230426_ZP.csv'

  # 'calc_errors': False,
  'calc_errors': True,
  # 'calc_metrics': False,
  'calc_metrics': True,

  # Folder with results to be evaluated.
  'results_path': config.results_path,

  # Folder for the calculated pose errors and performance scores.
  'eval_path': config.eval_path,

  # File with a list of estimation targets to consider. The file is assumed to
  # be stored in the dataset folder.
  'targets_filenames': [
    # 'test_targets_bop19_WIP_000028.json',
    # 'test_targets_bop19_WIP_000029.json'
  ]
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
parser.add_argument('--targets_filenames',
                    default=','.join(p['targets_filenames']),
                    help='Comma-separated names of files with results.')
                    #  default=p['targets_filenames'])
args = parser.parse_args()

p['renderer_type'] = str(args.renderer_type)
p['result_filenames'] = args.result_filenames.split(',')
p['targets_filenames'] = args.targets_filenames.split(',')
p['results_path'] = str(args.results_path)
p['eval_path'] = str(args.eval_path)
# p['targets_filename'] = str(args.targets_filename)

# Evaluation.
# ------------------------------------------------------------------------------
for result_filename in p['result_filenames']:

  misc.log('===========')
  misc.log('EVALUATING: {}/{}'.format(p['results_path'],result_filename))
  misc.log('===========')

  time_start = time.time()

  # Volume under recall surface (VSD) / area under recall curve (MSSD, MSPD).
  average_recalls = {}
  average_IoU = {}
  average_score = {}

  # Name of the result and the dataset.
  result_name = os.path.splitext(os.path.basename(result_filename))[0]
  dataset = str(result_name.split('_')[1].split('-')[0])
  split_num = result_name.split('_')[3]
  exp_type = result_name.split('_')[5]
  # Calculate the average estimation time per image.
  # ests = inout.load_bop_results(
    # os.path.join(p['results_path'], result_filename), version='bop19')
  ests = inout.load_bop_results(
    os.path.join(p['results_path'], result_filename), version='husky23')
  
  if p['predictor']=='ZP':
    if exp_type == 'exp0':
      p['targets_filenames'] = ['test_targets_bop19_WIP_00{:02d}25.json'.format(int(split_num)),
      'test_targets_bop19_WIP_00{:02d}26.json'.format(int(split_num)),
      'test_targets_bop19_WIP_00{:02d}28.json'.format(int(split_num)),
      'test_targets_bop19_WIP_00{:02d}29.json'.format(int(split_num)),
      'test_targets_bop19_WIP_00{:02d}02.json'.format(int(split_num)),
      'test_targets_bop19_WIP_00{:02d}03.json'.format(int(split_num)),
      'test_targets_bop19_WIP_00{:02d}13.json'.format(int(split_num))]
      # 'test_targets_bop19_WIP_00{:02d}18.json'.format(split_num),
      # 'test_targets_bop19_WIP_00{:02d}12.json'.format(int(split_num)),
      # 'test_targets_bop19_WIP_00{:02d}20.json'.format(split_num),
    elif exp_type == 'exp1':
      p['targets_filenames'] = ['test_targets_bop19_WIP_00{:02d}33.json'.format(int(split_num)),
      'test_targets_bop19_WIP_00{:02d}37.json'.format(int(split_num)),
      #'test_targets_bop19_WIP_00{:02d}38.json'.format(int(split_num)),
      'test_targets_bop19_WIP_00{:02d}43.json'.format(int(split_num)),
      'test_targets_bop19_WIP_00{:02d}44.json'.format(int(split_num)),
      'test_targets_bop19_WIP_00{:02d}45.json'.format(int(split_num)),
      # 'test_targets_bop19_WIP_00{:02d}46.json'.format(int(split_num)),
      'test_targets_bop19_WIP_00{:02d}48.json'.format(int(split_num)),
      'test_targets_bop19_WIP_00{:02d}50.json'.format(int(split_num)),
      'test_targets_bop19_WIP_00{:02d}51.json'.format(int(split_num)),
      'test_targets_bop19_WIP_00{:02d}52.json'.format(int(split_num)),
      'test_targets_bop19_WIP_00{:02d}84.json'.format(int(split_num))]
    elif exp_type == 'exp2':
      p['targets_filenames'] = ['test_targets_bop19_WIP_00{:02d}61.json'.format(int(split_num)),
      # 'test_targets_bop19_WIP_00{:02d}53.json'.format(int(split_num)),
      # 'test_targets_bop19_WIP_00{:02d}54.json'.format(int(split_num)),
      # 'test_targets_bop19_WIP_00{:02d}57.json'.format(int(split_num)),
      # 'test_targets_bop19_WIP_00{:02d}59.json'.format(int(split_num)),
      # 'test_targets_bop19_WIP_00{:02d}60.json'.format(int(split_num)),
      # 'test_targets_bop19_WIP_00{:02d}66.json'.format(int(split_num)),
      # 'test_targets_bop19_WIP_00{:02d}67.json'.format(int(split_num)),
      # 'test_targets_bop19_WIP_00{:02d}68.json'.format(int(split_num)),
      # 'test_targets_bop19_WIP_00{:02d}79.json'.format(int(split_num)),
      # 'test_targets_bop19_WIP_00{:02d}80.json'.format(int(split_num)),
      # 'test_targets_bop19_WIP_00{:02d}87.json'.format(int(split_num)),
      # 'test_targets_bop19_WIP_00{:02d}88.json'.format(int(split_num)),
      # 'test_targets_bop19_WIP_00{:02d}91.json'.format(int(split_num))
      ]
      # 'test_targets_bop19_WIP_00{:02d}74.json'.format(int(split_num)), #very sketchy run, bc of mppi controller
      # 'test_targets_bop19_WIP_00{:02d}56.json'.format(int(split_num)), #out bc of sketchy run
  else:
    if exp_type == 'exp0':
      p['targets_filenames'] = ['test_targets_bop19_WIP_010025.json']
    elif exp_type == 'exp1':
      p['targets_filenames'] = ['test_targets_bop19_WIP_010033.json']
    elif exp_type == 'exp2':
      p['targets_filenames'] = ['test_targets_bop19_WIP_010061.json']

  times_total = {}
  times_2DDet = {}
  times_PnP = {}
  times_available = True
  for est in ests:
    result_key = '{:06d}_{:06d}'.format(est['scene_id'], est['im_id'])
    if int(est['time'][2]) < 0:
      # All estimation times must be provided.
      times_available = False
      # break
      # raise ValueError(
          # 'The running time for scene {} and image {} is not the same for '
          # 'all estimates.'.format(est['scene_id'], est['im_id']))
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

  target_string = '{}'.format(p['targets_filenames'])
  target_string = target_string.split('[')
  target_string = target_string[1].split(']')[0]
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
      '--targets_filenames={}'.format(target_string),
      '--max_sym_disc_step={}'.format(p['max_sym_disc_step']),
      '--skip_missing=1',
      '--split_num={}'.format(split_num),
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

    if p['calc_errors']:
      if subprocess.call(calc_errors_cmd) != 0:
        raise RuntimeError('Calculation of pose errors failed.')

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
          '--visib_gt_min={}'.format(p['visib_gt_min']),
        ]
        '''
        calc_scores_cmd = [
          'python',
          os.path.join(os.path.dirname(os.path.realpath(__file__)), 'eval_calc_scores_edit.py'),
          '--error_dir_paths={}'.format(error_dir_path),
          '--eval_path={}'.format(p['eval_path']),
          # '--targets_filenames={}'.format(p['targets_filenames']),
          '--targets_filenames={}'.format(target_string),
          '--visib_gt_min={}'.format(p['visib_gt_min']),
          '--split_num={}'.format(split_num),
        ]
        calc_scores_cmd += ['--correct_th_{}={}'.format(
          error['type'], ','.join(map(str, correct_th)))]

        if p['calc_metrics']:
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
  misc.log('Evaluation of {}/{} took {}s.'.format(p['results_path'],result_filename, time_total))

  # Calculate the final scores.
  final_scores = {}
  final_scores['__config_number__'] = split_to_config_name.convert(split_num, config.exp_type)
  for error in p['errors']:
    final_scores['bop19_average_recall_{}'.format(error['type'])] =\
      average_recalls[error['type']]

  # Final score for the given dataset.
  final_scores['bop19_average_recall'] = np.mean([
    average_recalls['add']]) #, average_recalls['mssd'], average_recalls['mspd']])

  if p['predictor']=='ZP':
    # Average estimation time per image.
    final_scores['bop19_average_time_per_image'] = average_total_time_per_image
    final_scores['bop19_average_pnp_time_per_image'] = average_pnp_time_per_image
    final_scores['bop19_average_2D_detect_time_per_image'] = average_2ddet_time_per_image

  # final_scores['husky23_mean_score_config_{}'.format(config.int(split_num))] = average_score[error['type']]
  # final_scores['husky23_mean_IoU_config_{}'.format(config.int(split_num))] = average_IoU[error['type']]
  final_scores['husky23_mean_score'] = average_score[error['type']]
  final_scores['husky23_mean_IoU'] = average_IoU[error['type']]

  # Save the final scores.
  final_scores_path = os.path.join(
    p['eval_path'], result_name, 'scores_bop19.json')
  inout.save_json(final_scores_path, final_scores)

  # Print the final scores.
  misc.log('FINAL SCORES:')
  for score_name, score_value in final_scores.items():
    misc.log('- {}: {}'.format(score_name, score_value))

misc.log('Done.')
