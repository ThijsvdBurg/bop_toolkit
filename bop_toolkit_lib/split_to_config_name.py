# Author: P.M. van der Burg (pmvanderburg@tudelft.nl)

"""Conversion of the ZebraPose encoded configuration to the desired config name."""

# import os

def convert(config_encoding, exp_type):
  # num = 
  if int(config_encoding[2]) == 0:
    pnp_iters = 1
  elif int(config_encoding[2]) == 1:
    pnp_iters = 3
  elif int(config_encoding[2]) == 2:
    pnp_iters = 5
  elif int(config_encoding[2]) == 3:
    pnp_iters = 10
  elif int(config_encoding[2]) == 4:
    pnp_iters = 15
  elif int(config_encoding[2]) == 5:
    pnp_iters = 25
  elif int(config_encoding[2]) == 6:
    pnp_iters = 50
  elif int(config_encoding[2]) == 7:
    pnp_iters = 100
  elif int(config_encoding[2]) == 8:
    pnp_iters = 150
  elif int(config_encoding[2]) == 9:
    pnp_iters = 250
  if exp_type == 0:
    init_src='OptiTrack'
    # bbox_src = 'GT'
  elif exp_type == 1:
    init_src='MPPI'
    # bbox_src = 'MPPI '
  elif exp_type == 2:
    init_src='MPPI (based on ZP)'
    # bbox_src = 'MPPI '
  if config_encoding[0] == '0': # ZP
    if config_encoding[1] == '1':
      # pnp_init=1
      # config_name = 'Ours + OptiTrack bbox + initial pose + 100 RPnP iterations'
      # config_name = 'Ours + '+init_src+' bbox + initial pose + '+str(pnp_iters)+' RANSAC/PnP iterations'
      config_name = 'Ours + ' + init_src + ' bbox + initial pose'
    else:
      # pnp_init=0
      # config_name = 'Ours + 100 RANSAC/PnP iterations'
      # config_name = 'Ours + ' + init_src + ' bbox + ' + str(pnp_iters) + ' RPnP'
      config_name = 'Ours + ' + init_src + ' bbox'
  if config_encoding[0] == '1': # MPPI
    if config_encoding[1] == '1':
      # pnp_init=1
      # config_name = 'Ours + OptiTrack bbox + initial pose + 100 RPnP iterations'
      # config_name = 'Ours + '+init_src+' bbox + initial pose + '+str(pnp_iters)+' RANSAC/PnP iterations'
      # config_name = init_src + 'MPPI (based on ZP + initial pose) + ' + str(pnp_iters) + ' RPnP'
      config_name = 'MPPI (based on ZP + initial pose)'
    else:
      # pnp_init=0
      # config_name = 'Ours + 100 RANSAC/PnP iterations'
      # config_name = init_src + ' + ' + str(pnp_iters) + ' RPnP'
      config_name = init_src
  return config_name, pnp_iters

if __name__ == '__main__':
  convert()