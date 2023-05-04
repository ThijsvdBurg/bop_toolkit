# Author: P.M. van der Burg (pmvanderburg@tudelft.nl)

"""Conversion of the ZebraPose encoded configuration to the desired config name."""

# import os

def convert(config_encoding):
  # num = 
  if int(config_encoding[1]) == 0:
    pnp_iters = 1
  elif int(config_encoding[1]) == 1:
    pnp_iters = 3
  elif int(config_encoding[1]) == 2:
    pnp_iters = 5
  elif int(config_encoding[1]) == 3:
    pnp_iters = 10
  elif int(config_encoding[1]) == 4:
    pnp_iters = 15
  elif int(config_encoding[1]) == 5:
    pnp_iters = 25
  elif int(config_encoding[1]) == 6:
    pnp_iters = 50
  elif int(config_encoding[1]) == 7:
    pnp_iters = 100
  elif int(config_encoding[1]) == 8:
    pnp_iters = 150

  if config_encoding[0] == '1':
    # pnp_init=1
    config_name = 'Ours + initial MPPI pose + '+str(pnp_iters)+' RANSAC/PnP iterations'
  else:
    # pnp_init=0
    config_name = 'Ours + random RANSAC/PnP initialization + '+str(pnp_iters)+' RANSAC/PnP iterations'

  return config_name

if __name__ == '__main__':
  convert()