# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Visualization utilities."""

import os
# import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import pose_error

from pybop_lib.debug_tools import printdebug
from pybop_lib.debug_tools import printMaxMin
from pybop_lib.manipulate_depth import vis_depth

import matplotlib.pyplot as plt
import math

def draw_rect(im, rect, color=(1.0, 1.0, 1.0), width=1):
  """Draws a rectangle on an image.

  :param im: ndarray (uint8) on which the rectangle will be drawn.
  :param rect: Rectangle defined as [x, y, width, height], where [x, y] is the
    top-left corner.
  :param color: Color of the rectangle.
  :return: Image with drawn rectangle.
  """
  if im.dtype != np.uint8:
    raise ValueError('The image must be of type uint8.')

  im_pil = Image.fromarray(im)
  draw = ImageDraw.Draw(im_pil)
  draw.rectangle((rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]),
                 outline=tuple([int(c * 255) for c in color]), fill=None, width=width
                )
  del draw
  return np.asarray(im_pil)


def write_text_on_image(im, txt_list, loc=(3, 0), color=(1.0, 1.0, 1.0),
                        size=20):
  """Writes text info on an image.

  :param im: ndarray on which the text info will be written.
  :param txt_list: List of dictionaries, each describing one info line:
    - 'name': Entry name.
    - 'val': Entry value.
    - 'fmt': String format for the value.
  :param loc: Location of the top left corner of the text box.
  :param color: Font color.
  :param size: Font size.
  :return: Image with written text info.
  """
  im_pil = Image.fromarray(im)

  # Load font.
  try:
    font_path = os.path.join(os.path.dirname(__file__), 'droid_sans_mono.ttf')
    font = ImageFont.truetype(font_path, size)
  except IOError:
    misc.log('Warning: Loading a fallback font.')
    font = ImageFont.load_default()

  draw = ImageDraw.Draw(im_pil)
  for info in txt_list:
    if info['name'] != '':
      txt_tpl = '{}:{' + info['fmt'] + '}'
    else:
      txt_tpl = '{}{' + info['fmt'] + '}'
    if not 'iou' in info:
      txt = txt_tpl.format(info['name'], info['val'])
    else:
      txt = txt_tpl.format(info['name'], info['iou'][0])
    draw.text(loc, txt, fill=tuple([int(c * 255) for c in color]), font=font)
    text_width, text_height = font.getsize(txt)
    loc = (loc[0], loc[1] + text_height)
  del draw

  return np.array(im_pil)


def depth_for_vis(depth, valid_start=0.2, valid_end=1.0):
  """Transforms depth values from the specified range to [0, 255].

  :param depth: ndarray with a depth image (1 channel).
  :param valid_start: The beginning of the depth range.
  :param valid_end: The end of the depth range.
  :return: Transformed depth image.
  """
  mask = depth > 0
  depth_n = depth.astype(np.float)
  depth_n[mask] -= depth_n[mask].min()
  depth_n[mask] /= depth_n[mask].max() / (valid_end - valid_start)
  depth_n[mask] += valid_start
  return depth_n


def vis_object_poses(
      poses, K, renderer, rgb=None, depth=None, vis_rgb_path=None,
      vis_depth_diff_path=None, vis_depth_diff_path_debug=None,
      vis_rgb_resolve_visib=False,vis_iou_path=None):
  """Visualizes 3D object models in specified poses in a single image.

  Two visualizations are created:
  1. An RGB visualization (if vis_rgb_path is not None).
  2. A Depth-difference visualization (if vis_depth_diff_path is not None).
  3. A bbox visualisation (if vis_iou_bbox is not False).

  :param poses: List of dictionaries, each with info about one pose:
    - 'obj_id': Object ID.
    - 'R': 3x3 ndarray with a rotation matrix.
    - 't': 3x1 ndarray with a translation vector.
    - 'text_info': Info to write at the object (see write_text_on_image).
  :param K: 3x3 ndarray with an intrinsic camera matrix.
  :param renderer: Instance of the Renderer class (see renderer.py).
  :param rgb: ndarray with the RGB input image.
  :param depth: ndarray with the depth input image.
  :param vis_rgb_path: Path to the output RGB visualization.
  :param vis_depth_diff_path: Path to the output depth-difference visualization.
  :param vis_rgb_resolve_visib: Whether to resolve visibility of the objects
    (i.e. only the closest object is visualized at each pixel).
  """
  fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

  # Indicators of visualization types.
  vis_rgb = vis_rgb_path is not None
  vis_depth_diff = vis_depth_diff_path is not None
  vis_iou = vis_iou_path is not None


  if vis_rgb and rgb is None:
    raise ValueError('RGB visualization triggered but RGB image not provided.')

  if (vis_depth_diff or (vis_rgb and vis_rgb_resolve_visib)) and depth is None:
    raise ValueError('Depth visualization triggered but D image not provided.')

  # Prepare images for rendering.
  im_size = None
  ren_rgb = None
  ren_rgb_info = None
  ren_depth = None
  ren_iou_info = None

  if vis_rgb:
    im_size = (rgb.shape[1], rgb.shape[0])
    ren_rgb = np.zeros(rgb.shape, np.uint8)
    ren_rgb_info = np.zeros(rgb.shape, np.uint8)

  if vis_iou:
    ren_iou_info = np.zeros(rgb.shape, np.uint8)

  if vis_depth_diff:
    if im_size and im_size != (depth.shape[1], depth.shape[0]):
        raise ValueError('The RGB and D images must have the same size.')
    else:
      im_size = (depth.shape[1], depth.shape[0])

  if vis_depth_diff or (vis_rgb and vis_rgb_resolve_visib):
    ren_depth = np.zeros((im_size[1], im_size[0]), np.float32)

  # Render the pose estimates one by one.
  for pose in poses:

    # Rendering.
    ren_out = renderer.render_object(
      pose['obj_id'], pose['R'], pose['t'], fx, fy, cx, cy)

    m_rgb = None
    if vis_rgb:
      m_rgb = ren_out['rgb']

    m_mask = None
    if vis_depth_diff or (vis_rgb and vis_rgb_resolve_visib):
      m_depth = ren_out['depth']

      # Get mask of the surface parts that are closer than the
      # surfaces rendered before.
      visible_mask = np.logical_or(ren_depth == 0, m_depth < ren_depth)
      m_mask = np.logical_and(m_depth != 0, visible_mask)

      ren_depth[m_mask] = m_depth[m_mask].astype(ren_depth.dtype)

    # Combine the RGB renderings.
    if vis_rgb:
      if vis_rgb_resolve_visib:
        ren_rgb[m_mask] = m_rgb[m_mask].astype(ren_rgb.dtype)
      else:
        ren_rgb_f = ren_rgb.astype(np.float32) + m_rgb.astype(np.float32)
        ren_rgb_f[ren_rgb_f > 255] = 255
        ren_rgb = ren_rgb_f.astype(np.uint8)

      # Draw 2D bounding box and write text info.
      obj_mask = np.sum(m_rgb > 0, axis=2)
      ys, xs = obj_mask.nonzero()
      if len(ys):
        # bbox_color = model_color
        # text_color = model_color
        bbox_color = (0.3, 0.3, 0.3)
        text_color = (1.0, 1.0, 1.0)
        text_size = 11

        bbox = misc.calc_2d_bbox(xs, ys, im_size)
        im_size = (obj_mask.shape[1], obj_mask.shape[0])
        ren_rgb_info = draw_rect(ren_rgb_info, bbox, bbox_color)

        if 'text_info' in pose:
          #text_loc = (bbox[0] + 2, bbox[1])
          text_loc = (bbox[0] + 2 + bbox[2], bbox[1])
          ren_rgb_info = write_text_on_image(
            ren_rgb_info, pose['text_info'], text_loc, color=text_color,
            size=text_size)

    # calculate and render IoU.
    if vis_iou:

      # Draw 2D bounding box and write text info.
      # obj_mask = np.sum(m_rgb > 0, axis=2)
      # ys, xs = obj_mask.nonzero()
      if len(ys):
        # bbox_color = model_color
        # text_color = model_color
        bbox_gt_color = (0,1.0,0)
        bbox_est_color = (1.0,0,0)
        # text_color = (1.0, 1.0, 1.0)
        # text_size = 11
        bbox_width = 2
        ren_iou_info = draw_rect(ren_iou_info, bbox, bbox_gt_color, bbox_width)

        bbox_est = pose['bbox']
        ren_iou_info = draw_rect(ren_iou_info, bbox_est, bbox_est_color, bbox_width)

        iou = [pose_error.intersection_over_union(bbox, bbox_est)]
        if 'text_info' in pose:
          pose['text_info'][0]['iou'] = iou
          text_loc = (bbox[0] + 2, bbox[1])
          ren_iou_info = write_text_on_image(
            ren_iou_info, pose['text_info'], text_loc, color=text_color,
            size=text_size)

  # Blend and save the RGB visualization.
  if vis_rgb:
    misc.ensure_dir(os.path.dirname(vis_rgb_path))

    vis_im_rgb = 0.5 * rgb.astype(np.float32) + \
                 0.5 * ren_rgb.astype(np.float32) + \
                 1.0 * ren_rgb_info.astype(np.float32)
    vis_im_rgb[vis_im_rgb > 255] = 255
    inout.save_im(vis_rgb_path, vis_im_rgb.astype(np.uint8), jpg_quality=95)

  # Blend and save the RGB visualization.
  if vis_iou:
    misc.ensure_dir(os.path.dirname(vis_iou_path))

    vis_im_iou = 0.5 * rgb.astype(np.float32) + \
                 0.5 * ren_rgb.astype(np.float32) + \
                 1.0 * ren_iou_info.astype(np.float32)
    vis_im_iou[vis_im_iou > 255] = 255
    inout.save_im(vis_iou_path, vis_im_iou.astype(np.uint8), jpg_quality=95)

  # Save the image of depth differences.
  if vis_depth_diff:
    misc.ensure_dir(os.path.dirname(vis_depth_diff_path))

    # Calculate the depth difference at pixels where both depth maps are valid.
    valid_mask = (depth > 0) * (ren_depth > 0)
    depth_diff = valid_mask * (ren_depth.astype(np.float32) - depth)

    # Get mask of pixels where the rendered depth is at most by the tolerance
    # delta behind the captured depth (this tolerance is used in VSD).
    delta = 15
    below_delta = valid_mask * (depth_diff < delta)
    below_delta_vis = (255 * below_delta).astype(np.uint8)

    depth_diff_vis = 255 * depth_for_vis(depth_diff - depth_diff.min())

    # Pixels where the rendered depth is more than the tolerance delta behing
    # the captured depth will be cyan.
    depth_diff_vis = np.dstack(
      [below_delta_vis, depth_diff_vis, depth_diff_vis]).astype(np.uint8)

    depth_diff_vis[np.logical_not(valid_mask)] = 0
    depth_diff_valid = depth_diff[valid_mask]
    depth_info = [
      {'name': 'min diff', 'fmt': ':.3f', 'val': np.min(depth_diff_valid)},
      {'name': 'max diff', 'fmt': ':.3f', 'val': np.max(depth_diff_valid)},
      {'name': 'mean diff', 'fmt': ':.3f', 'val': np.mean(depth_diff_valid)},
    ]
    depth_diff_vis = write_text_on_image(depth_diff_vis, depth_info)
    inout.save_im(vis_depth_diff_path, depth_diff_vis)

def AUC_graph(cumulative_auc, res, th):
  """
  ==========
  plot(x, y)
  ==========

  See `~matplotlib.axes.Axes.plot`.
  """

  import numpy as np

  # plt.style.use('_mpl-gallery')
  #print('linspace',linspace)
  print('cumulative auc',cumulative_auc)
  # make data
  x = np.linspace(0.1*th, th, num=res)
  y = cumulative_auc

  # plot
  fig, ax = plt.subplots()

  ax.plot(x, y, linewidth=2.0)

  ax.set(xlim=(0, th+.0005),
        ylim=(0, 1))

  plt.show()


def visualise_tensor(intensor, str_id, num_ch, batch_id=0, eval_output_path=os.getcwd):
# def visualise_tensor(tensor, ch=0, allkernels=False, nrows=4, ncols=4):
    """
    The function visualise_tensor() takes the following arguments:

    tensor_gpu: the tensor to be visualised
    num_ch: the number of images in the tensor
    batch_id: the index of the batch to save consecutive images
    eval_output_path: path in which the resulting visualisations will be saved
    
    """
    misc.ensure_dir(eval_output_path)    
    str_id=str_id+'_mppi_net_params'
    tensortype = type(intensor)
    # print(tensortype)
    if str(tensortype) == "<class 'torch.Tensor'>":
      print('input tensor is recognized as class torch.Tensor')
      tensor_gpu = intensor
      tensor_cpu = tensor_gpu.detach().cpu()
      if tensor_cpu.ndim == 4:
        tensor=tensor_cpu.permute(1,2,3,0)
      elif tensor_cpu.ndim == 3:
        tensor=tensor_cpu
    
    elif str(tensortype) == "<class 'numpy.ndarray'>":
      tensor = intensor
      # print('tensor ndim',tensor.ndim)
      if tensor.ndim == 3:
        print('dim of numpy array is 3', tensor.shape)
        # tensor=tensor
    else:
      raise NotImplementedError(f"unknown tensor type: {tensortype}")
    # tensor=tensor_cpu.numpy()
    # print('tensor t type and shape', type(tensor),tensor.shape)
    
    # print('tensor[:,0,0,0]', type(tensor[:,0,0,0]),tensor[:,0,0,0])
    # if allkernels: 
    #   tensor = tensor.view((1,-1) + tensor.shape[-2:])
    # elif ch >= 0: 
    #   tensor_chan = tensor[:,ch,:,:]
    # elif ch < 0:
    #     c = int(tensor.shape[1] / abs(ch))
    #     row = int(c / nrows) if (c % nrows == 0) else int(c / nrows) + 1
    #     col = min(ncols, c) if (c < ncols) else c % ncols
    #     print('row={}, col={}'.format(row, col))     # debug only !!!

    # print('tensor t type and shape', type(tensor),tensor.shape)
    # for index,kernel in enumerate(tensor):
        # print(kernel.shape)
        # print(index)
    if num_ch == 0:
      kernels = np.array(tensor)
      print('numch=0 kernels.shape',kernels.shape)

      fig = plt.figure()

      plt.imshow( kernels )   # show image on subplot with index i+1 and size row x column !!!
      vis_path = os.path.join(eval_output_path, 'test_{}_{}.png'.format(str_id,batch_id))
      fig.savefig(vis_path)
    elif num_ch == 1:
      kernels = np.array(tensor)
      print('numch=1 kernels[0].shape',kernels[0].shape)

      fig = plt.figure()

      plt.imshow( kernels[0].squeeze() )   # show image on subplot with index i+1 and size row x column !!!
      vis_path = os.path.join(eval_output_path, 'test_{}_{}.png'.format(str_id,batch_id))
      fig.savefig(vis_path)

    else:
      tensor = np.array(tensor)
      kernels = np.array([kernel for kernel in tensor])

      print('kernels shape', kernels.shape)
      print( 'numch=16 kernels[0].shape',kernels[0].shape,num_ch )
      nrows = ncols = int(math.sqrt(num_ch))
      fig = plt.figure(figsize=(ncols, nrows)) #    fig = plt.figure()
      for i in range((nrows * ncols)):   # number of subplots to show all kernels/filters of a layer !!!
        ax1 = fig.add_subplot(nrows, ncols , i+1 )   # add subplot to figure with index i+1 and size row x column !!!
        ax1.imshow( (kernels[i]).squeeze() )   # show image on subplot with index i+1 and size row x column !!!
        ax1.axis('off')   # remove axis from the plot/subplot with index i+1 and size row x column !!!
        ax1.set_xticklabels([])   # remove tick labels from the plot/subplot with index i+! and size row x column !!!
        ax1.set_yticklabels([])   # remove tick labels from the plot/subplot with index i+! and size row x column !!!
      plt.tight_layout()      ## adjust spacing between subplots to minimize the overlaps !!
      vis_path = os.path.join(eval_output_path, 'test_{}_{}.png'.format(str_id,batch_id)) # vis_path = os.path.join( eval_output_path, "test_bin_{}.png".format(batch_id) )
      fig.savefig(vis_path)

    # plt.show() # !! debug only
