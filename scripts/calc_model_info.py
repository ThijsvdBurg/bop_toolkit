# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates the 3D bounding box and the diameter of 3D object models."""
from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from pybop_lib.debug_tools import printdebug

# PARAMETERS.
################################################################################
p = {
  # See dataset_params.py for options.
  'dataset': config.dataset_name,

  # Type of input object models.
  # 'model_type': None,
  'model_type': eval,

  # Folder containing the BOP datasets.
  'datasets_path': config.datasets_path,
}
################################################################################


# Load dataset parameters.
dp_model = dataset_params.get_model_params(
  p['datasets_path'], p['dataset'], p['model_type'])

models_info = {}
for obj_id in dp_model['obj_ids']:
    misc.log('Processing model of object {}...'.format(obj_id))

    model = inout.load_ply(dp_model['model_tpath'].format(obj_id=obj_id))

    # Calculate 3D bounding box.
    ref_pt_array_min = model['pts'].min(axis=0).flatten()
    ref_pt_map_min   = map( float, ref_pt_array_min)
    ref_pt_list = list(ref_pt_map_min)

    #size = map(float, (model['pts'].max(axis=0) - ref_pt).flatten())
    ref_pt_array_max = model['pts'].max(axis=0).flatten()
    #ref_pt_map_max   = map( float, ref_pt_map_max)
    size             = ref_pt_array_max - ref_pt_array_min

    # Calculated diameter.
    diameter = misc.calc_pts_diameter(model['pts'])

    models_info[obj_id] = {
        'min_x': ref_pt_list[0], 'min_y': ref_pt_list[1], 'min_z': ref_pt_list[2],
        'size_x': size[0], 'size_y': size[1], 'size_z': size[2],
        'diameter': diameter
    }

# Save the calculated info about the object models.
inout.save_json(dp_model['models_info_path'], models_info)
