import torch
from bop_toolkit_lib import visualization

tensor = torch.randn(16, 3,256,256,dtype=torch.float)

# visualization.visualise_tensor(tensor, ch=0)
visualization.visualise_tensor(tensor)