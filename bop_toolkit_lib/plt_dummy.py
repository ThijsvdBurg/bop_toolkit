# import matplotlib.pyplot as plt
# import numpy as np

# 
# torch.randn(32,32,dtype=torch.float)
# plot = plt.imshow(img)

import matplotlib.pyplot as plt
import numpy as np
import torch

# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

# fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()
# ax.plot(t, s)

# img = torch.randn(32,32,dtype=torch.float)
img = plt.imread('/home/pmvanderburg/Screenshot.png')
ax1.imshow(img)

# ax.set(xlabel='time (s)', ylabel='voltage (mV)',
    #    title='About as simple as it gets, folks')
# ax.grid()

# fig.savefig("test.png")
plt.show()
