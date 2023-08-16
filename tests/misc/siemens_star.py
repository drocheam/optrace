#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

#  create figure without frame and axes
fig = plt.figure(figsize=(4, 4), frameon=False)
ax = plt.axes()
ax.set_axis_off()

spokes = 128

# background rectangle
t1 = plt.Polygon([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], color="k", linewidth=0)
plt.gca().add_patch(t1)

# draw spokes
for i in range(spokes):
    step = 2*2*np.pi/spokes
    phi0 = i*step
    phi1 = i*step + step/2
    t1 = plt.Polygon([[0, 0], [np.cos(phi0), np.sin(phi0)], [np.cos(phi1), np.sin(phi1)]], color="w", linewidth=0)
    plt.gca().add_patch(t1)

# set extent
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.tight_layout()

plt.savefig(f"./optrace/ressources/images/siemens_star.webp", bbox_inches='tight', pad_inches=0, dpi=300, transparent=False)
