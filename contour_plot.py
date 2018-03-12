import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from scipy.ndimage.filters import gaussian_filter

import argparse

description = 'Calculate model regimes'
parser = argparse.ArgumentParser()
parser.add_argument('--runs', required=True, help='Runs of simulator')
args = parser.parse_args()

x = []
y = []
z = []
with open(args.runs, 'r') as run_file:
    header = run_file.readline()
    for line in run_file:
        (y_val, x_val, z_val) = line.rstrip().split("\t")
        x.append(float(x_val))
        y.append(float(y_val))
        z.append(float(z_val))

x = np.array(x).reshape(len(np.unique(y)), len(np.unique(x)))
y = np.array(y).reshape(len(np.unique(y)), len(np.unique(x)))
z = np.array(z).reshape(len(np.unique(y)), len(np.unique(x)))

# Smooth out noise
smooth_z = gaussian_filter(z, 2)

# Default delta is large because that makes it fast, and it illustrates
# the correct registration between image and contours.
delta = 0.5

#extent = (-3, 4, -4, 3)

# position of contours
#all_levels=np.array([0, 1, 2, 3, 4, 5])
win_levels=np.array([0, 1])

norm = cm.colors.Normalize(vmax=abs(smooth_z).max(), vmin=-abs(smooth_z).max())
cmap = cm.PRGn

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.set_xlabel('Arrival time (lag)')
ax.set_ylabel('Challenger inoculum')

# heatmap
#plt.imshow(z)
#plt.show()

cset1 = plt.contourf(x, y, smooth_z, win_levels, alpha = 0.5)
                     #cmap=cm.get_cmap(cmap, len(levels) - 1), norm=norm)

# Draw a lines on the countour boundaries
#cset2 = plt.contour(x, y, smooth_z, cset1.levels, colors='r')

# Turn off dashed countours
#for c in cset2.collections:
#    c.set_linestyle('solid')

# Resident wins boundary
cset3 = plt.contour(x, y, smooth_z, win_levels, colors='r', linewidths=2)
plt.title('Isogenic challenger')
#plt.colorbar(cset1) # legend

# t_com
plt.axvline(x=3.76, color = 'k', linestyle='--')

plt.show()
