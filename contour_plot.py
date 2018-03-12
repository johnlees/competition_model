import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

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

x = np.array(x).reshape(len(np.unique(x)), len(np.unique(y)))
y = np.array(y).reshape(len(np.unique(x)), len(np.unique(y)))
z = np.array(z).reshape(len(np.unique(x)), len(np.unique(y)))

# Default delta is large because that makes it fast, and it illustrates
# the correct registration between image and contours.
delta = 0.5

#extent = (-3, 4, -4, 3)

# Boost the upper limit to avoid truncation errors.
#levels = np.arange(-2.0, 1.601, 0.4)
levels=np.array([0, 1, 2, 3])

norm = cm.colors.Normalize(vmax=abs(z).max(), vmin=-abs(z).max())
cmap = cm.PRGn

fig, ax = plt.subplots()
ax.set_yscale("log")

cset1 = plt.contourf(x, y, z, levels,
                     cmap=cm.get_cmap(cmap, len(levels) - 1), norm=norm)
# It is not necessary, but for the colormap, we need only the
# number of levels minus 1.  To avoid discretization error, use
# either this number or a large number such as the default (256).

# If we want lines as well as filled regions, we need to call
# contour separately; don't try to change the edgecolor or edgewidth
# of the polygons in the collections returned by contourf.
# Use levels output from previous call to guarantee they are the same.

cset2 = plt.contour(x, y, z, cset1.levels, colors='k')

# We don't really need dashed contour lines to indicate negative
# regions, so let's turn them off.

for c in cset2.collections:
    c.set_linestyle('solid')

# It is easier here to make a separate call to contour than
# to set up an array of colors and linewidths.
# We are making a thick green line as a zero contour.
# Specify the zero level as a tuple with only 0 in it.

cset3 = plt.contour(z, y, z, (0,), colors='g', linewidths=2)
plt.title('Filled contours')
plt.colorbar(cset1)

plt.show()
