import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, ticker
from math import log

from scipy.ndimage.filters import gaussian_filter

import argparse

description = 'Calculate model regimes'
parser = argparse.ArgumentParser()
parser.add_argument('--runs', required=True, help='Runs of simulator')
parser.add_argument('--boundary', type=float, default=10.0, help='Population boundary line')
parser.add_argument('--output', default='isogenic_domains', help='Output prefix')
parser.add_argument('--smooth', action='store_true', default=False, help='Smooth noisy data')
args = parser.parse_args()

plt.rcParams.update({'font.size': 14})

x = []
y = []
z = []
with open(args.runs, 'r') as run_file:
    header = run_file.readline()
    for line in run_file:
        (C_size, t_chal, avg_R, avg_C) = line.rstrip().split("\t")
        if float(avg_C) < 1:
            avg_C = 0.001
        x.append(float(t_chal))
        y.append(float(C_size))
        z.append(float(avg_C))

x = np.array(x).reshape(len(np.unique(y)), len(np.unique(x)))
y = np.array(y).reshape(len(np.unique(y)), len(np.unique(x)))
z = np.array(z).reshape(len(np.unique(y)), len(np.unique(x)))

# Smooth out noise
if args.smooth:
    z = gaussian_filter(z, 1.2)

#extent = (-3, 4, -4, 3)

# position of contours
win_levels=np.array([0, args.boundary])
all_levels=np.array([0, 1, 10, 100, 1000])

norm = cm.colors.Normalize(vmax=abs(z).max(), vmin=-abs(z).max())
cmap = cm.PRGn

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.set_xlabel('Lag in arrival time (hrs)')
ax.set_ylabel('Challenger inoculum (CFU)')

cset1 = plt.contourf(x, y, z, win_levels, locator=ticker.LogLocator(), colors=('#289600', '#ffffff'))
                     #cmap=cm.get_cmap(cmap, len(levels) - 1), norm=norm)
# t_com
plt.axvline(x=3.76, color = 'k', linestyle='--', label='t_com')
plt.text(2.7, 3000, 't_com',rotation=90)

#cset2 = plt.contour(x, y, z, all_levels, locator=ticker.LogLocator(), colors='k', linewidths = 1)
#plt.clabel(cset2, fmt='%1.0f', inline=1, fontsize=10)

# Resident wins boundary
cset3 = plt.contour(x, y, z, win_levels, colors='k', linewidths=2)
plt.title('Isogenic challenger')
#plt.colorbar(cset1) # legend


plt.savefig(args.output + ".pdf")
plt.close()

