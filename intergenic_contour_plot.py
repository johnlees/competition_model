import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from scipy.ndimage.filters import gaussian_filter

import argparse

description = 'Calculate model regimes'
parser = argparse.ArgumentParser()
parser.add_argument('--runs', required=True, help='Runs of simulator')
parser.add_argument('--output', default='intergenic_domains', help='Output prefix')
parser.add_argument('--title', default='Intergenic challenger', help='Plot title')
parser.add_argument('--smooth', action='store_true', default=False, help='Smooth noisy data')
args = parser.parse_args()

x = []
y = []
z = []
with open(args.runs, 'r') as run_file:
    header = run_file.readline()
    for line in run_file:
        (x_val, y_val, final_R, final_C, z_val) = line.rstrip().split("\t")
        x.append(float(x_val))
        y.append(float(y_val))
        z.append(float(z_val))

x = np.array(x).reshape(len(np.unique(y)), len(np.unique(x)))
y = np.array(y).reshape(len(np.unique(y)), len(np.unique(x)))
z = np.array(z).reshape(len(np.unique(y)), len(np.unique(x)))

# Smooth out noise
if args.smooth:
    z = gaussian_filter(z, 0.6)

#extent = (-3, 4, -4, 3)

# position of contours
levels=np.array([0, 0.49, 0.51, 1])
#levels=np.linspace(0,1,11)

norm = cm.colors.Normalize(vmax=1, vmin=0)
cmap = cm.coolwarm

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('Gamma (challenger -> resident)')
ax.set_ylabel('Gamma (resident -> challenger)')

cset1 = plt.contourf(x, y, z, levels, colors=('#fc8d59', '#ffffbf', '#91bfdb'))
#cset1.cmap.set_under('yellow')
#cset1.cmap.set_over('cyan')
#cset1 = plt.contourf(x, y, z, levels, cmap='coolwarm')

# Resident wins boundary
cset3 = plt.contour(x, y, z, levels, colors='k', linewidths=2)
plt.title(args.title)
#plt.colorbar(cset1) # legend
#plt.clabel(cset3, fmt='%2.1f', colors='k', fontsize=14) # contour labels

# Labels
#fmt = {}
#strs = ['challenger', 'co-exist', 'resident', 'no']
#for l, s in zip(cset1.levels, strs):
#    fmt[l] = s

#plt.clabel(cset3, cset3.levels, inline=True, fmt=fmt, fontsize=10)

plt.savefig(args.output + ".pdf")
plt.close()

