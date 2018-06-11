#!python

from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from collections import defaultdict

from scipy.ndimage.filters import gaussian_filter

import argparse

description = 'Calculate model regimes'
parser = argparse.ArgumentParser()
parser.add_argument('--runs', required=True, help='Runs of simulator')
parser.add_argument('--output', default='intergenic_domains', help='Output prefix')
parser.add_argument('--title', default='Intergenic challenger', help='Plot title')
parser.add_argument('--smooth', action='store_true', default=False, help='Smooth noisy data')
args = parser.parse_args()

x = defaultdict(list)
y = defaultdict(list)
z = defaultdict(list)
cset = {}

with open(args.runs, 'r') as run_file:
    header = run_file.readline()
    for line in run_file:
        (x_val, y_val, final_R, final_C, z_val, tchal) = line.rstrip().split("\t")
        x[tchal].append(float(x_val))
        y[tchal].append(float(y_val))
        z[tchal].append(float(z_val))

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('Gamma (challenger -> resident)')
ax.set_ylabel('Gamma (resident -> challenger)')
# position of contours
levels=np.array([0, 0.49, 0.51, 1])

colors = ['k', 'k', 'k', 'k']
linestyles = ['-', '--', ':', '-.']

for tchal in z:
    new_x = np.array(x[tchal]).reshape(len(np.unique(y[tchal])), len(np.unique(x[tchal])))
    new_y = np.array(y[tchal]).reshape(len(np.unique(y[tchal])), len(np.unique(x[tchal])))
    new_z = np.array(z[tchal]).reshape(len(np.unique(y[tchal])), len(np.unique(x[tchal])))

    # Smooth out noise
    if args.smooth:
        new_z = gaussian_filter(new_z, 0.6)

    if tchal == "1":
        cset1 = plt.contourf(new_x, new_y, new_z, levels, colors=('#a200a5', '#ffffff', '#289600'))

    cset[tchal] = plt.contour(new_x, new_y, new_z, levels, colors=colors.pop(), linewidths=1, linestyles=linestyles.pop())

    #fmt = {}
    #strs = [tchal, tchal, '', tchal]
    #for l, s in zip(levels, strs):
    #    fmt[l] = s
    #plt.clabel(cset[tchal], cset[tchal].levels, inline=True, fmt=fmt, fontsize=10)


plt.title(args.title)
#plt.colorbar(cset1) # legend

plt.savefig(args.output + ".pdf")
plt.close()

