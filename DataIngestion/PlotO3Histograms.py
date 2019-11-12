import matplotlib
matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.pyplot import cm
from matplotlib.patches import CirclePolygon
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from scipy.special import erf
from scipy.optimize import minimize, minimize_scalar
import scipy.stats as st
from scipy.integrate import simps, quad
from scipy.interpolate import interp2d

from astropy.table import Table



_50_area = [82,37,399,1378,214,104,31,44,134,152,169,144,162,286,217,14,196,310,79,11,19,24,5,32,108,3472,643,8040,70,534,81,536,8061]
_90_area = [24220,1748,303,2107,318,24264,2482,14753,359,228,23,104,151,443,7246,921,826,49,1483,1172,488,765,967,939,691,252,1166,448,1131,7461,1444,156,387]
dists = [108,709,548,438,1584,230,632,241,1528,1946,267,874,2839,869,227,781,5263,1849,926,797,1136,3931,3154,2950,1987,1388,227,421,377,156,1628,812,1473]

bns_dists = [230,241,227,227,377,156]
bns_50_area = [8040,3472,79,31,214,1378]
bns_90_area = [24264,14753,7246,1166,1131,7461]

histogram_data = [
    (r"50th percentile area [deg$^2$]", "lightseagreen", _50_area, "O3_50_area.png"),
    (r"90th percentile area [deg$^2$]", "steelblue", _90_area, "O3_90_area.png"),
    (r"Mean distance [Mpc]", "darkorange", dists, "O3_dist.png"),

    (r"BNS 50th percentile area [deg$^2$]", "lightseagreen", bns_dists, "BNS_O3_50_area.png"),
    (r"BNS 90th percentile area [deg$^2$]", "steelblue", bns_50_area, "BNS_O3_90_area.png"),
    (r"BNS Mean distance [Mpc]", "darkorange", bns_90_area, "BNS_O3_dist.png")
]

for hd in histogram_data:
    x_axis_label = hd[0]
    clr = hd[1]
    data = hd[2]
    fname = hd[3]
    print("Creating %s..." % fname)

    # histogram on log scale.
    # Use non-equal bin sizes, such that they look equal on log scale.
    hist, bins, patches = plt.hist(data, bins=8)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    fig = plt.figure(figsize=(4, 4), dpi=1000)
    ax = fig.add_subplot(111)
    ax.hist(data, bins=logbins, edgecolor='black', linewidth=1.2, facecolor=clr)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(r"$N$")
    ax.set_xscale('log')

    fig.savefig(fname, bbox_inches='tight')
    plt.close('all')
    print("\t... Done.")
