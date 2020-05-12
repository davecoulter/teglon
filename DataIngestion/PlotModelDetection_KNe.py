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
from scipy import interpolate
from scipy.interpolate import interp1d, interp2d, spline
import scipy as sp
print(sp.__version__)

from astropy.table import Table
import csv
from collections import OrderedDict
import matplotlib.patheffects as path_effects


from scipy.interpolate import griddata
from matplotlib import ticker

import astropy.constants as c

C_CGS = c.c.cgs.value
M_SUN_CGS = c.M_sun.cgs.value
G_CGS = c.G.cgs.value
R_NS_CGS = 20e5 # NS radius cm == 15 km

is_log = False

blue_kn = True # else, red_kn

is_Ye_0_05 = False
is_Ye_0_10 = False
is_Ye_0_20 = False
is_Ye_0_30 = False
is_Ye_0_40 = False
is_Ye_0_45 = False

if blue_kn:
    is_Ye_0_45 = True
else:
    is_Ye_0_10 = True


ye_thresh = 0.05 # default
start_file = 0
end_file = 112

if is_Ye_0_10:
    ye_thresh = 0.10
    start_file = 112
    end_file = 224
elif is_Ye_0_20:
    ye_thresh = 0.20
    start_file = 224
    end_file = 336
elif is_Ye_0_30:
    ye_thresh = 0.30
    start_file = 336
    end_file = 448
elif is_Ye_0_40:
    ye_thresh = 0.40
    start_file = 448
    end_file = 560
elif is_Ye_0_45:
    ye_thresh = 0.45
    start_file = 560
    end_file = 672


model_tups = []


for i in range(start_file, end_file):
    file_num = i+1
    results_table = Table.read("../Events/S190814bv/ModelDetection/Detection_Results_KN_%i.prob" % file_num,
                               format='ascii.ecsv')

    vej = list(results_table['vej'])
    mej = list(results_table['mej'])
    ye = list(results_table['ye'])
    prob = list(results_table['Prob'])

    for j, y in enumerate(ye):
        if y == ye_thresh:
            model_tups.append((vej[j], mej[j], prob[j]))

print("Done reading in data... %s rows" % len((model_tups)))

all_vej = np.asarray([mt[0] for mt in model_tups])
all_mej = np.asarray([mt[1] for mt in model_tups])

points = [(mt[0], mt[1]) for mt in model_tups]
values = [mt[2] for mt in model_tups]

model_vej = np.logspace(np.log10(np.min(all_vej)), np.log10(np.max(all_vej)), 1000)
model_mej = np.logspace(np.log10(np.min(all_mej)), np.log10(np.max(all_mej)), 1000)
grid_vej, grid_mej = np.meshgrid(model_vej, model_mej)

# grid_vej, grid_mej = np.mgrid[model_vej[0]:model_vej[-1]:]


gamma = 1.0/np.sqrt(1.0 - model_vej**2)
grid_Mass_KE = grid_mej*(gamma - 1.0)*M_SUN_CGS*C_CGS**2 # ergs

grid_prob = griddata(points, values, (grid_vej, grid_mej), method='linear', rescale=True)
min_prob = np.min(values)
max_prob = np.max(values)

print(np.nanmin(grid_prob))
print(np.nanmax(grid_prob))

print(max_prob)
print(min_prob)

# raise Exception()



# default to linear
norm = colors.Normalize(min_prob, max_prob)
tks = np.linspace(min_prob, max_prob, 5)
if is_log:
    norm = colors.LogNorm(min_prob, max_prob)
    tks = np.logspace(np.log10(min_prob), np.log10(max_prob), 5)



fig = plt.figure(figsize=(10, 10), dpi=1000)
ax = fig.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')

# sorted_tups = sorted(model_tups, key=lambda x: x[2], reverse=False)
# for mt in sorted_tups:
#     clr = plt.cm.viridis(norm(mt[2]))
#     ax.plot(mt[0], mt[1], marker='s', color=clr, markersize=5.0, alpha=1.0) # alpha=0.5


def bound_mass(beta):
    mass_Msol = ((5.0*R_NS_CGS*C_CGS**2)/(3.0*G_CGS))*(1.0/np.sqrt(1.0 - beta**2) - 1.0)/M_SUN_CGS
    return mass_Msol

b_mass = bound_mass(model_vej)
ceiling = np.full(len(b_mass), 5e-1)

ax.plot(model_vej, b_mass, linestyle="--", color="black", label="Binding Energy", zorder=9900)
# ax.fill_between(model_vej, ceiling, b_mass, zorder=9900, hatch="\\\\", edgecolor="gray", facecolor="none", linewidth=0.0)
ax.fill_between(model_vej, ceiling, b_mass, zorder=9900, color="gray")




ke_mass_lvls = [1e49, 1e50, 1e51, 1e52, 1e53]
# ke_mass_lvls = [1e-4*M_SUN_CGS*C_CGS**2/1.7871525773260012, 1e-3*M_SUN_CGS*C_CGS**2/1.7871525773260012, 1e-2*M_SUN_CGS*C_CGS**2/1.7871525773260012, 1e-1*M_SUN_CGS*C_CGS**2/1.7871525773260012]

manual_locations = [(8e-2, 2.5e-3), (6e-1, 3e-4), (6e-1, 5e-3), (6e-1, 3e-2),  (6e-1, 3e-1)]
ke_lines = ax.contour(grid_vej, grid_mej, grid_Mass_KE, levels=ke_mass_lvls, colors='black',
                      locator=ticker.LogLocator(), zorder=8888)
plt.setp(ke_lines.collections, path_effects=[path_effects.withStroke(linewidth=2.0, foreground='white')])

ke_fmt_dict = {
    1e49:r"$10^{49}$ ergs",
    1e50:r"$10^{50}$ ergs",
    1e51:r"$10^{51}$ ergs",
    1e52:r"$10^{52}$ ergs",
    1e53:r"$10^{53}$ ergs"
}
ke_clbls = ax.clabel(ke_lines, inline=True, fontsize=12, fmt=ke_fmt_dict, inline_spacing=10.0, manual=manual_locations)
plt.setp(ke_clbls, path_effects=[path_effects.withStroke(linewidth=1.5, foreground='white')], zorder=8888)




ax.contourf(grid_vej, grid_mej, grid_prob, cmap=plt.cm.viridis,
            levels=np.logspace(np.log10(min_prob), np.log10(max_prob), 300), zorder=8800)

if blue_kn:
    CS_lines = ax.contour(grid_vej, grid_mej, grid_prob, colors="red", linewidths=2.0,
                          levels=[0.0, 0.01, 0.1, 0.25], zorder=9900) # , 0.35
    plt.setp(CS_lines.collections, path_effects=[path_effects.withStroke(linewidth=2.5, foreground='black')])

    fmt_dict = {
        0.0:"",
        0.01:"1%",
        0.10:"10%",
        0.25:"25%",
        # 0.35:"35%"
    }
    clbls = ax.clabel(CS_lines, inline=True, fontsize=24, fmt=fmt_dict, inline_spacing=100.0)
    plt.setp(clbls, path_effects=[path_effects.withStroke(linewidth=1.0, foreground='black')], zorder=9900)
else:
    CS_lines = ax.contour(grid_vej, grid_mej, grid_prob, colors="red", linewidths=2.0,
                          levels=[0.0, 1e-6, 2e-6, 5e-6], zorder=9900)
    plt.setp(CS_lines.collections, path_effects=[path_effects.withStroke(linewidth=2.5, foreground='black')])

    fmt_dict = {
        0.0: "",
        1e-6: r"$1 \times 10^{-4}$ %",
        2e-6: r"$3 \times 10^{-4}$ %",
        5e-6: r"$6 \times 10^{-4}$ %",

    }
    clbls = ax.clabel(CS_lines, inline=True, fontsize=24, fmt=fmt_dict, inline_spacing=120.0)
    plt.setp(clbls, path_effects=[path_effects.withStroke(linewidth=1.0, foreground='black')], zorder=9900)


ax.text(9e-2, 2.1e-1, r"$U_{R_{20}} \geq KE$", rotation=31, fontsize=18, zorder=9999)


sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
sm.set_array([]) # can be an empty list

tks_strings = []
if blue_kn:
    tks_strings = [
        "0%",
        "11%",
        "22%",
        "33%",
        "44%",
    ]
else:
    tks_strings = [
        "0.1",
        "3",
        "6",
        "9",
        "10",
    ]


# tks_strings = ["0%", "11%", "22%", "33%", "44%"]

cb = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.02, alpha=0.80)
cb.set_ticks(tks)
cb.ax.set_yticklabels(tks_strings, fontsize=24)
if blue_kn:
    cb.set_label("", fontsize=24, labelpad=-5.0)
else:
    cb.set_label(r"$\times 10^{-4}$ %", fontsize=24, labelpad=-5.0)
cb.ax.tick_params(length=8.0, width=2.0)
cb.ax.locator_params(nbins=5)


# ax.text(7e-2, 2e-4, r"$Y_e$ = " + "%0.2f" % actual_ye, fontsize=24)
ax.text(7e-2, 2e-4, r"$Y_e$ = " + "%0.2f" % ye_thresh, fontsize=24,
        path_effects=[path_effects.withStroke(linewidth=2.5, foreground='white')], zorder=9999)

if blue_kn:
    # SSS17a - Blue KN
    ax.errorbar(0.25, 0.025, fmt="*", mfc="deepskyblue", mec="black", ms=24.0, zorder=9999, mew=1.5)
    ax.text(0.25, 0.011, "AT 2017gfo\nBlue Component", fontsize=18, ha="center", zorder=9999,
            path_effects=[path_effects.withStroke(linewidth=2.0, foreground='white')])
else:
    # SSS17a - Red KN
    ax.errorbar(0.15, 0.035, fmt="*", mfc="red", mec="black", ms=24.0, zorder=9999, mew=1.5)
    ax.text(0.15, 0.016, "AT 2017gfo\nRed Component", fontsize=18, ha="center", zorder=9999,
            path_effects=[path_effects.withStroke(linewidth=2.0, foreground='white')])


ax.grid(which='both', axis='both', linestyle=':', color="gray", zorder=1)

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2.0)
cb.outline.set_linewidth(2.0)

# ax.set_zorder(9999)
ax.tick_params(axis='both', which='major', labelsize=24, length=12.0, width=2)
ax.tick_params(axis='both', which='minor', labelsize=24, length=8.0, width=2)

ax.set_ylim([1e-4, 5e-1])

ax.set_ylabel(r'$\mathrm{M_{ej}}$ ($\mathrm{M_{\odot}}$)', fontsize=32, labelpad=9.0)
ax.set_xlabel(r'$\mathrm{V_{ej}}$ ($\beta$)', fontsize=32)

fig.savefig('KNE_Sensitivity_Test_ye_%0.3f.png' % ye_thresh, bbox_inches='tight')
plt.close('all')
print("... Done.")
