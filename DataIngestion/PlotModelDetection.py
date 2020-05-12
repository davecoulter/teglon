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
from matplotlib.lines import Line2D
from collections import OrderedDict



# results_table = Table.read("../Events/S190814bv/Models/Detection_Results.prob", format='ascii.ecsv')
results_table = Table.read("../Events/S190425z/ModelDetection/Detection_Results.prob", format='ascii.ecsv')

models = np.asarray(results_table['Model'])
masses = np.asarray(results_table['Mass'])
velocities = np.asarray(results_table['Velocity'])
Xlans = np.asarray(results_table['X_lan'])
probs = np.asarray(results_table['Prob'])

print("Velocities:")
unique_velocities = list(sorted(set(velocities)))
print(unique_velocities)
print("\n\n")

# print(sorted(set(masses)))


# for v in unique_velocities:

# match_index = np.where(velocities == v)[0]
# print(match_index)
#
# print("Models:")
# print(models[match_index])
# print("\n\n")
#
# print("Masses:")
# masses_at_v = masses[match_index]
# print(masses_at_v)
# print("\n\n")
#
# print("Xlans:")
# Xlans_at_v = Xlans[match_index]
# print(sorted(set(Xlans_at_v)))
# print("\n\n")
#
# print("Probs:")
# probs_at_v = probs[match_index]
# print(probs_at_v)
# print("\n\n")

model_tuples = []
for i in range(len(models)):
    model_tuples.append((masses[i], Xlans[i], probs[i], velocities[i]))

print("# of models for %s" % len(model_tuples))

min_prob = np.min(probs)
max_prob = np.max(probs)
print(min_prob, max_prob)
norm = colors.Normalize(min_prob, max_prob)

# if min_prob <= 0.0:
#     min_prob = 1e-18
# norm = colors.LogNorm(min_prob, max_prob)

fig = plt.figure(figsize=(7,7), dpi=1000)
ax = fig.add_subplot(111)

max_model = {}
for mt in model_tuples:
    # local_frac = mt[2]/max_prob
    # clr = plt.cm.viridis(norm(local_frac))
    # clr = plt.cm.viridis(norm(mt[2]))

    # key = (mt[0], mt[1]) # mass vs xlans
    key = (mt[3], mt[1]) # velocity vs xlans
    if key not in max_model:
        # max_model[key] = (mt[2], mt[3]) # prob and velocity
        max_model[key] = (mt[2], mt[0]) # prob and mass
    else:
        if mt[2] >= max_model[key][0]:
            # max_model[key] = (mt[2], mt[3]) # prob and velocity
            max_model[key] = (mt[2], mt[0])  # prob and mass


markers = OrderedDict()
markers[0.020] = ("p", 17)
markers[0.025] = ("H", 17)
markers[0.030] = ("D", 14)
markers[0.035] = ("^", 16)
markers[0.040] = ("o", 16)
legend_elements = []

for key,value in markers.items():
    legend_elements.append(Line2D([0],
                                  [0],
                                  linestyle='None',
                                  marker=value[0],
                                  markerfacecolor="None",
                                  markeredgecolor="k",
                                  markersize=value[1],
                                  label=r"%0.3f $\mathrm{M_{\odot}}$" % key))


for key, value in max_model.items():
    clr = plt.cm.viridis(norm(value[0]))

    # velocity vs xlans...
    ax.plot(key[0], key[1], markers[value[1]][0], color=clr, markersize=markers[value[1]][1]) # , alpha=0.25

    # ax.annotate("%0.3fc" % value[1], xy=(key[0], key[1]), xytext=(key[0], key[1])) # annotate velocity
    # ax.annotate(("%0.3f" % value[1]) + r"$\mathrm{M_{\odot}}$", xy=(key[0], key[1]), xytext=(key[0], key[1]), fontsize=8.0) # annotate mass
    # ax.annotate(r"%0.3f $\mathrm{M_{\odot}}$" % value[1], xy=(key[0]-0.018, key[1]*1.4), fontsize=8.0)  # annotate mass

sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
sm.set_array([]) # can be an empty list
tks = np.linspace(min_prob, max_prob, 5)
# tks = np.logspace(np.log10(min_prob), np.log10(max_prob), 9)
tks_strings = []
for t in tks:
    tks_strings.append('%0.2f' % (t*100))
    # tks_strings.append('%0.2E' % (t * 100))

cb = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.04875, pad=0.02, alpha=0.80)
cb.ax.set_yticklabels(tks_strings, fontsize=16)
cb.set_label("% Prob To Detect", fontsize=16, labelpad=9.0)
cb.ax.tick_params(length=6.0) # width=2.0,
cb.ax.locator_params(nbins=5)
# cb.outline.set_linewidth(2.0)

# ax.xaxis.set_tick_params(width=2.0)
# ax.yaxis.set_tick_params(width=2.0)
# ax.set_yticks([1.0e-09, 1.0e-05, 1.0e-04, 1.0e-03, 1.0e-02, 1.0e-01])
# ax.set_xticks([0.025, 0.03, 0.035, 0.04])

ax.set_yscale('log')

# ax.annotate(r'$\mathrm{v_{ejecta}}$=%s', xy=(0.6, 0.9), xycoords='axes fraction', fontsize=12)
# ax.set_xlim([0, 0.45])
# ax.set_ylim([1e-10, 1.0])


ax.legend(handles=legend_elements, numpoints=1, fontsize=14) # labelspacing=2.0, , loc="upper right"
# plt.xlabel(r'$\mathrm{Mass_{\odot}}$',fontsize=16)
plt.xlabel(r'$\mathrm{v_{ejecta}/c}$',fontsize=16)
plt.ylabel(r'$\mathrm{\chi_{lanthanide}}$',fontsize=16)

# fig.savefig('Prob2Detect_all.png', bbox_inches='tight')
fig.savefig('S190425z_Prob2Detect_KNe_v_vs_Xlans.png', bbox_inches='tight')
plt.close('all')
print("... Done.")
