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



model_X_input = np.linspace(-2.0, 10.0, 100)
model_Y_input = np.linspace(-23.0, -11.0, 100)



# Unpack unordered results
M_list = []
M_dM = OrderedDict()
M_prob = OrderedDict()
test_prob_list = []
for i in range(16):
    sub_dir = i+1
    results_table = Table.read("../Events/S190814bv/ModelDetection/Detection_Results_Linear_%s.prob" % sub_dir,
                               format='ascii.ecsv')
    # results_table = Table.read("../Events/S190425z/ModelDetection/Detection_Results_Linear_%s.prob" % sub_dir,
    #                            format='ascii.ecsv')
    # results_table = Table.read("../Events/S200213t/ModelDetection/Detection_Results_Linear_%s.prob" % sub_dir,
    #                            format='ascii.ecsv')

    M_list = list(results_table['M'])
    dm_list = list(results_table['dM'])
    prob_list = list(results_table['Prob'])
    test_prob_list += prob_list
    for j, M in enumerate(M_list):

        if M not in M_dM:
            M_dM[M] = []

        if M not in M_prob:
            M_prob[M] = []

        M_dM[M].append(dm_list[j])
        M_prob[M].append(prob_list[j])

print(np.min(test_prob_list), np.max(test_prob_list))

M_dM_probs = OrderedDict()
dMs = []
for key, value in M_dM.items():

    sorted_indices = np.argsort(M_dM[key])
    dM = np.asarray(M_dM[key])[sorted_indices]
    dMs = dM
    prob = np.asarray(M_prob[key])[sorted_indices]

    f_dM_prob = interp1d(dM, prob, kind="slinear")
    interp_probs = f_dM_prob(model_X_input)

    M_dM_probs[key] = interp_probs


column_interp = OrderedDict()
for i, dM in enumerate(model_X_input):
    dm_prob_list = []

    sorted_Ms = []
    for j, M in enumerate(sorted(M_dM_probs.keys())):
        sorted_Ms.append(M)
        dm_prob_list.append(M_dM_probs[M][i])

    f_M_prob = interp1d(sorted_Ms, dm_prob_list, kind="slinear")
    interp_probs = f_M_prob(model_Y_input)

    column_interp[dM] = interp_probs


model_tuples = []
for i, (dM, prob_col) in enumerate(column_interp.items()):
    for index, prob in enumerate(prob_col):
        model_tuples.append((model_Y_input[index], dM, prob))


X = []
Y = []
Z = []

for mt in model_tuples:

    Y.append(mt[0])
    X.append(mt[1])
    Z.append(mt[2])

_2d_func = interp2d(X, Y, Z, kind="linear")
xx, yy = np.meshgrid(model_X_input,model_Y_input)
z_new = _2d_func(model_X_input, model_Y_input)



# transient_colors = {
#     "Ia":"orangered",
#     "Ia-91bg":"white",
#     "II":"forestgreen",
#     "Ia-91T":"black",
#     "SLSN-I":"mediumturquoise",
#     "IIb":"orange",
#     "Ic":"grey",
#     "Ib":"orchid",
#     "IIn":"cornflowerblue"
# }
# transients = []
# with open('dm7_i_all_transients.txt') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=' ')
#     next(csv_reader)
#     for row in csv_reader:
#         transient_type = row[0]
#         dm7 = float(row[1])
#         abs_peak = float(row[2])
#
#         transients.append((abs_peak, dm7/7.0, transient_colors[transient_type], transient_type))


min_prob = np.min(z_new)
max_prob = np.max(z_new)

print(min_prob, max_prob)
norm = colors.Normalize(min_prob, max_prob)

fig = plt.figure(figsize=(10, 10), dpi=1000)
ax = fig.add_subplot(111)

ax.invert_yaxis()

# 0814 contours
manual_locations = [(10.0, -17.0), (8.0, -20.0), (5.0, -21.0), (3.0, -21), (0.0, -22.0)]
ax.contourf(xx, yy, z_new, levels=np.linspace(0.0, max_prob, 200), cmap=plt.cm.viridis)

CS = ax.contour(xx, yy, z_new, levels=[0.0, 0.10, 0.30, 0.5, 0.7, 0.9, max_prob], colors="red")
ax.clabel(CS, inline=1, fontsize=24, manual=manual_locations, fmt="%0.1f")

# for i in range(5):
#     p = CS.collections[i].get_paths()[1]
#     v = p.vertices
#     x = v[:,0]
#     y = v[:,1]
#
#     upper_bound = -19.0
#     # upper_bound = -17.8
#     # lower_bound = -15.0
#     lower_bound = -11.8
#
#     model_y1 = list(np.linspace(y.min(), upper_bound, 15))
#     # model_y2 = list(np.linspace(upper_bound, lower_bound, 8))
#     model_y2 = list(np.linspace(upper_bound, lower_bound, 12))
#     model_y3 = list(np.linspace(lower_bound, y.max(), 4))
#
#     y1 = y[np.where(y <= upper_bound)[0]]
#     x1 = x[np.where(y <= upper_bound)[0]]
#
#     y2 = y[(y <= lower_bound) & (y > upper_bound)]
#     x2 = x[(y <= lower_bound) & (y > upper_bound)]
#
#     y3 = y[np.where(y > lower_bound)[0]]
#     x3 = x[np.where(y > lower_bound)[0]]
#
#     tks_x1 = interpolate.splrep(y1, x1, s=1)
#     tks_x2 = interpolate.splrep(y2, x2, s=0.02)
#     tks_x3 = interpolate.splrep(y3, x3, s=1)
#
#     new_x1 = interpolate.splev(model_y1, tks_x1, der=0)
#     new_x2 = interpolate.splev(model_y2, tks_x2, der=0)
#     new_x3 = interpolate.splev(model_y3, tks_x3, der=0)
#
#     xx1 = list(new_x1)
#     xx2 = list(new_x2)
#     xx3 = list(new_x3)
#     xxx = []
#     xxx += xx1
#     xxx += xx2
#     xxx += xx3
#
#     yyy = []
#     yyy += model_y1
#     yyy += model_y2
#     yyy += model_y3
#
#     model_y = np.linspace(y.min(), y.max(), 1000)
#     tks_final = interpolate.splrep(yyy, xxx, s=50)
#     model_x = interpolate.splev(model_y, tks_final, der=0)
#
#     # ax.plot(test1, xnew1, 'b-', zorder=9999, linewidth=1.0)
#     # ax.plot(test2, xnew2, 'b-', zorder=9999, linewidth=1.0)
#     # ax.plot(test3, xnew3, 'g-', zorder=9999, linewidth=1.0)
#     # ax.plot(model_x, model_y, 'k-', zorder=9999, linewidth=1.0)
#     ax.plot(xxx,yyy,'r', linewidth=3.0) #zorder=9999,

# transient_legend = {
#     "Ia":False,
#     "Ia-91bg":False,
#     "II":False,
#     "Ia-91T":False,
#     "SLSN-I":False,
#     "IIb":False,
#     "Ic":False,
#     "Ib":False,
#     "IIn":False
# }
# for t in transients:
#     if not transient_legend[t[3]]:
#         transient_legend[t[3]] = True
#         ax.plot(t[1], t[0], "*", markerfacecolor=t[2], markeredgecolor="None", markersize=12, alpha=0.75, label=t[3])
#     else:
#         ax.plot(t[1], t[0], "*", markerfacecolor=t[2], markeredgecolor="None", markersize=12, alpha=0.75)

sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
sm.set_array([]) # can be an empty list
tks = np.linspace(min_prob, max_prob, 5)

cb = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.02, alpha=0.80) # fraction=0.04875

# cb.set_ticks([0.0, 0.944] + list(tks))
cb.set_ticks(tks)
tks_strings = ["0%", "25%", "50%", "70%", "95%"]
cb.ax.set_yticklabels(tks_strings, fontsize=24)
cb.set_label("", fontsize=16, labelpad=9.0)
cb.ax.tick_params(length=6.0) # width=2.0,
cb.ax.locator_params(nbins=5)

# SSS17a
# dm7 = 0.56
# Mabs = -16.56
ax.plot(0.56, -16.56, '*', color='yellow', markeredgecolor='yellow', markersize=24) #markeredgecolor="black" , alpha=0.25

# ax.annotate('0.9', xy=(0.4, -18.1), fontsize=24, color='red')
# ax.annotate('0.7', xy=(0.4, -17.4), fontsize=24, color='red')
# ax.annotate('0.5', xy=(0.4, -16.8), fontsize=24, color='red')
# ax.annotate('0.3', xy=(0.4, -16.4), fontsize=24, color='red')
# ax.annotate('0.1', xy=(0.4, -15.9), fontsize=24, color='red')
# ax.annotate('.9', xy=(-0.4, -13.0), fontsize=24, color='red')
# ax.annotate('.7', xy=(-0.16, -13.8), fontsize=24, color='red')
# ax.annotate('.5', xy=(-0.02, -14.4), fontsize=24, color='red')
# ax.annotate('.3', xy=(0.08, -14.8), fontsize=24, color='red')
# ax.annotate('.1', xy=(0.18, -15.1), fontsize=24, color='red')

ax.tick_params(axis='both', which='major', labelsize=24)

plt.xlabel(r'$\Delta$M $\mathrm{day^{-1}}$',fontsize=32)
plt.ylabel(r'$\mathrm{M_{0}}$',fontsize=32)

fig.savefig('190814_Linear_Prob2Detect_all.png', bbox_inches='tight')
plt.close('all')
print("... Done.")
