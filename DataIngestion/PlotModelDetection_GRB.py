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



grb_axis = "onaxis"
# grb_axis = "offaxis"


# Unpack unordered results
E_keys = []
E_n = OrderedDict()
E_prob = OrderedDict()
all_probs = []
for i in range(10):
    sub_dir = i+1
    results_table = Table.read("../Events/S190814bv/ModelDetection/Detection_Results_grb_%s_%s.prob" %
                               (grb_axis, sub_dir), format='ascii.ecsv')

    E_result = list(results_table['E'])
    n_result = list(results_table['n'])
    prob_result = list(results_table['Prob'])

    all_probs += prob_result

    for j, E in enumerate(E_result):
        if E not in E_n:
            E_n[E] = []
        if E not in E_prob:
            E_prob[E] = []
        if E not in E_keys:
            E_keys.append(E)

        E_n[E].append(n_result[j])
        E_prob[E].append(prob_result[j])

n_rows = OrderedDict()
prob_rows = OrderedDict()
for E, n_list in E_n.items():

    sorted_indices = np.argsort(E_n[E]) # sorted indices of n_list
    n_sorted = np.asarray(E_n[E])[sorted_indices]
    prob_sorted = np.asarray(E_prob[E])[sorted_indices]

    n_rows[E] = n_sorted
    prob_rows[E] = prob_sorted

Sorted_E = sorted(E_keys)
Sorted_E_n = OrderedDict()
Sorted_E_prob = OrderedDict()

for E in Sorted_E:
    Sorted_E_n[E] = n_rows[E]
    Sorted_E_prob[E] = prob_rows[E]



model_E_input = np.logspace(-2.0, 2.0, 100)
model_n_input = np.logspace(-6.0, 0.0, 100)

# Interpolate rows
Interp_E_prob = OrderedDict()
for E in Sorted_E:
    n = Sorted_E_n[E]
    p = prob_rows[E]
    test_f = interp1d(n, p, kind="slinear")
    Interp_E_prob[E] = test_f(model_n_input)

# # Stand-alone interpolate columns
# Interp_n_prob = OrderedDict()
# nlist = Sorted_E_n[E_keys[0]]
# for i, n in enumerate(nlist):
#     p = []
#     for E in E_keys:
#         p.append(Sorted_E_prob[E][i])
#
#     test_f = interp1d(E_keys, p, kind="slinear")
#     Interp_n_prob[n] = test_f(model_E_input)

# row-dependent interpolate columns
Interp_n_prob = OrderedDict()
all_interp_probs = []
for i, n in enumerate(model_n_input):

    column_probs = []
    sorted_Es = []

    for j, E in enumerate(Interp_E_prob.keys()):
        sorted_Es.append(E)
        column_probs.append(Interp_E_prob[E][i])

    test_f = interp1d(sorted_Es, column_probs, kind="slinear")
    interp_probs = test_f(model_E_input)
    Interp_n_prob[n] = interp_probs

all_interp_probs = []
for n,n_probs in Interp_n_prob.items():
    for p in n_probs:
        all_interp_probs.append(p)


# print(np.min(all_interp_probs), np.max(all_interp_probs))
# min_prob = np.min(all_probs)
# max_prob = np.max(all_probs)
# print(min_prob, max_prob)






model_tuples = []
for i, (n, prob_col) in enumerate(Interp_n_prob.items()):
    for j, p in enumerate(prob_col):
        model_tuples.append((n, model_E_input[j], p))

X = []
Y = []
Z = []
for mt in model_tuples:
    X.append(mt[0])
    Y.append(mt[1])
    Z.append(mt[2])


log_x = np.log10(X)
log_y = np.log10(Y)
_2d_func = interp2d(log_x, log_y, Z, kind="linear")
log_xx = np.log10(model_n_input)
log_yy = np.log10(model_E_input)
xx, yy = np.meshgrid(model_n_input, model_E_input)
yy = yy/10.0 # change to FOE
# z_new = _2d_func(model_n_input, model_E_input)
z_new = _2d_func(log_xx, log_yy)

print(np.min(z_new), np.max(z_new))



print("\n\nDone with interp\n\n")
# raise Exception("Stop")



# min_prob = np.min(z_new)
# max_prob = np.max(z_new)
# norm = colors.Normalize(min_prob, max_prob)

# min_prob = np.min(all_probs)
# max_prob = np.max(all_probs)
# norm = colors.Normalize(min_prob, max_prob)
min_prob = np.min(Z)
max_prob = np.max(Z)
norm = colors.Normalize(min_prob, max_prob)


fig = plt.figure(figsize=(10,10), dpi=1000)
ax = fig.add_subplot(111)

# By Row
# for E in Sorted_E:
#
#     probs = Interp_E_prob[E]
#     for i, p in enumerate(probs):
#         clr = plt.cm.viridis(norm(p))
#         ax.plot(model_n_input[i], E, 's', color=clr, markeredgecolor=clr, markersize=17.5)
#
#     # n_list = Sorted_E_n[E]
#     # prob_list = Sorted_E_prob[E]
#
#     # for i, p in enumerate(prob_list):
#     #     clr = plt.cm.viridis(norm(p))
#     #     ax.plot(n_list[i], E, 's', color=clr, markeredgecolor=clr, markersize=17.5)


# # By Column
# nlist = Sorted_E_n[E_keys[0]]
# for n in nlist:
#     probs = Interp_n_prob[n]
#     for i, p in enumerate(probs):
#         clr = plt.cm.viridis(norm(p))
#         ax.plot(n, model_E_input[i], 's', color=clr, markeredgecolor=clr, markersize=17.5)
#
#
# test_E = Sorted_E[15]
# test_n_list = Sorted_E_n[test_E]
# test_prob_list = Sorted_E_prob[test_E]
# for n in test_n_list:
#     ax.plot(n, test_E, 's', color='None', markeredgecolor='r', markersize=17.5)



print("plotting...")
# # By Column
# for i, n in enumerate(model_n_input):
#     n_probs = Interp_n_prob[n]
#     for j, prob in enumerate(n_probs):
#         clr = plt.cm.viridis(norm(prob))
#         ax.plot(n, model_E_input[j], 's', color=clr, markeredgecolor=clr, markersize=17.5)

# for mt in model_tuples:
#     clr = plt.cm.viridis(norm(mt[2]))
#     ax.plot(mt[0], mt[1], 's', color=clr, markeredgecolor=clr, markersize=17.5)


CS_filled = ax.contourf(xx, yy, z_new, cmap=plt.cm.viridis, levels=np.linspace(0.0, max_prob, 200))
CS_lines = ax.contour(xx, yy, z_new, colors="red", levels=[0.0, 0.10, 0.30, 0.5, 0.7, 0.9])

if grb_axis == "onaxis":

    # load GRBs
    grbs = []
    class sGRB:
        def __init__(self, name, Eiso, Eiso_err_upper, Eiso_err_lower, n, n_err_upper, n_err_lower):
            self.name = name
            self.Eiso = Eiso
            self.Eiso_err_upper = Eiso_err_upper
            self.Eiso_err_lower = Eiso_err_lower
            self.n = n
            self.n_err_upper = n_err_upper
            self.n_err_lower = n_err_lower


    with open("../GRBModels/fong_2015_table_3.txt", 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)  # skip header

        for row in csvreader:
            # print(row)
            name = row[0]

            # Eiso from Fong et al 2015 in units of 10^52 ergs -- convert to FOE
            eiso = float(row[1])*10.0
            eiso_err_upper = float(row[2])*10.0
            eiso_err_lower = float(row[3])*10.0

            n = float(row[4])
            n_err_upper = float(row[5])
            n_err_lower = float(row[6])

            grbs.append(sGRB(name, eiso, eiso_err_upper, eiso_err_lower, n, n_err_upper, n_err_lower))

    for g in grbs:

        if g.n > 1e-5 and g.n < 1.0 and g.Eiso > 1e-3 and g.Eiso < 9.0:
            xerror = [[g.n_err_lower], [g.n_err_upper]]
            yerror = [[g.Eiso_err_lower], [g.Eiso_err_upper]]
            ax.errorbar(x=g.n, y=g.Eiso, xerr=xerror, yerr=yerror, fmt='D', color='black', ms=10)

    ax.errorbar(x=1e-6, y=0, fmt='D', color='black', label="SGRBs from\nFong et al. 2015", ms=16)

    manual_locations = [(1e-2, 1e-2), (4e-2, 1.2e-2), (1.2e-1, 1.3e-2),  (2e-1, 2e-2), (6e-1, 4e-2)]
    ax.clabel(CS_lines, inline=False, fontsize=24, fmt="%0.1f", manual=manual_locations)

    ax.annotate(r'$\mathrm{\theta_{obs}=0\degree}$', xy=(3.0e-6, 2e-3), xycoords="data", fontsize=40, color='white')

elif grb_axis == "offaxis":
    # manual_locations = [(1e-4, 1.1), (3e-4, 1.0), (6e-4, 7e-1), (1e-3, 6e-1), (4e-3, 1e-1)]
    # ax.clabel(CS_lines, inline=False, fontsize=24, fmt="%0.1f", manual=manual_locations, rightside_up=True)
    ax.text(7e-5, 1.1, "0.1", fontsize=24, color='red', rotation=-45)  # rotation is counter-clockwise
    ax.text(2e-4, 8e-1, "0.3", fontsize=24, color='red', rotation=-45)  # rotation is counter-clockwise
    ax.text(6e-4, 4.6e-1, "0.5", fontsize=24, color='red', rotation=-45)  # rotation is counter-clockwise
    ax.text(2e-3, 2.5e-1, "0.7", fontsize=24, color='red', rotation=-45)  # rotation is counter-clockwise
    ax.text(6e-3, 2e-1, "0.9", fontsize=24, color='red', rotation=-45)  # rotation is counter-clockwise

    ax.annotate(r'$\mathrm{\theta_{obs}=10\degree}$', xy=(3.0e-5, 2e-3), xycoords="data", fontsize=40, color='white')



ax.set_yscale('log')
ax.set_xscale('log')

sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
sm.set_array([]) # can be an empty list

tks = np.linspace(min_prob, max_prob, 5)
# tks = np.linspace(np.min(prob), np.max(prob), 5)
# tks = np.logspace(np.log10(min_prob), np.log10(max_prob), 9)

# tks_strings = []
# for t in tks:
#     tks_strings.append('%0.2f' % (t * 100))
tks_strings = ["0%", "25%", "50%", "70%", "95%"]

cb = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.02) #, alpha=0.80
cb.set_ticks(tks)
cb.ax.tick_params(length=6.0) # width=2.0,

cb.ax.set_yticklabels(tks_strings, fontsize=32)
cb.set_label("", fontsize=16, labelpad=9.0)
cb.ax.tick_params(length=12.0, width=2) # width=2.0,
cb.ax.locator_params(nbins=5)
# cb.outline.set_linewidth(2.0)

ax.margins(x=0,y=0)
ax.tick_params(axis='both', which='major', labelsize=24, length=12.0, width=2)
ax.tick_params(axis='both', which='minor', labelsize=24, length=8.0, width=2)

ax.set_xlim([1e-5, 1.0])
ax.set_ylim([1e-3, 1e1])

plt.xlabel(r'n $\mathrm{\left(cm^{-3}\right)}$',fontsize=40)
plt.ylabel(r'$\mathrm{E_{K,iso}}$ $\left(\times 10^{51} \mathrm{ergs}\right)}$',fontsize=40)
plt.legend(loc="lower left", framealpha=1.0, fontsize=24, borderpad=0.2, handletextpad=0.2)




fig.savefig('GW190814_GRB_Prob2Detect_%s.png' % grb_axis, bbox_inches='tight')
plt.close('all')



#
#
# model_n_input = np.logspace(-6.0, 0.0, 100)
# test_f = interp1d(test_n_list, test_prob_list, kind="slinear")
# model_probs = test_f(model_n_input)
#
# fig = plt.figure(figsize=(10,10), dpi=1000)
# ax = fig.add_subplot(111)
# for i, n in enumerate(test_n_list):
#     ax.plot(n, test_prob_list[i], marker='s', markerfacecolor='None', markeredgecolor='k', markersize=17.5)
#
# for i, mp in enumerate(model_probs):
#     ax.plot(model_n_input[i], mp, 'b+', markersize=12.0)
#
#
#
#
# ax.margins(x=0,y=0)
# ax.tick_params(axis='both', which='major', labelsize=24, length=12.0, width=2)
# ax.set_xscale('log')
# plt.xlabel(r'n $\mathrm{[cm^{-3}]}$',fontsize=32)
# plt.ylabel('Probability',fontsize=32)
# fig.savefig('190814_GRB_slice.png', bbox_inches='tight')


plt.close('all')
print("... Done.")