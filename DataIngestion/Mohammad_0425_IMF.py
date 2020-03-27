import matplotlib

# region Imports
matplotlib.use("Agg")

import os


import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.pyplot import cm
from matplotlib.patches import CirclePolygon
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys

sys.path.append('../')

import optparse

from configparser import RawConfigParser
# import multiprocessing as mp
# # from multiprocessing import get_context
import mysql.connector

import mysql.connector as test
# print(test.__version__)

from mysql.connector.constants import ClientFlag
from mysql.connector import Error
import csv
import time
import pickle
from collections import OrderedDict

import numpy as np
from scipy.special import erf
from scipy.optimize import minimize, minimize_scalar
import scipy.stats as st
from scipy.integrate import simps, quad, trapz
from scipy.interpolate import interp1d, interp2d

import astropy as aa
from astropy import cosmology
from astropy.cosmology import WMAP5, WMAP7, LambdaCDM
from astropy.coordinates import Distance
from astropy.coordinates.angles import Angle
from astropy.cosmology import z_at_value
from astropy import units as u
import astropy.coordinates as coord
from dustmaps.config import config
from dustmaps.sfd import SFDQuery

import shapely as ss
from shapely.ops import transform as shapely_transform
from shapely.geometry import Point
from shapely.ops import linemerge, unary_union, polygonize, split
from shapely import geometry

import healpy as hp
from ligo.skymap import distance

from HEALPix_Helpers import *
from Tile import *
from SQL_Polygon import *
from Pixel_Element import *
from Completeness_Objects import *

import psutil
import shutil
import urllib.request
import requests
from bs4 import BeautifulSoup
from dateutil.parser import parse

import glob
import gc
import json

import MySQLdb as my
from collections import OrderedDict
from scipy.integrate import simps

from astropy.table import Table
import pdb
import re
from scipy.stats import norm
from scipy.optimize import curve_fit


blue = [1.1562043795620427, 4.160583941605839, 2.8029197080291968]
green = [5.185401459854013, 4.545985401459854, 2.522627737226277,
         2.5489051094890507, 2.2686131386861312, 1.0072992700729926, 0.490510948905109]
purple = [4.581021897810218, 5.465693430656934, 2.82919708029197, 4.712408759124087,
          1.4890510948905105, 3.8189781021897797, 2.4087591240875907, 1.5240875912408756,
          0.4291970802919702, 2.233576642335766, 1.401459854014598, 0.24525547445255397,
          0.21897810218978017, 0.13138686131386873]
cyan = [4.493430656934306, 1.2875912408759116, 3.652554744525547, 2.25985401459854, 1.331386861313868]
red = [3.740145985401459, 0.3766423357664227, 1.7080291970802914, 1.6817518248175176, 1.2525547445255467,
       0.29781021897810145, 1.1562043795620427]


adjusted_green = [green[0] - blue[2],
                  green[1],
                  green[2],
                  green[3],
                  green[4],
                  green[5],
                  green[6]]

adjusted_red = [red[0] - adjusted_green[4],
       red[1],
       red[2] - adjusted_green[5],
       red[3] - adjusted_green[6],
       red[4],
       red[5],
       red[6]]

adjusted_cyan = [cyan[0] - adjusted_red[0] - adjusted_green[4],
        cyan[1] - adjusted_red[1],
        cyan[2] - adjusted_red[2] - adjusted_green[5],
        cyan[3] - adjusted_red[3] - adjusted_green[6],
        cyan[4] - adjusted_red[6]
        ]

adjusted_purple = [purple[0] - blue[1],
                   purple[1] - blue[2] - adjusted_green[0],
                   purple[2] - adjusted_green[3],
                   purple[3] - adjusted_cyan[0] - adjusted_red[0] - adjusted_green[4],
                   purple[4] - adjusted_cyan[1] - adjusted_red[1],
                   purple[5] - adjusted_cyan[2] - adjusted_red[2] - adjusted_green[5],
                   purple[6] - adjusted_cyan[3] - adjusted_red[3] - adjusted_green[6],
                   purple[7] - adjusted_red[4],
                   purple[8] - adjusted_red[5],
                   purple[9] - adjusted_cyan[4] - adjusted_red[6],
                   purple[10],
                   purple[11],
                   purple[12],
                   purple[13]]








blue_x = [1.3754310344827587, 1.4064655172413794, 1.4685344827586206]
green_x = [1.4685344827586206, 1.4995689655172413, 1.5306034482758621,
           1.5616379310344828, 1.5926724137931034, 1.654741379310345,
           1.6867456896551725]
purple_x = [1.4064655172413794, 1.4685344827586206, 1.5616379310344828, 1.5926724137931034, 1.623706896551724,
            1.654741379310345, 1.6867456896551725, 1.717780172413793, 1.7488146551724137, 1.7798491379310346,
            1.8108836206896552, 1.841918103448276, 1.8729525862068965, 1.935991379310345]
cyan_x = [1.5926724137931034, 1.623706896551724, 1.654741379310345, 1.6867456896551725, 1.7788793103448275]
red_x = [1.5936422413793103, 1.623706896551724, 1.6557112068965516, 1.6867456896551725, 1.717780172413793,
         1.7488146551724137, 1.7798491379310346]



all_x = [1.3754310344827587, 1.4064655172413794, 1.4685344827586206, 1.4995689655172413, 1.5306034482758621,
           1.5616379310344828, 1.623706896551724, 1.654741379310345, 1.6867456896551725, 1.717780172413793,
         1.7488146551724137, 1.7798491379310346, 1.8108836206896552, 1.841918103448276, 1.8729525862068965, 1.935991379310345]

# fig = plt.figure(figsize=(10,10), dpi=800)
# ax = fig.add_subplot(111)
#
# ax.plot(blue_x, blue, 'bs')
# ax.plot(green_x, green, 'gs')
# ax.plot(purple_x, purple, color='purple', marker='s', linestyle='None')
# ax.plot(cyan_x, cyan, 'cs')
# ax.plot(red_x, red, 'rs')
#
#
# fig.savefig("test_hist.png", bbox_inches='tight')
# plt.close('all')



# fig = plt.figure(figsize=(10,10), dpi=800)
# ax = fig.add_subplot(111)

def gaus(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

x = np.linspace(1.1,2.1,1000)


n1 = len(blue_x)
mean1 = sum(np.asarray(blue_x) * np.asarray(blue))/n1
sigma1 = sum(blue * (blue_x - mean1)**2)/n1
popt1, pcov1 = curve_fit(gaus, blue_x, blue, p0=[1, mean1, sigma1])
# ax.plot(blue_x, blue,'bs',label=r'$9M_{\odot} < M < 10M_{\odot}$')
# ax.plot(x, gaus(x, *popt1),'b:')

n2 = len(green_x)
mean2 = sum(np.asarray(green_x) * np.asarray(adjusted_green))/n2
sigma2 = sum(adjusted_green * (green_x - mean2)**2)/n2
popt2, pcov2 = curve_fit(gaus, green_x, adjusted_green, p0=[1, mean2, sigma2])
# ax.plot(green_x, adjusted_green,'gs',label=r'$10M_{\odot} < M < 13M_{\odot}$')
# ax.plot(x, gaus(x, *popt2),'g:')

_red_x = red_x[1:-2]
_adjusted_red = adjusted_red[1:-2]
n3 = len(_red_x)
mean3 = sum(np.asarray(_red_x) * np.asarray(_adjusted_red))/n3
sigma3 = sum(_adjusted_red * (_red_x - mean3)**2)/n3
popt3, pcov3 = curve_fit(gaus, _red_x, _adjusted_red, p0=[1, mean3, sigma3])
# ax.plot(red_x, adjusted_red,'rs',label=r'$13M_{\odot} < M < 15M_{\odot}$')
# ax.plot(x, gaus(x, *popt3),'r:')

n4 = len(cyan_x)
mean4 = sum(np.asarray(cyan_x) * np.asarray(adjusted_cyan))/n4
sigma4 = sum(adjusted_cyan * (cyan_x - mean4)**2)/n4
popt4, pcov4 = curve_fit(gaus, cyan_x, adjusted_cyan, p0=[1, mean4, sigma4])
# ax.plot(cyan_x, adjusted_cyan,'cs',label=r'$15M_{\odot} < M < 18M_{\odot}$')
# ax.plot(x, gaus(x, *popt4),'c:')

n5 = len(purple_x)
mean5 = sum(np.asarray(purple_x) * np.asarray(adjusted_purple))/n5
sigma5 = sum(adjusted_purple * (purple_x - mean5)**2)/n5
popt5, pcov5 = curve_fit(gaus, purple_x, adjusted_purple, p0=[1, mean5, sigma5])
# ax.plot(purple_x, adjusted_purple, color='purple', marker='s', linestyle='None',
#         label=r'$18M_{\odot} < M < 120M_{\odot}$')
# ax.plot(x, gaus(x, *popt5), color='purple', linestyle=':')


# ax.set_ylabel("PDF")
# ax.set_xlabel("Baryonic Mass")
# ax.legend()
#
#
# fig.savefig("test_hist2.png", bbox_inches='tight')
# plt.close('all')







# fig = plt.figure(figsize=(10,10), dpi=800)
# ax = fig.add_subplot(111)
#
# ax.plot(x, norm(popt1[1], popt1[2]).pdf(x), 'b:')
# ax.plot(x, norm(popt2[1], popt2[2]).pdf(x), 'g:')
# ax.plot(x, norm(popt3[1], popt3[2]).pdf(x), 'r:')
# ax.plot(x, norm(popt4[1], popt4[2]).pdf(x), 'c:')
# ax.plot(x, norm(popt5[1], popt5[2]).pdf(x), color='purple', linestyle=':')
#
# fig.savefig("test_hist3.png", bbox_inches='tight')
# plt.close('all')



m2 = np.linspace(1.2,1.7,100)
m1 = 3.4 - m2

m2_high = np.linspace(1.3,1.8,100)
m1_high = 3.5 - m2

m2_low = np.linspace(1.1,1.6,100)
m1_low = 3.3 - m2


# fig = plt.figure(figsize=(10,10), dpi=800)
# ax = fig.add_subplot(111)
#
#
# ax.plot(m1_high, m2_high, 'k:')
# ax.plot(m1, m2, 'k-')
# ax.plot(m1_low, m2_low, 'k:')
#
# ax.set_xlabel(r"m1")
# ax.set_ylabel(r"m2")
# fig.savefig("test_hist4.png", bbox_inches='tight', linestyle=':')
# plt.close('all')

imf = lambda m: m**(-2.35)
def prob_m(m, delta_m):
    low = m - delta_m
    high = m + delta_m

    x1 = norm(mean1, sigma1).cdf(high) - norm(mean1, sigma1).cdf(low)
    x2 = norm(mean2, sigma2).cdf(high) - norm(mean2, sigma2).cdf(low)
    x3 = norm(mean3, sigma3).cdf(high) - norm(mean3, sigma3).cdf(low)
    x4 = norm(mean4, sigma4).cdf(high) - norm(mean4, sigma4).cdf(low)
    x5 = norm(mean5, sigma5).cdf(high) - norm(mean5, sigma5).cdf(low)

    imf_norm = quad(imf, 9, 120)[0]

    p1 = (quad(imf, 9, 10)[0] / imf_norm * x1) * (10 - 9)
    p2 = quad(imf, 10, 13)[0] / imf_norm * x2 * (13 - 10)
    p3 = quad(imf, 13, 15)[0] / imf_norm * x3 * (15 - 13)
    p4 = quad(imf, 15, 18)[0] / imf_norm * x4 * (18 - 15)
    p5 = quad(imf, 18, 120)[0] / imf_norm * x5 * (120 - 18)

    return p1 + p2 + p3 + p4 + p5

dm_arr = np.linspace(0.01, 0.2, 25)
for i, dm in enumerate(dm_arr):

    z = {}
    prob_list = []
    for j, m_1 in enumerate(m1):
        m2_arr = np.linspace(m2_low[j], m2_high[j], 10)
        z[m_1] = {}
        for k, m_2 in enumerate(m2_arr):
            p = prob_m(m_1, dm) * prob_m(m_2, dm)
            prob_list.append(p)
            z[m_1][m_2] = p

    min_prob = np.min(prob_list)
    max_prob = np.max(prob_list)
    clr_norm = colors.Normalize(min_prob, max_prob)

    fig = plt.figure(figsize=(8,8), dpi=180)
    ax = fig.add_subplot(111)

    for j, (key1, value1) in enumerate(z.items()):
        for k, (key2, value2)  in enumerate(value1.items()):
            clr = plt.cm.viridis(clr_norm(value2))
            ax.plot(key1, key2, 's', color=clr, markeredgecolor=clr, markersize=14)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    ax.tick_params(axis='both', which='major', labelsize=20, length=6.0, width=2.0)
    ax.annotate("dm = %0.2f" % dm, xy=(2.0, 1.7), xycoords='data', fontsize=20, zorder=9999)
    ax.set_xlabel(r"m1 $\mathrm{[M_{\odot}]}$", fontsize=20)
    ax.set_ylabel(r"m2 $\mathrm{[M_{\odot}]}$", fontsize=20)

    sm = plt.cm.ScalarMappable(norm=clr_norm, cmap=plt.cm.viridis)
    sm.set_array([])  # can be an empty list

    tks_str = np.linspace(0.0, 1.0, 10)
    tks = np.linspace(min_prob, max_prob, 10)
    tks_strings = []
    for t in tks_str:
        tks_strings.append('%0.1f' % t)

    cb = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.02) #, alpha=0.80
    cb.ax.tick_params(length=6.0, width=2.0)
    cb.set_ticks(tks)
    cb.ax.set_yticklabels(tks_strings, fontsize=20)
    cb.outline.set_linewidth(2)

    ax.xaxis.grid(color='gray', linestyle=':', zorder=0, alpha=0.4)
    ax.yaxis.grid(color='gray', linestyle=':', zorder=0, alpha=0.4)

    fig.savefig("IMF_Weighted_BNS/m1_m2_%s.png" % str(i).zfill(3), bbox_inches='tight')
    plt.close('all')




print("... Done.")