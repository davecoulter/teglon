import matplotlib

matplotlib.use("Agg")

import os

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
import multiprocessing as mp
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
from scipy.integrate import simps, quad
from scipy.interpolate import interp2d

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


# abs_mag_array = np.linspace(-9.0, -21.0, 30)
# dm_list_of_list = []
# for am in abs_mag_array:
#     dm = []
#     if am <= -15.0:
#         dm1 = [-5.0, -4.0, -3.0]
#         dm2 = list(np.linspace(-2.0, 1.0, 20))
#         dm3 = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
#         dm += dm1
#         dm += dm2
#         dm += dm3
#
#         dm_list_of_list.append(dm)
#     else:
#         dm1 = [-5.0, -4.0, -3.0, -2.0]
#         dm2 = list(np.linspace(-1.0, 10, 26))
#         dm += dm1
#         dm += dm2
#
#         dm_list_of_list.append(dm)


abs_mag_array = np.linspace(-11.0, -23.0, 40)
dm_list_of_list = []
for am in abs_mag_array:
    dm = []
    if am > -17.0:
        dm1 = [-2.0]
        dm2 = list(np.linspace(-1.5, 1.5, 31))
        dm3 = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
        dm += dm1
        dm += dm2
        dm += dm3
    else:
        dm1 = [-2.0, -1.5]
        dm2 = list(np.linspace(-1.0, 10, 38))
        dm += dm1
        dm += dm2

    print(len(dm))
    dm_list_of_list.append(dm)



# dm_array = np.linspace(-5.0, 10.0, 30)
time_array = np.linspace(0.00, 40.0, 400) # in days

# abs_mag_array = np.linspace(-16.0, -20.0, 40)
# dm_array = np.linspace(-1.0, 8.0, 40)
# time_array = np.linspace(0.00, 26.0, 251) # in days

abs_lcs_array = {}
for dm_index, abs_mag in enumerate(abs_mag_array):
    for dm in dm_list_of_list[dm_index]:
        key = (abs_mag, dm)
        lc = abs_mag + time_array * dm
        abs_lcs_array[key] = lc

print("Number of generated models: %s" % len(abs_lcs_array))


cols = ['time', 'sdss_g', 'sdss_r', 'sdss_i', 'Clear']
dtype = ['f8', 'f8', 'f8', 'f8', 'f8']

j = 0
for i, (key, lc) in enumerate(abs_lcs_array.items()):

    current_sub_dir = "../LinearModels/%s" % j
    if i % 100 == 0:
        j += 1
        current_sub_dir = "../LinearModels/%s" % j
        os.mkdir(current_sub_dir)

    abs_mag = "%0.3f" % key[0]
    dm = "%0.3f" % key[1]

    meta = ["{key}={value}".format(key="M", value=key[0]), "{key}={value}".format(key="dM", value=key[1])]

    result_table = Table(dtype=dtype, names=cols)
    result_table.meta['comment'] = meta

    print("Writing linear model: (%s, %s)" % key)
    for i, epoch in enumerate(lc):
        result_table.add_row([time_array[i], epoch, epoch, epoch, epoch])

    result_table.write("%s/%s_%s.dat" % (current_sub_dir, abs_mag.replace(".", "_"),
                                                      dm.replace(".", "_")), overwrite=True, format='ascii.ecsv')



