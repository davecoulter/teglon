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
# import ray

import sys

sys.path.append('../')

import optparse

from configparser import RawConfigParser
import multiprocessing as mp
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
from scipy.stats import norm

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
from scipy import stats

from astropy.table import Table
import pdb
import re
# endregion

# region Config Settings
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
configFile = os.path.join(__location__, 'Settings.ini')

db_config = RawConfigParser()
db_config.read(configFile)

db_name = db_config.get('database', 'DATABASE_NAME')
db_user = db_config.get('database', 'DATABASE_USER')
db_pwd = db_config.get('database', 'DATABASE_PASSWORD')
db_host = db_config.get('database', 'DATABASE_HOST')
db_port = int(db_config.get('database', 'DATABASE_PORT'))

isDEBUG = False

# Set up dustmaps config
config["data_dir"] = "./"

# Generate all pixel indices
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
GW190814_t_0 = 58709.882824224536  # time of GW190814 merger


# endregion

# region DB CRUD
# Database SELECT
# For every sub-query, the iterable result is appended to a master list of results
def bulk_upload(query):
    success = False
    try:

        conn = mysql.connector.connect(user=db_user, password=db_pwd, host=db_host, port=db_port, database=db_name)
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        success = True

    except Error as e:
        print("Error in uploading CSV!")
        print(e)
    finally:
        cursor.close()
        conn.close()

    return success


def query_db(query_list, commit=False):
    # query_string = ";".join(query_list)

    results = []
    try:
        chunk_size = 1e+6

        db = my.connect(host=db_host, user=db_user, passwd=db_pwd, db=db_name, port=db_port)
        cursor = db.cursor()

        for q in query_list:
            cursor.execute(q)

            if commit:  # used for updates, etc
                db.commit()

            streamed_results = []
            print("fetching results...")
            while True:
                r = cursor.fetchmany(1000000)
                count = len(r)
                streamed_results += r
                size_in_mb = sys.getsizeof(streamed_results) / 1.0e+6

                print("\tfetched: %s; current length: %s; running size: %0.3f MB" % (
                    count, len(streamed_results), size_in_mb))

                if not r or count < chunk_size:
                    break

        results.append(streamed_results)

    # cnx = mysql.connector.connect(user=db_user, password=db_pwd, host=db_host, port=db_port, database=db_name)
    # cursor = cnx.cursor()
    # for result in cursor.execute(query_string, multi=True):

    # 	streamed_results = []
    # 	print("fetching results...")

    # 	i = 0
    # 	while True:
    # 		i += chunk_size
    # 		print("Fetching: %s records" % i)

    # 		partial_result = result.fetchmany(chunk_size)
    # 		count = len(partial_result)
    # 		streamed_results += partial_result
    # 		size_in_mb = sys.getsizeof(streamed_results)/1.0e+6

    # 		print("\tfetched: %s; current length: %s; running size: %0.3f MB" % (count, len(streamed_results), size_in_mb))

    # 		if not partial_result or count < chunk_size:
    # 			break

    # 	results.append(streamed_results)

    except Error as e:
        print('Error:', e)
    finally:
        cursor.close()
        # cnx.close()
        db.close()

    # fake = [[[(1)]]]
    return results


# return fake

def batch_query(query_list):
    return_data = []
    batch_size = 500
    ii = 0
    jj = batch_size
    kk = len(query_list)

    print("\nLength of data to query: %s" % kk)
    print("Query batch size: %s" % batch_size)
    print("Starting loop...")

    number_of_queries = len(query_list) // batch_size
    if len(query_list) % batch_size > 0:
        number_of_queries += 1

    query_num = 1
    payload = []
    while jj < kk:
        t1 = time.time()

        print("%s:%s" % (ii, jj))
        payload = query_list[ii:jj]
        return_data += query_db(payload)

        ii = jj
        jj += batch_size
        t2 = time.time()

        print("\n********* start DEBUG ***********")
        print("Query %s/%s complete - execution time: %s" % (query_num, number_of_queries, (t2 - t1)))
        print("********* end DEBUG ***********\n")

        query_num += 1

    print("Out of loop...")

    t1 = time.time()

    print("\n%s:%s" % (ii, kk))

    payload = query_list[ii:kk]
    return_data += query_db(payload)

    t2 = time.time()

    print("\n********* start DEBUG ***********")
    print("Query %s/%s complete - execution time: %s" % (query_num, number_of_queries, (t2 - t1)))
    print("********* end DEBUG ***********\n")

    return return_data


def insert_records(query, data):
    _tstart = time.time()
    success = False
    try:
        conn = mysql.connector.connect(user=db_user, password=db_pwd, host=db_host, port=db_port, database=db_name)
        cursor = conn.cursor()
        cursor.executemany(query, data)

        conn.commit()
        success = True
    except Error as e:
        print('Error:', e)
    finally:
        cursor.close()
        conn.close()

    _tend = time.time()
    print("\n********* start DEBUG ***********")
    print("insert_records execution time: %s" % (_tend - _tstart))
    print("********* end DEBUG ***********\n")
    return success


def batch_insert(insert_statement, insert_data, batch_size=50000):
    _tstart = time.time()

    i = 0
    j = batch_size
    k = len(insert_data)

    print("\nLength of data to insert: %s" % len(insert_data))
    print("Insert batch size: %s" % batch_size)
    print("Starting loop...")

    number_of_inserts = len(insert_data) // batch_size
    if len(insert_data) % batch_size > 0:
        number_of_inserts += 1

    insert_num = 1
    payload = []
    while j < k:
        t1 = time.time()

        print("%s:%s" % (i, j))
        payload = insert_data[i:j]

        if insert_records(insert_statement, payload):
            i = j
            j += batch_size
        else:
            raise ("Error inserting batch! Exiting...")

        t2 = time.time()

        print("\n********* start DEBUG ***********")
        print("INSERT %s/%s complete - execution time: %s" % (insert_num, number_of_inserts, (t2 - t1)))
        print("********* end DEBUG ***********\n")

        insert_num += 1

    print("Out of loop...")

    t1 = time.time()

    print("\n%s:%s" % (i, k))

    payload = insert_data[i:k]
    if not insert_records(insert_statement, payload):
        raise ("Error inserting batch! Exiting...")

    t2 = time.time()

    print("\n********* start DEBUG ***********")
    print("INSERT %s/%s complete - execution time: %s" % (insert_num, number_of_inserts, (t2 - t1)))
    print("********* end DEBUG ***********\n")

    _tend = time.time()

    print("\n********* start DEBUG ***********")
    print("batch_insert execution time: %s" % (_tend - _tstart))
    print("********* end DEBUG ***********\n")


# endregion


import matplotlib.ticker
from scipy.special import gammainc, gamma, gammaincc

h = 0.7
phi = 1.6e-2*h**3              # +/- 0.3 Mpc^-3
a = -1.07                       # +/- 0.07
L_B_star = 1.2e+10/h**2         # +/- 0.1
delta_dist = 16.7               # Mpc

select_galaxies = '''
    SELECT 
        POW(10,-0.4*((g.B - (5*log10(g.dist*1e+6)-5)) - 5.48))/%s as L_Sun__L_star
    FROM 
        Galaxy g 
    WHERE 
        flag1='G' and 
        B IS NOT NULL and 
        dist IS NOT NULL and  
        dist BETWEEN %s AND %s 
'''

interval = 0

d1 = delta_dist * interval
d2 = delta_dist * (interval + 1.0)

print("Distance cut: [%s, %s]" % (d1, d2))

galaxies = query_db([select_galaxies % (L_B_star, d1, d2)])[0]
luminosities = []
for g in galaxies:
    luminosities.append(g[0])

fig = plt.figure(figsize=(10, 10), dpi=800)
ax = fig.add_subplot(111)


# Lineplot Version
# y, binEdges = np.histogram(np.log10(luminosities), bins=np.linspace(-4.0, 2.0, 35))
y, binEdges = np.histogram(np.log10(luminosities), bins=np.linspace(-6.0, 2.0, 45))
bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
ax.plot(bincenters, y, '-')

# Histogram Version
# ax.hist(np.log10(luminosities), bins=np.linspace(-3, 2, 50), log=True, histtype='step')

# Plot Schechter Function
cosmo = LambdaCDM(H0=100*h, Om0=0.27, Ode0=0.73)
z2 = z_at_value(cosmo.luminosity_distance, d2*u.Mpc)
v2 = cosmo.comoving_volume(z2)
delta_v = v2.value

if interval > 0:
    z1 = z_at_value(cosmo.luminosity_distance, d1*u.Mpc)
    v1 = cosmo.comoving_volume(z1)
    delta_v = (v2-v1).value

# Multiply the Schechter by the volume to go from number density to number...
schect = lambda x: h**3 * delta_v  * phi * x**(a+1) * np.exp(-x)
input = 10**np.linspace(-6.0, 2.0, 45)
# input = 10**np.linspace(-4.0, 2.0, 35)
output = schect(input)
ax.plot(np.log10(input), output, 'r--')

ax.annotate("%s-%s Mpc" % (int(np.floor(d1)), int(np.floor(d2))), xy=(-2.0, 1e5), xycoords="data", fontsize=24)


ax.set_yscale("log")

# ax.set_xlim([-3.0, 2.0])
ax.set_xlim([-6.0, 2.0])
ax.set_ylim([1.0, 1.0e+6])
# ax.set_xticks([-3, -2, -1, 0, 1])
ax.set_xticks([-6, -5, -4, -3, -2, -1, 0, 1])
ax.set_yticks([1, 10, 1e2, 1e3, 1e4, 1e5])

logfmt = matplotlib.ticker.LogFormatterExponent(base=10.0, labelOnlyBase=True)
ax.yaxis.set_major_formatter(logfmt)


ax.set_xlabel(r"Log($L_B/L^{*}_{B}$)", fontsize=24)
ax.set_ylabel(r"Log($N$)", fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=24, length=8.0, width=2)
fig.savefig("new_completeness_test.png", bbox_inches='tight')
plt.close('all')




# x1 = 0.626
x1 = 1e-10
# int_schecht = lambda dv: dv * phi * L_B_star * gammaincc(a + 2, x1) * gamma(a + 2)
lum_dens3 = 1.98e-2*1e10 # from GLADE
int_schecht = lambda dv: dv * lum_dens3

distances = []
completeness = []
for i in range(70): # 12
    d1 = delta_dist * i
    d2 = delta_dist * (i + 1.0)
    distances.append(d2)

    z2 = z_at_value(cosmo.luminosity_distance, d2 * u.Mpc)
    v2 = cosmo.comoving_volume(z2)
    delta_v = v2.value

    if i > 0:
        z1 = z_at_value(cosmo.luminosity_distance, d1 * u.Mpc)
        v1 = cosmo.comoving_volume(z1)
        delta_v = (v2 - v1).value

    galaxies = query_db([select_galaxies % (L_B_star, d1, d2)])[0]
    print("Total returned galaxies: %s" % len(galaxies))
    luminosities = []
    for g in galaxies:
        l = float(g[0])
        if l >= x1:
            luminosities.append(l * L_B_star)
    print("Galaxies that made the cut: %s" % len(luminosities))

    total_glade_lum = np.sum(luminosities)
    schect_lum = int_schecht(delta_v)

    completeness.append(100 * total_glade_lum/schect_lum)

fig = plt.figure(figsize=(10, 10), dpi=800)
ax = fig.add_subplot(111)

print(distances)
ax.plot(distances, completeness, marker='.', linestyle='-', color="b")
ax.set_xlim([0,1200])
# ax.set_xlim([0,200])
# ax.set_xticks([40., 60., 80., 100., 120., 140., 160., 180., 200.])

ax.set_xlabel("Luminosity distance [Mpc]", fontsize=24)
ax.set_ylabel("Completeness [%]", fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=24, length=8.0, width=2)
fig.savefig("new_completeness_test2.png", bbox_inches='tight')
plt.close('all')