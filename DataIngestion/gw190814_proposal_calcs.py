import matplotlib

# region imports
matplotlib.use("Agg")

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

import os
import optparse
import subprocess as s

from configparser import RawConfigParser
import multiprocessing as mp
import mysql.connector

import mysql.connector as test

print(test.__version__)

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
from scipy.integrate import simps
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
from shapely.geometry import JOIN_STYLE

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
from scipy.stats import pearsonr
# endregion

# region config

# Set up dustmaps config
config["data_dir"] = "./"

# Generate all pixel indices
# cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
configFile = os.path.join(__location__, 'Settings.ini')

db_config = RawConfigParser()
db_config.read(configFile)

db_name = db_config.get('database', 'DATABASE_NAME')
db_user = db_config.get('database', 'DATABASE_USER')
db_pwd = db_config.get('database', 'DATABASE_PASSWORD')
db_host = db_config.get('database', 'DATABASE_HOST')
db_port = db_config.get('database', 'DATABASE_PORT')

isDEBUG = False
# endregion

# region db CRUD
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

        db = my.connect(host=db_host, user=db_user, passwd=db_pwd, db=db_name, port=3306)
        cursor = db.cursor()

        for q in query_list:
            cursor.execute(q)

            if commit:  # used for updates, etc
                db.commit()

            streamed_results = []
            print("\tfetching results...")
            while True:
                r = cursor.fetchmany(1000000)
                count = len(r)
                streamed_results += r
                size_in_mb = sys.getsizeof(streamed_results) / 1.0e+6

                print("\t\tfetched: %s; current length: %s; running size: %0.3f MB" % (
                count, len(streamed_results), size_in_mb))

                if not r or count < chunk_size:
                    break
            results.append(streamed_results)

    except Error as e:
        print('Error:', e)
    finally:
        cursor.close()
        db.close()

    return results


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

start = time.time()

flag_good_galaxies = False
reset_flag_good_galaxies = False
plot_demographics = False
plot_completeness = False
plot_2D_histogram = False

create_2dF_static_grid = False
insert_2dF_static_grid = create_2dF_static_grid and False
create_tile_pixel_relations = True

plot_localization = False
load_ozDES = False
plot_ozDES = False


map_nside = 1024

h = 0.7
phi = 1.6e-2*h**3              # +/- 0.3 Mpc^-3
a = -1.07                       # +/- 0.07
L_B_star = 1.2e+10/h**2         # +/- 0.1
cosmo = LambdaCDM(H0=100*h, Om0=0.27, Ode0=0.73)
cosmo_high = LambdaCDM(H0=20.0, Om0=0.27, Ode0=0.73)
cosmo_low = LambdaCDM(H0=140.0, Om0=0.27, Ode0=0.73)

path_format = "{}/{}"
ps1_strm_dir = "../PS1_DR2_QueryData/PS1_STRM"
northern_95th_pixel_ids = path_format.format(ps1_strm_dir, "northern_95th_pixel_ids.txt")
southern_95th_pixel_ids = path_format.format(ps1_strm_dir, "southern_95th_pixel_ids.txt")

# region Load pixel_ids from file...
pixel_select = '''
    SELECT 
        id, 
        HealpixMap_id, 
        Pixel_Index, 
        Prob, 
        Distmu, 
        Distsigma, 
        Distnorm, 
        Mean, 
        Stddev, 
        Norm, 
        N128_SkyPixel_id 
    FROM HealpixPixel 
    WHERE id IN (%s) 
'''
northern_pixel_ids = []
with open(northern_95th_pixel_ids, 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
    next(csvreader)  # skip header

    for row in csvreader:
        id = row[0]
        northern_pixel_ids.append(id)

pixel_result_north = query_db([pixel_select % ",".join(northern_pixel_ids)])[0]
print("Total NSIDE=1024 pixels in Northern 95th: %s" % len(pixel_result_north))

southern_pixel_ids = []
with open(southern_95th_pixel_ids, 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
    next(csvreader)  # skip header

    for row in csvreader:
        id = row[0]
        southern_pixel_ids.append(id)

pixel_result_south = query_db([pixel_select % ",".join(southern_pixel_ids)])[0]
print("Total NSIDE=1024 pixels in Southern 95th: %s" % len(pixel_result_south))

map_pix_north = []
map_pix_south = []
all_map_pix = []
for m in pixel_result_north:
    pix_id = int(m[0])
    index = int(m[2])
    prob = float(m[3])
    dist = float(m[7])
    stddev = float(m[8])
    p = Pixel_Element(index, map_nside, prob, pixel_id=pix_id, mean_dist=dist, stddev_dist=stddev)
    map_pix_north.append(p)
    all_map_pix.append(p)

for m in pixel_result_south:
    pix_id = int(m[0])
    index = int(m[2])
    prob = float(m[3])
    dist = float(m[7])
    stddev = float(m[8])
    p = Pixel_Element(index, map_nside, prob, pixel_id=pix_id, mean_dist=dist, stddev_dist=stddev)
    map_pix_south.append(p)
    all_map_pix.append(p)

# Sort all_map_pix
all_map_pix = sorted(all_map_pix, key=lambda x: x.prob, reverse=True)
# endregion

# region Compute Volume Information for the North where we have PS1 information
map_pix_dist = np.asarray([mp.mean_dist for mp in map_pix_north])
map_pix_dist_far = np.asarray([mp.mean_dist+2.0*mp.stddev_dist for mp in map_pix_north])
map_pix_dist_near = np.asarray([mp.mean_dist-2.0*mp.stddev_dist for mp in map_pix_north])

map_pix_z_limits = {}
for mp in map_pix_north:
    max_dist = mp.mean_dist+2.0*mp.stddev_dist
    min_dist = mp.mean_dist-2.0*mp.stddev_dist

    # Stretch range with cosmologies with H0 ranging from 20 to 140...
    min_z = z_at_value(cosmo_high.luminosity_distance, min_dist * u.Mpc)
    max_z = z_at_value(cosmo_low.luminosity_distance, max_dist * u.Mpc)

    if mp.id not in map_pix_z_limits:
        map_pix_z_limits[mp.id] = (min_z, max_z, min_dist, max_dist)

dist_grid = np.logspace(np.log10(min(map_pix_dist_near)-5.0), np.log10(max(map_pix_dist_far)+5.0), 1000)
z_grid = [z_at_value(cosmo.luminosity_distance, d * u.Mpc) for d in dist_grid]
z_model = interp1d(dist_grid, z_grid)
z_near = z_model(map_pix_dist_near)
z_far = z_model(map_pix_dist_far)
print("Min z: %0.4f; Max z: %0.4f" % (min(z_near), max(z_far)))

# volume_far = cosmo.comoving_volume(z_far)
# volume_min = cosmo.comoving_volume(z_near)
volume_far = cosmo_high.comoving_volume(z_far)
volume_min = cosmo_low.comoving_volume(z_near)

sky_fraction_per_pix = (1.0 / hp.nside2npix(map_nside))
d_volume = (volume_far - volume_min) * sky_fraction_per_pix
event_95th_volume = np.sum(d_volume)
print("Northern 95th percentile volume: %s" % event_95th_volume)
# endregion

# region Load PS1 galaxies and make cuts
Unique_PS1_Galaxies_Northern_95th = path_format.format(ps1_strm_dir, "Unique_PS1_Galaxies_Northern_95th.txt")
ps1_galaxies = []
with open(Unique_PS1_Galaxies_Northern_95th, 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
    next(csvreader)  # skip header

    for row in csvreader:
        ps1_galaxy_id = int(row[0])
        gaia_ra = float(row[1])
        gaia_dec = float(row[2])
        synth_B1 = float(row[3])
        synth_B2  = float(row[4])
        z_phot0 = float(row[5])
        z_photErr = float(row[6])
        ps1_hp_id = int(row[7])
        N128_SkyPixel_id = int(row[8])
        prob_Galaxy = float(row[9])

        kron_g = float(row[10])
        kron_r = float(row[11])
        kron_i = float(row[12])
        kron_z = float(row[13])
        kron_y = float(row[14])

        # look up z-range for the pixel that hosts this galaxy
        z_limits = map_pix_z_limits[ps1_hp_id]
        z_min = z_limits[0]
        z_max = z_limits[1]

        # Only let things through in the z +/- z_err that fall in the range, and that have physical synth B vals...
        if (z_phot0 > 0.0) and \
            (z_phot0 + z_photErr) >= z_min and \
            (z_phot0 - z_photErr) <= z_max and \
            synth_B2 > 0:

            PS1_z_dist = cosmo.luminosity_distance(z_phot0).value
            ps1_galaxies.append((ps1_galaxy_id, gaia_ra, gaia_dec, synth_B1, synth_B2, z_phot0, z_photErr, PS1_z_dist,
                                 ps1_hp_id, N128_SkyPixel_id, prob_Galaxy, kron_g, kron_r, kron_i, kron_z, kron_y))
print("Total PS1 galaxies within northern 95th, within z-range: %s" % len(ps1_galaxies))

zs = [p[5] for p in ps1_galaxies]
prob_Gal = [p[10] for p in ps1_galaxies]
r_kron_mags = [p[11] for p in ps1_galaxies]
ras = [p[1] for p in ps1_galaxies]
decs = [p[2] for p in ps1_galaxies]

count = 0
for p in prob_Gal:
    if p >= 0.9:
        count += 1
print("Number of galaxies with prob_Galaxy >= 0.9: %s" % count)

count = 0
for i in range(len(prob_Gal)):
    p = prob_Gal[i]
    r = r_kron_mags[i]

    if p >= 0.9 and r <= 22.0:
        count += 1
print("Number of galaxies with prob_Galaxy >= 0.9 AND kron R < 22.0: %s" % count)

all_cuts_p = []
all_cuts_r = []
all_cuts_z = []
all_cuts_ra = []
all_cuts_dec = []
all_cuts_PS1_galaxies = []
for i in range(len(prob_Gal)):
    p = prob_Gal[i]
    r = r_kron_mags[i]
    z = zs[i]
    ra = ras[i]
    dec = decs[i]

    if p >= 0.9 and r <= 22.0 and z <= 0.2:
        all_cuts_p.append(p)
        all_cuts_r.append(r)
        all_cuts_z.append(z)
        all_cuts_ra.append(ra)
        all_cuts_dec.append(dec)
        all_cuts_PS1_galaxies.append(ps1_galaxies[i])
print("Number of galaxies with prob_Galaxy >= 0.9 AND kron R < 22.0 and z <= 0.2: %s" % len(all_cuts_p))

# Get full PS1_STRM rows for Galaxies_That_Made_All_Cuts
select_PS1_STRM = '''
    SELECT ps.* FROM PS1_STRM ps
    JOIN PS1_Galaxy pg ON ps.uniquePspsOBid = pg.uniquePspsOBid
    WHERE pg.id IN (%s)
'''
photo_z_result = query_db([select_PS1_STRM % ",".join([str(g[0]) for g in all_cuts_PS1_galaxies])])[0]

Galaxies_That_Made_All_Cuts = path_format.format(ps1_strm_dir, "Galaxies_That_Made_All_Cuts.txt")
with open(Galaxies_That_Made_All_Cuts, 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',', skipinitialspace=True)

    csvwriter.writerow(("# objID", "uniquePspsOBid", "raMean", "decMean", "l", "b", "class", "prob_Galaxy", "prob_Star",
                        "prob_QSO", "extrapolation_Class", "cellDistance_Class", "cellID_Class", "z_phot", "z_photErr",
                        "z_phot0", "extrapolation_Photoz", "cellDistance_Photoz", "cellID_Photoz"))
    for row in photo_z_result:
        csvwriter.writerow(row)

if flag_good_galaxies:
    update_galaxies = '''
        UPDATE PS1_Galaxy SET GoodCandidate = 1 WHERE id IN (%s) 
    '''
    query_db([update_galaxies % ",".join([str(g[0]) for g in all_cuts_PS1_galaxies])], commit=True)

if reset_flag_good_galaxies:
    reset_galaxies = '''
            UPDATE PS1_Galaxy SET GoodCandidate = 0 WHERE id > 0  
        '''
    query_db([reset_galaxies], commit=True)
# endregion

# region ozDES
_14_0_20_0 = "14.0_20.0"

_20_0_20_5 = "20.0_20.5"
_20_5_21_0 = "20.5_21.0"
_21_0_21_5 = "21.0_21.5"
_21_5_22_0 = "21.5_22.0"
Q3 = "Q3"
Q4 = "Q4"

osDES_keylist = [
    (_20_0_20_5, Q3, "red", "-", "20.0--20.5"),
    (_20_0_20_5, Q4, "red", "--", ""),

    (_20_5_21_0, Q3, "blue", "-", "20.5--21.0"),
    (_20_5_21_0, Q4, "blue", "--", ""),

    (_21_0_21_5, Q3, "darkgreen", "-", "21.0--21.5"),
    (_21_0_21_5, Q4, "darkgreen", "--", ""),

    (_21_5_22_0, Q3, "khaki", "-", "21.5--22.0"),
    (_21_5_22_0, Q4, "khaki", "--", "")
]

ozDES_data = OrderedDict()
if load_ozDES:

    # Load all datasets
    for key in osDES_keylist:
        file_path = path_format.format(ps1_strm_dir, "{}_{}.csv".format(key[0], key[1]))

        if key[0] not in ozDES_data:
            ozDES_data[key[0]] = OrderedDict()

        ozDES_data[key[0]][key[1]] = [[], []]

        with open(file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)

            for row in csvreader:
                exp_time = float(row[0])
                z_complete = float(row[1])/100.

                ozDES_data[key[0]][key[1]][0].append(exp_time)
                ozDES_data[key[0]][key[1]][1].append(z_complete)


# Group PS1 candidates by bin
ps1_galaxies_by_mag_bin = OrderedDict()
ps1_galaxies_by_mag_bin[_14_0_20_0] = ([], "blue")
ps1_galaxies_by_mag_bin[_20_0_20_5] = ([], "orange")
ps1_galaxies_by_mag_bin[_20_5_21_0] = ([], "red")
ps1_galaxies_by_mag_bin[_21_0_21_5] = ([], "green")
ps1_galaxies_by_mag_bin[_21_5_22_0] = ([], "black")


# ps1_galaxies.append((ps1_galaxy_id, gaia_ra, gaia_dec, synth_B1, synth_B2, z_phot0, z_photErr, PS1_z_dist,
#                                  ps1_hp_id, N128_SkyPixel_id, prob_Galaxy, kron_g, kron_r, kron_i, kron_z, kron_y))
for g in all_cuts_PS1_galaxies:
    r = g[12]
    if r < 20.0:
        ps1_galaxies_by_mag_bin[_14_0_20_0][0].append(g)
    elif r >= 20.0 and r < 20.5:
        ps1_galaxies_by_mag_bin[_20_0_20_5][0].append(g)
    elif r >= 20.5 and r < 21.0:
        ps1_galaxies_by_mag_bin[_20_5_21_0][0].append(g)
    elif r >= 21.0 and r < 21.5:
        ps1_galaxies_by_mag_bin[_21_0_21_5][0].append(g)
    elif r >= 21.5:
        ps1_galaxies_by_mag_bin[_21_5_22_0][0].append(g)

frac_lt_20 = len(ps1_galaxies_by_mag_bin[_14_0_20_0][0])/len(all_cuts_PS1_galaxies)
print("Percentage < 20 mag: %0.3f" % frac_lt_20)

frac_20_0_20_5 = len(ps1_galaxies_by_mag_bin[_20_0_20_5][0])/len(all_cuts_PS1_galaxies)
print("Percentage > 20 and < 20.5 mag: %0.3f" % frac_20_0_20_5)

frac_20_5_21_0 = len(ps1_galaxies_by_mag_bin[_20_5_21_0][0])/len(all_cuts_PS1_galaxies)
print("Percentage > 20.5 and < 21.0 mag: %0.3f" % frac_20_5_21_0)

frac_21_0_21_5 = len(ps1_galaxies_by_mag_bin[_21_0_21_5][0])/len(all_cuts_PS1_galaxies)
print("Percentage > 21.0 and < 21.5 mag: %0.3f" % frac_21_0_21_5)

frac_gt_21_5 = len(ps1_galaxies_by_mag_bin[_21_5_22_0][0])/len(all_cuts_PS1_galaxies)
print("Percentage > 21.5 mag: %0.3f" % frac_gt_21_5)

# endregion

# region Plots
if plot_demographics:
    fig = plt.figure(figsize=(10, 10), dpi=800)
    ax1 = fig.add_subplot(221)
    n1, bins1, patches1 = ax1.hist(zs, histtype='step', bins=np.linspace(0, 1, 20), color="black")
    ax1.set_xlabel("Photo z")
    ax1.set_ylabel("Count")

    ax2 = fig.add_subplot(222)
    n2, bins2, patches2 = ax2.hist(prob_Gal, histtype='step', bins=np.linspace(0.7, 1, 20), color="dodgerblue")
    ax2.set_xlabel("prob_Galaxy")
    ax2.set_ylabel("Count")

    ax3 = fig.add_subplot(223)
    n3, bins3, patches3 = ax3.hist(r_kron_mags, histtype='step', bins=np.linspace(14, 22, 20), color="red")
    ax3.set_xlabel("Kron r [mag]")
    ax3.set_ylabel("Count")

    ax1.hist(all_cuts_z, histtype='stepfilled', bins=bins1, color='black')
    ax2.hist(all_cuts_p, histtype='stepfilled', bins=bins2, color='dodgerblue')
    ax3.hist(all_cuts_r, histtype='stepfilled', bins=bins3, color='red')

    fig.savefig("PS1_Sample_Stats.png", bbox_inches='tight')
    plt.close('all')

if plot_completeness:
    # GLADE
    # Unique_GLADE_Galaxies_Northern_95th = path_format.format(ps1_strm_dir, "Unique_GLADE_Galaxies_Northern_95th.txt")
    # glade_galaxies = []
    # with open(Unique_GLADE_Galaxies_Northern_95th, 'r') as csvfile:
    #     csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
    #     next(csvreader)  # skip header
    #
    #     for row in csvreader:
    #         glade_galaxy_id = int(row[0])
    #         ra = float(row[1])
    #         dec = float(row[2])
    #         B = float(row[3])
    #         z = float(row[4])
    #         z_dist = float(row[5])
    #         z_dist_err = float(row[6])
    #         gd2_hp_id = int(row[7])
    #
    #         # look up z-range for the pixel that hosts this galaxy
    #         z_limits = map_pix_z_limits[gd2_hp_id]
    #         min_dist = z_limits[2]
    #         max_dist = z_limits[3]
    #
    #         # Only let things through in the z +/- z_err that fall in the range, and that have physical synth B vals...
    #         if (z_dist + z_dist_err) >= min_dist and (z_dist - z_dist_err) <= max_dist:
    #             glade_galaxies.append((glade_galaxy_id, ra, dec, B, z, z_dist, z_dist_err, gd2_hp_id))
    #
    # print("Total GLADE galaxies within northern 95th, within z-range: %s" % len(glade_galaxies))
    #
    # glade_luminosities = []
    # for g in glade_galaxies:
    #     B = g[3]
    #     z_dist = g[5]
    #
    #     L_Sun__L_star = 10 ** (-0.4 * ((B - (5 * np.log10(z_dist * 1e+6) - 5)) - 5.48)) / L_B_star
    #     glade_luminosities.append(L_Sun__L_star)

    ps1_luminosities = []
    for g in ps1_galaxies:
        synth_B2 = g[4]
        z_dist = g[7]

        L_Sun__L_star = 10**(-0.4*((synth_B2 - (5*np.log10(z_dist*1e+6)-5)) - 5.48))/L_B_star
        ps1_luminosities.append(L_Sun__L_star)

    total_luminosities = []
    # total_luminosities += glade_luminosities
    total_luminosities += ps1_luminosities
    total_lum = np.sum(total_luminosities)

    mean_B_lum_density = 1.98e-2 # in units of (L10 = 10^10 L_B, solar)
    sch_L10 = event_95th_volume.value * mean_B_lum_density
    total_complete = total_lum/sch_L10
    print("GLADE + PS1 Completeness: %0.4f" % total_complete)


    y_ps1, binEdges_ps1 = np.histogram(np.log10(ps1_luminosities), bins=np.linspace(-6.0, 2.0, 45))
    bincenters_ps1 = 0.5 * (binEdges_ps1[1:] + binEdges_ps1[:-1])

    # y_glade, binEdges_glade = np.histogram(np.log10(glade_luminosities), bins=np.linspace(-6.0, 2.0, 45))
    # bincenters_glade = 0.5 * (binEdges_ps1[1:] + binEdges_ps1[:-1])

    # y_tot, binEdges_tot = np.histogram(np.log10(total_luminosities), bins=np.linspace(-6.0, 2.0, 45))
    # bincenters_tot = 0.5 * (binEdges_tot[1:] + binEdges_tot[:-1])

    schect = lambda x, vol: h**3 * vol  * phi * x**(a+1) * np.exp(-x)
    schect_input = 10**np.linspace(-6.0, 2.0, 45)
    schect_output = schect(schect_input, event_95th_volume.value)

    fig = plt.figure(figsize=(10, 10), dpi=800)
    ax = fig.add_subplot(111)

    ax.plot(np.log10(schect_input), schect_output, 'r--')
    ax.plot(bincenters_ps1, y_ps1, '-', color='blue')
    # ax.plot(bincenters_glade, y_glade, '-', color='green')
    # ax.plot(bincenters_tot, y_tot, '--', color='orange')
    # print(central_L_Sun__L_star)
    # print(central_expected_number)
    # ax.plot(central_L_Sun__L_star, central_expected_number, 'k+')

    ax.set_yscale("log")

    ax.set_xlim([-6.0, 2.0])
    ax.set_ylim([1.0, 1.0e+6])
    ax.set_xticks([-6, -5, -4, -3, -2, -1, 0, 1])
    ax.set_yticks([1, 10, 1e2, 1e3, 1e4, 1e5])

    logfmt = matplotlib.ticker.LogFormatterExponent(base=10.0, labelOnlyBase=True)
    ax.yaxis.set_major_formatter(logfmt)

    ax.set_xlabel(r"Log($L_B/L^{*}_{B}$)", fontsize=24)
    ax.set_ylabel(r"Log($N$)", fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24, length=8.0, width=2)
    # fig.savefig("GLADE_GW190814_Completeness.png", bbox_inches='tight')
    fig.savefig("PS1_GW190814_Completeness.png", bbox_inches='tight')
    plt.close('all')

if plot_2D_histogram:
    # 2D histogram
    fig = plt.figure(figsize=(10, 10), dpi=800)
    ax = fig.add_subplot(111)

    all_Bs = []
    all_zs = []
    for p in ps1_galaxies:
        all_Bs.append(p[4])
        all_zs.append(p[5])
    hist_obj = ax.hist2d(all_zs, all_Bs, bins=[np.linspace(0.0, 0.1, 15), np.linspace(16.0, 22.0, 15)])

    central_distances = [cosmo.luminosity_distance((z1 + z2)/2).value for z1, z2 in zip(hist_obj[1][:-1], hist_obj[1][1:])]
    central_volumes = [(cosmo.comoving_volume(z2) -
                        cosmo.comoving_volume(z1)).value *
                        sky_fraction_per_pix *
                        len(pixel_result_north) for z1, z2 in zip(hist_obj[1][:-1], hist_obj[1][1:])]

    central_mags = [(m1 + m2)/2 for m1, m2 in zip(hist_obj[2][:-1], hist_obj[2][1:])]

    central_expected_number = []
    central_L_Sun__L_star = []
    for i in range(len(central_mags)):
        # compute luminosity expected in this mag bin for this volume shell
        i_dist = central_distances[i] # Mpc
        i_mag = central_mags[i] # mag
        i_vol = central_volumes[i]
        i_L_Sun__L_star = 10 ** (-0.4 * ((i_mag - (5 * np.log10(i_dist * 1e+6) - 5)) - 5.48)) / L_B_star

        i_log_lum = np.log10(i_L_Sun__L_star)
        central_L_Sun__L_star.append(i_log_lum)

        num = schect(i_L_Sun__L_star, i_vol)
        central_expected_number.append(num)

    min_z = z_at_value(cosmo.luminosity_distance, 163 * u.Mpc)
    mean_z = z_at_value(cosmo.luminosity_distance, 267 * u.Mpc)
    max_z = z_at_value(cosmo.luminosity_distance, 371 * u.Mpc)

    ax.vlines(mean_z, 16, 22, colors='r', linestyles='-')
    ax.vlines(min_z, 16, 22, colors='r', linestyles='--')
    ax.vlines(max_z, 16, 22, colors='r', linestyles='--')

    ax.set_xlabel("Photo z")
    ax.set_ylabel("Synthetic B [mag]")
    plt.colorbar(hist_obj[3], ax=ax)

    fig.savefig("PS1_GW190814_z_vs_B.png", bbox_inches='tight')
    plt.close('all')

_2dF_radius = 1.05 # degrees
_2df_tiles = []
if create_2dF_static_grid:
    # Create static grid for 2dF
    central_ra = 12.63029862
    central_dec = -25.08122407
    box_height = 4.0931975
    box_width = 3.5350342

    northern_limit = central_dec + box_height / 2.0
    southern_limit = central_dec - box_height / 2.0
    eastern_limit = central_ra + box_width / 2.0
    western_limit = central_ra - box_width / 2.0

    frac_dec_tile, num_dec_tiles = math.modf(box_height / _2dF_radius)

    total_dec_tiles = int(num_dec_tiles) + 1
    if num_dec_tiles == 0:
        num_dec_tiles = 1

    dec_differential = (_2dF_radius - (frac_dec_tile * _2dF_radius)) / num_dec_tiles

    dec_delta = _2dF_radius - dec_differential
    starting_dec = southern_limit + _2dF_radius / 2.0

    decs = []
    for i in range(total_dec_tiles):
        d = starting_dec + i * dec_delta
        decs.append(d)

    ras_over_decs = []
    for d in decs:

        adjusted_tile_width = _2dF_radius / np.abs(np.cos(np.radians(d)))
        frac_ra_tile, num_ra_tiles = math.modf(box_width / adjusted_tile_width)
        total_ra_tiles = int(num_ra_tiles) + 1

        if num_ra_tiles == 0:
            num_ra_tiles = 1

        ra_differential = (adjusted_tile_width - (frac_ra_tile * adjusted_tile_width)) / num_ra_tiles
        ra_delta = adjusted_tile_width - ra_differential

        ras = []
        starting_ra = (adjusted_tile_width / 2.0) + western_limit
        for i in range(total_ra_tiles):
            r = starting_ra + i * ra_delta
            ras.append(r)

        ras_over_decs.append(ras)

    print("All Sky statistics for 2dF dense grid")
    print("\tNum of dec strips: %s" % len(decs))

    north_index = find_nearest(decs, northern_limit)
    south_index = find_nearest(decs, southern_limit)
    equator_index = find_nearest(decs, 0.0)

    print("\tNorthern most dec: %s" % decs[north_index])
    print("\tSouthern most dec: %s" % decs[south_index])

    east_index = find_nearest(ras_over_decs[equator_index], eastern_limit)
    west_index = find_nearest(ras_over_decs[equator_index], western_limit)

    print("\tEastern most ra: %s" % ras_over_decs[equator_index][east_index])
    print("\tWestern most ra: %s" % ras_over_decs[equator_index][west_index])

    print("\tNum of ra tiles in northern most dec slice: %s" % len(ras_over_decs[north_index]))
    print("\tNum of ra tiles at celestial equator: %s" % len(ras_over_decs[equator_index]))

    print("Constructing grid of coordinates...")
    RA = []
    DEC = []
    for i, d in enumerate(decs):
        for ra in ras_over_decs[i]:
            RA.append(ra)
            DEC.append(d)

    print(RA)
    print(DEC)

    ebv = None
    with open('ebv.pkl', 'rb') as handle:
        ebv = pickle.load(handle)

    sky_pixels = None
    with open('sky_pixels.pkl', 'rb') as handle:
        sky_pixels = pickle.load(handle)

    nside128 = 128
    # Get 2dF static grid...
    for i in range(len(RA)):
        t = Tile(RA[i], DEC[i], width=None, height=None, nside=map_nside, radius=_2dF_radius)
        t.field_name = str(i)
        n128_index = hp.ang2pix(nside128, 0.5 * np.pi - t.dec_rad, t.ra_rad)  # theta, phi
        t.mwe = ebv[n128_index]
        t.id = sky_pixels[nside128][n128_index].id
        _2df_tiles.append(t)

    print("Number of dense grid tiles: %s" % len(_2df_tiles))

    if insert_2dF_static_grid:
        raise Exception("Do you mean to do this? If so, comment me out.")

        select_detector_id = "SELECT id FROM Detector WHERE Name='%s'"
        insert_static_tile = "INSERT INTO StaticTile (Detector_id, FieldName, RA, _Dec, Coord, Poly, EBV, N128_SkyPixel_id) VALUES (%s, %s, %s, %s, ST_PointFromText(%s, 4326), ST_GEOMFROMTEXT(%s, 4326), %s, %s)"
        _2dF_detector = Detector("2dF", detector_width_deg=None, detector_height_deg=None, detector_radius_deg=_2dF_radius)
        _2dF_id = query_db([select_detector_id % '2dF'])[0][0][0]

        static_tile_data = []
        for t in _2df_tiles:
            static_tile_data.append((_2dF_id,
                                     t.field_name,
                                     t.ra_deg,
                                     t.dec_deg,
                                     "POINT(%s %s)" % (t.dec_deg, t.ra_deg - 180.0),
                                     # Dec, RA order due to MySQL convention for lat/lon
                                     t.query_polygon_string,
                                     str(t.mwe),
                                     t.id))

        batch_insert(insert_static_tile, static_tile_data)

if create_tile_pixel_relations:
    raise Exception("Do you mean to do this? If so, comment me out.")

    __map_pixel_select = "SELECT id, HealpixMap_id, Pixel_Index, Prob, Distmu, Distsigma, Distnorm, Mean, Stddev, Norm, N128_SkyPixel_id FROM HealpixPixel WHERE HealpixMap_id = %s"
    __map_pixels = query_db([__map_pixel_select % "2"])[0]
    __map_pixel_dict = {}
    for p in __map_pixels:
        __map_pixel_dict[int(p[2])] = p

    select_detector_id = "SELECT id, Name, Deg_width, Deg_height, Deg_radius, Area FROM Detector WHERE Name='%s'"
    tile_select = "SELECT id, Detector_id, FieldName, RA, _Dec, Coord, Poly, EBV, N128_SkyPixel_id FROM StaticTile WHERE Detector_id = %s "
    tile_pixel_upload_csv = path_format.format(ps1_strm_dir, "s_tile_pixel_upload.csv")

    tile_pixel_data = []
    _2df_detector = query_db([select_detector_id % "2dF"])[0][0]
    _2df_id = _2df_detector[0]
    _2df_static_tile_rows = query_db([tile_select % _2df_id])[0]

    _2dF_tiles = []
    for r in _2df_static_tile_rows:
        t = Tile(float(r[3]), float(r[4]), width=None, height=None, nside=map_nside, radius=_2dF_radius)
        tile_id = int(r[0])
        for p in t.enclosed_pixel_indices:
            tile_pixel_data.append((tile_id, __map_pixel_dict[p][0]))

    print("Appending `%s`" % tile_pixel_upload_csv)
    with open(tile_pixel_upload_csv, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        for data in tile_pixel_data:
            csvwriter.writerow(data)

    print("Bulk uploading Tile-Pixel...")
    st_hp_upload_sql = """LOAD DATA LOCAL INFILE '%s'
                        INTO TABLE StaticTile_HealpixPixel
                        FIELDS TERMINATED BY ','
                        LINES TERMINATED BY '\n'
                        (StaticTile_id, HealpixPixel_id);"""

    success = bulk_upload(st_hp_upload_sql % tile_pixel_upload_csv)
    if success:
        print("Upload successful!")

    print("Removing `%s`..." % tile_pixel_upload_csv)
    os.remove(tile_pixel_upload_csv)

if plot_localization:

    # Get configured Detectors
    detector_select = "SELECT id, Name, Deg_width, Deg_width, Deg_radius, Area, MinDec, MaxDec FROM Detector WHERE id = 1"
    detector_result = query_db([detector_select])[0]
    detectors = [Detector(dr[1], float(dr[2]), float(dr[2]), detector_id=int(dr[0])) for dr in detector_result]

    # region build multipolygons
    cutoff_50th = 0.5
    cutoff_90th = 0.90
    cutoff_95th = 0.95
    index_50th = 0
    index_90th = 0
    index_95th = 0

    print("Find index for 50th...")
    cum_prob = 0.0
    for i in range(len(all_map_pix)):
        cum_prob += all_map_pix[i].prob
        index_50th = i

        if (cum_prob >= cutoff_50th):
            break

    print("... %s" % index_50th)

    print("Find index for 90th...")
    cum_prob = 0.0
    for i in range(len(all_map_pix)):
        cum_prob += all_map_pix[i].prob
        index_90th = i

        if (cum_prob >= cutoff_90th):
            break
    print("... %s" % index_90th)

    print("Find index for 95th...")
    cum_prob = 0.0
    for i in range(len(all_map_pix)):
        cum_prob += all_map_pix[i].prob
        index_95th = i

        if (cum_prob >= cutoff_95th):
            break
    print("... %s" % index_95th)

    print("Build multipolygons...")
    net_50_polygon = []
    for p in all_map_pix[0:index_50th]:
        net_50_polygon += p.query_polygon
    joined_50_poly = unary_union(net_50_polygon)

    # Fix any seams
    eps = 0.00001
    merged_50_poly = []
    smoothed_50_poly = joined_50_poly.buffer(eps, 1, join_style=JOIN_STYLE.mitre).buffer(-eps, 1, join_style=JOIN_STYLE.mitre)

    try:
        test_iter = iter(smoothed_50_poly)
        merged_50_poly = smoothed_50_poly
    except TypeError as te:
        merged_50_poly.append(smoothed_50_poly)

    print("Number of sub-polygons in `merged_50_poly`: %s" % len(merged_50_poly))
    sql_50_poly = SQL_Polygon(merged_50_poly, detectors[0])

    net_90_polygon = []
    for p in all_map_pix[0:index_90th]:
        net_90_polygon += p.query_polygon
    joined_90_poly = unary_union(net_90_polygon)

    # Fix any seams
    merged_90_poly = []
    smoothed_90_poly = joined_90_poly.buffer(eps, 1, join_style=JOIN_STYLE.mitre).buffer(-eps, 1, join_style=JOIN_STYLE.mitre)

    try:
        test_iter = iter(smoothed_90_poly)
        merged_90_poly = smoothed_90_poly
    except TypeError as te:
        merged_90_poly.append(smoothed_90_poly)

    print("Number of sub-polygons in `merged_90_poly`: %s" % len(merged_90_poly))
    sql_90_poly = SQL_Polygon(merged_90_poly, detectors[0])
    print("... done.")


    net_95_polygon = []
    for p in all_map_pix[0:index_95th]:
        net_95_polygon += p.query_polygon
    joined_95_poly = unary_union(net_95_polygon)

    # Fix any seams
    merged_95_poly = []
    smoothed_95_poly = joined_95_poly.buffer(eps, 1, join_style=JOIN_STYLE.mitre).buffer(-eps, 1,
                                                                                         join_style=JOIN_STYLE.mitre)

    try:
        test_iter = iter(smoothed_95_poly)
        merged_95_poly = smoothed_95_poly
    except TypeError as te:
        merged_95_poly.append(smoothed_95_poly)

    print("Number of sub-polygons in `merged_95_poly`: %s" % len(merged_95_poly))
    sql_95_poly = SQL_Polygon(merged_95_poly, detectors[0])
    print("... done.")
    # endregion


    fig = plt.figure(figsize=(10, 10), dpi=1000)
    ax = fig.add_subplot(111)

    m = Basemap(projection='stere',
                lon_0=15.0,
                lat_0=-20.0,
                llcrnrlat=-35.0,
                urcrnrlat=-19.5,
                llcrnrlon=8.0,
                urcrnrlon=24.5)

    # Scale colormap
    # pix_90 = map_pix_sorted[0:index_90th]
    # pixel_probs = [p.prob for p in pix_90]
    min_prob = np.min(all_cuts_r)
    max_prob = np.max(all_cuts_r)
    print("min prob: %s" % min_prob)
    print("max prob: %s" % max_prob)
    norm = colors.Normalize(min_prob, max_prob)

    # Write out 95th polygons as regions
    for pi, poly in enumerate(sql_95_poly.query_polygon):

        ra_deg, dec_deg = zip(*[(c[0], c[1]) for c in poly.exterior.coords])

        with open(path_format.format(ps1_strm_dir, "GW190814_95th_%s.reg" % pi), 'w') as regfile:
            regfile.write("# Region file format: DS9 version 4.1\n")
            regfile.write("global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
            regfile.write("fk5\n")
            regfile.write("polygon(")

            c_list = []
            for ci in range(len(ra_deg)):
                ri = ra_deg[ci]
                di = dec_deg[ci]
                c_list.append("%0.8f" % ri)
                c_list.append("%0.8f" % di)
            regfile.write("%s)\n" % ",".join(c_list))

    sql_50_poly.plot(m, ax, edgecolor='black', linewidth=1.0, facecolor='None')
    sql_90_poly.plot(m, ax, edgecolor='black', linewidth=0.75, facecolor='None', alpha=0.8)
    sql_95_poly.plot(m, ax, edgecolor='k', linewidth=0.50, facecolor='k', alpha=0.05)

    for mag_range, galaxy_list_tuple in ps1_galaxies_by_mag_bin.items():

        galaxy_list = galaxy_list_tuple[0]
        galaxy_clr = galaxy_list_tuple[1]

        for i, g in enumerate(galaxy_list):
            ra, dec = g[1], g[2]
            angular_radius_deg = 15/3600. # 15 arcseconds

            c1 = Point(ra, dec).buffer(angular_radius_deg)
            c2 = shapely_transform(lambda x, y, z = None: ((ra - (ra - x)/np.abs(np.cos(np.radians(y)))), y), c1)

            ra_deg, dec_deg = zip(*[(c[0], c[1]) for c in c2.exterior.coords])

            x2, y2 = m(ra_deg, dec_deg)
            lat_lons = np.vstack([x2, y2]).transpose()
            ax.add_patch(Polygon(lat_lons, edgecolor=galaxy_clr, facecolor="None", linewidth=0.1))

    for t in _2df_tiles:
        t.plot(m, ax, edgecolor="red", facecolor="None", linewidth="0.5")


    # region Draw axes and colorbar
    # draw parallels.
    sm_label_size = 18
    label_size = 28
    title_size = 36

    _90_x1 = 0.77
    _90_y1 = 0.558

    _90_x2 = 0.77
    _90_y2 = 0.40

    _90_text_y = 0.37
    _90_text_x = 0.32

    _50_x1 = 0.60
    _50_y1 = 0.51

    _50_x2 = 0.48
    _50_y2 = 0.40

    _50_text_y = 0.37
    _50_text_x = 0.64

    parallels = list(np.arange(-90., 90., 10.))
    parallels += [-28.2491950622]
    dec_ticks = m.drawparallels(parallels, labels=[0, 1, 0, 0])
    for i, tick_obj in enumerate(dec_ticks):
        a = coord.Angle(tick_obj, unit=u.deg)

        for text_obj in dec_ticks[tick_obj][1]:
            direction = '+' if a.dms[0] > 0.0 else '-'
            text_obj.set_text(r'${0}{1:0g}^{{\degree}}$'.format(direction, np.abs(a.dms[0])))
            text_obj.set_size(sm_label_size)
            x = text_obj.get_position()[0]

            new_x = x * (1.0 + 0.08)
            text_obj.set_x(new_x)

    # draw meridians
    meridians = np.arange(0., 360., 7.5)
    ra_ticks = m.drawmeridians(meridians, labels=[0, 0, 0, 1])

    RA_label_dict = {
        7.5: r'$00^{\mathrm{h}}30^{\mathrm{m}}$',
        15.0: r'$01^{\mathrm{h}}00^{\mathrm{m}}$',
        22.5: r'$01^{\mathrm{h}}30^{\mathrm{m}}$',
    }

    for i, tick_obj in enumerate(ra_ticks):
        for text_obj in ra_ticks[tick_obj][1]:
            if tick_obj in RA_label_dict:
                text_obj.set_text(RA_label_dict[tick_obj])
                text_obj.set_size(sm_label_size)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Greys)
    sm.set_array([])  # can be an empty list

    tks = np.linspace(min_prob, max_prob, 6)
    tks_strings = []
    for t in tks:
        tks_strings.append('%0.2f' % (t * 100))

    # cb = fig.colorbar(sm, ax=ax, ticks=tks, orientation='vertical', fraction=0.04875, pad=0.02, alpha=0.80)
    # cb.ax.set_yticklabels(tks_strings, fontsize=16)
    # cb.set_label("2D Pixel Probability", fontsize=label_size, labelpad=9.0)
    # cb.ax.tick_params(width=2.0, length=6.0)
    # cb.outline.set_linewidth(2.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)

    ax.invert_xaxis()

    plt.ylabel(r'$\mathrm{Declination}$', fontsize=label_size, labelpad=36)
    plt.xlabel(r'$\mathrm{Right\;Ascension}$', fontsize=label_size, labelpad=30)
    # endregion

    fig.savefig('GW190814_PS1_galaxies.png', bbox_inches='tight')  # ,dpi=840
    plt.close('all')
    print("... Done.")

if plot_ozDES:

    fig = plt.figure(figsize=(10, 8), dpi=800)
    ax1 = fig.add_subplot(111)

    for key in osDES_keylist:
        exp_time = ozDES_data[key[0]][key[1]][0]
        z_complete = np.asarray(ozDES_data[key[0]][key[1]][1])*100.0

        if key[4] != "":
            ax1.plot(exp_time, z_complete, color=key[2], linestyle=key[3], label=key[4])
        else:
            ax1.plot(exp_time, z_complete, color=key[2], linestyle=key[3])

    ax1.set_xlim([0, 1600])
    ax1.set_xlabel("Exposure time (minutes)")
    ax1.set_ylabel("Redshift Completeness (%)")
    ax1.legend(loc="lower right")
    fig.savefig(path_format.format(ps1_strm_dir, "OzDES_Fig5.png"), bbox_inches='tight')
    plt.close('all')

# endregion


end = time.time()
duration = (end - start)
print("\n********* start DEBUG ***********")
print("Execution time: %s" % duration)
print("********* end DEBUG ***********\n")

