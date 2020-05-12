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
from AAOmega_Objects import *

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
import random

from scipy.stats import norm, gaussian_kde, mode
from scipy.integrate import trapz
from functools import reduce
# endregion

# region config

# Set up dustmaps config
config["data_dir"] = "./"

# Generate all pixel indices
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

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


localization_plots = False
mag_dist_and_fractional_completeness = True
H0_calc = False


# region shared stuff
path_format = "{}/{}"
ps1_strm_dir = "../PS1_DR2_QueryData/PS1_STRM"

healpix_map_id = 2
map_nside = 1024

# Load the polygons from AAOmega
ozDES_data = OrderedDict()
ozDES_models = OrderedDict()
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
            success = float(row[1]) / 100.

            ozDES_data[key[0]][key[1]][0].append(exp_time)
            ozDES_data[key[0]][key[1]][1].append(success)

        exp_times = ozDES_data[key[0]][key[1]][0]
        successes = ozDES_data[key[0]][key[1]][1]

        # Assume it stays linear to min exp time. Get slope of first two points...
        m = (successes[1] - successes[0]) / (exp_times[1] - exp_times[0])
        y_0 = successes[0] - m * exp_times[0]
        model = interp1d([0] + exp_times, [y_0] + successes)

        if key[0] not in ozDES_models:
            ozDES_models[key[0]] = OrderedDict()

        ozDES_models[key[0]][key[1]] = model


def get_osDES_model(kron_r_mag):
    model = ozDES_models[_20_0_20_5][Q3]
    required_exps = 1

    if r >= 20.5 and r < 21.0:
        model = ozDES_models[_20_5_21_0][Q3]
        required_exps = 2
    elif r >= 21.0 and r < 21.5:
        model = ozDES_models[_21_0_21_5][Q3]
        required_exps = 3
    elif r >= 21.5:
        model = ozDES_models[_21_5_22_0][Q3]
        required_exps = 4

    return required_exps, model


northern_95th_AAOmega_galaxy_dict_by_galaxy_id = {}
northern_95th_AAOmega_galaxy_dict_by_pixel_index = {}
# Get galaxies in north
good_candidate_ps1_galaxy_select = '''
    SELECT
        ps1.id as Galaxy_id, 
        ps1.gaia_ra,
        ps1.gaia_dec,
        ps1.rMeanKronMag,
        ps.z_phot0,
        ps.prob_Galaxy,
        hp.id as Pixel_id,
        hp.Pixel_Index,
        ps.z_photErr,
        ps1.synth_B2
    FROM 
        PS1_Galaxy ps1
    JOIN HealpixPixel_PS1_Galaxy hp_ps1 on hp_ps1.PS1_Galaxy_id = ps1.id
    JOIN PS1_STRM ps on ps.uniquePspsOBid = ps1.uniquePspsOBid
    JOIN HealpixPixel hp on hp.id = hp_ps1.HealpixPixel_id
    WHERE 
        ps1.GoodCandidate = 1 
    '''
northern_95th_galaxy_result = query_db([good_candidate_ps1_galaxy_select])[0]
print("Returned %s galaxies" % len(northern_95th_galaxy_result))
for g in northern_95th_galaxy_result:
    galaxy_id = int(g[0])
    ra = float(g[1])
    dec = float(g[2])
    r = float(g[3])
    z = float(g[4])
    prob_Galaxy = float(g[5])
    pix_id = int(g[6])
    pix_index = int(g[7])

    z_err = float(g[8])
    synth_B = float(g[9])

    required_exps, osDES_model = get_osDES_model(r)
    aaomega_galaxy = AAOmega_Galaxy(galaxy_id, ra, dec, prob_Galaxy, z, r, pix_index, required_exps, osDES_model,
                                    z_photErr=z_err, synth_B=synth_B)

    if pix_index not in northern_95th_AAOmega_galaxy_dict_by_pixel_index:
        northern_95th_AAOmega_galaxy_dict_by_pixel_index[pix_index] = {}

    northern_95th_AAOmega_galaxy_dict_by_pixel_index[pix_index][galaxy_id] = aaomega_galaxy
    northern_95th_AAOmega_galaxy_dict_by_galaxy_id[galaxy_id] = aaomega_galaxy

northern_AAOmega_pixel_dict = {}
northern_95th_pixel_select = '''
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
northern_95th_pixel_ids = []
northern_95th_pixel_ids_file = path_format.format(ps1_strm_dir, "northern_95th_pixel_ids.txt")
with open(northern_95th_pixel_ids_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
    next(csvreader)  # skip header
    for row in csvreader:
        id = row[0]
        northern_95th_pixel_ids.append(id)
northern_95th_pixel_result = query_db([northern_95th_pixel_select % ",".join(northern_95th_pixel_ids)])[0]
print("Total NSIDE=1024 pixels in Northern 95th: %s" % len(northern_95th_pixel_result))
for m in northern_95th_pixel_result:
    pix_id = int(m[0])
    index = int(m[2])
    prob = float(m[3])
    dist = float(m[7])
    stddev = float(m[8])

    contained_galaxies_dict = {}
    if index in northern_95th_AAOmega_galaxy_dict_by_pixel_index:
        contained_galaxies_dict = northern_95th_AAOmega_galaxy_dict_by_pixel_index[index]

        gal_count = len(contained_galaxies_dict)
        for gal_id, gal in contained_galaxies_dict.items():
            gal.prob_fraction = prob / gal_count

    p = AAOmega_Pixel(index, map_nside, prob, contained_galaxies_dict, pixel_id=pix_id, mean_dist=dist,
                      stddev_dist=stddev)
    northern_AAOmega_pixel_dict[index] = p






_50_AAOmega_pixel_dict = {}
_50th_pixel_select = '''
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
_50th_pixel_ids = []
_50th_pixel_ids_file = path_format.format(ps1_strm_dir, "50th_pixel_ids.txt")
with open(_50th_pixel_ids_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
    next(csvreader)  # skip header
    for row in csvreader:
        id = row[0]
        _50th_pixel_ids.append(id)

_50th_pixel_result = query_db([_50th_pixel_select % ",".join(_50th_pixel_ids)])[0]
print("Total NSIDE=1024 pixels in 50th: %s" % len(_50th_pixel_result))
for m in _50th_pixel_result:
    pix_id = int(m[0])
    index = int(m[2])
    prob = float(m[3])
    dist = float(m[7])
    stddev = float(m[8])

    _50_AAOmega_pixel_dict[index] = (pix_id, index, prob, dist, stddev)










#
#
# southern_AAOmega_pixel_dict = {}
# southern_95th_pixel_select = '''
#             SELECT
#                 id,
#                 HealpixMap_id,
#                 Pixel_Index,
#                 Prob,
#                 Distmu,
#                 Distsigma,
#                 Distnorm,
#                 Mean,
#                 Stddev,
#                 Norm,
#                 N128_SkyPixel_id
#             FROM HealpixPixel
#             WHERE id IN (%s)
#         '''
# southern_95th_pixel_ids = []
# southern_95th_pixel_ids_file = path_format.format(ps1_strm_dir, "southern_95th_pixel_ids.txt")
# with open(southern_95th_pixel_ids_file, 'r') as csvfile:
#     csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
#     next(csvreader)  # skip header
#     for row in csvreader:
#         id = row[0]
#         southern_95th_pixel_ids.append(id)
# southern_95th_pixel_result = query_db([southern_95th_pixel_select % ",".join(southern_95th_pixel_ids)])[0]
# print("Total NSIDE=1024 pixels in Southern 95th: %s" % len(southern_95th_pixel_result))
# for m in southern_95th_pixel_result:
#     pix_id = int(m[0])
#     index = int(m[2])
#     prob = float(m[3])
#     dist = float(m[7])
#     stddev = float(m[8])
#
#     p = AAOmega_Pixel(index, map_nside, prob, contained_galaxies={}, pixel_id=pix_id, mean_dist=dist,
#                       stddev_dist=stddev)
#     southern_AAOmega_pixel_dict[index] = p
#
# count_south = len(southern_AAOmega_pixel_dict)
# print("Count south: %s" % count_south)
#
# for s_index, s_pix in southern_AAOmega_pixel_dict.items():
#     j = random.randint(0, len(northern_AAOmega_pixel_dict) - 1)
#     n_pix_index = list(northern_AAOmega_pixel_dict.keys())[j]
#     p = northern_AAOmega_pixel_dict[n_pix_index]
#
#     fake_galaxies = {}
#     for g_id, g in p.contained_galaxies.items():
#         fake_g_id = g_id + 100000
#         a = AAOmega_Galaxy(fake_g_id, p.coord.ra.degree, p.coord.dec.degree, g.prob_galaxy, g.z, g.kron_r,
#                            s_pix.index, g.required_exps, g.efficiency_func)
#
#         fake_galaxies[fake_g_id] = a
#
#     gal_count = len(fake_galaxies)
#     for gal_id, gal in fake_galaxies.items():
#         gal.prob_fraction = s_pix.prob / gal_count
#     s_pix.contained_galaxies = fake_galaxies
#
# northern_AAOmega_pixel_dict.update(southern_AAOmega_pixel_dict)


filename = 'Obs10.reg'
filepath = path_format.format(ps1_strm_dir, filename)
aaomega_detector = Detector("2dF", detector_width_deg=None, detector_height_deg=None, detector_radius_deg=1.05)
aaomega_tiles = []
with open(filepath, 'r') as f:
    lines = f.readlines()

    for l in lines:
        if "circle(" in l:
            # EX: circle(0:52:11.133, -25:43:00.494, 3780.000") # color=red
            tup_str = l.split("{")[1].split("}")[0]
            tile_tup = tup_str.split(",")
            tile_num = int(tile_tup[0])
            exp_num = int(tile_tup[1])

            tokens = l.replace("circle(", "").replace("\") # color=red text={%s}\n" % tup_str, "").split(",")
            ra = tokens[0].strip()
            dec = tokens[1].strip()

            c = coord.SkyCoord(ra, dec, unit=(u.hour, u.deg))

            radius_deg = float(tokens[2]) / 3600.  # input in arcseconds

            a = AAOmega_Tile(c.ra.degree, c.dec.degree, map_nside, radius_deg,
                             northern_AAOmega_pixel_dict, num_exposures=exp_num,
                             tile_num=tile_num)

            if len(a.contained_pixels_dict) > 0:
                print("RA Hour: %s; Dec Deg: %s" % (c.ra.hms[0], c.dec.dms[0]))

                aaomega_tiles.append(a)


total_exps = 0
total_slews = 0
sortedTiles = sorted(aaomega_tiles, key=lambda x: x.tile_num)
all_gals, all_prob = 0, 0
all_gal_ids = []
print("Tile Num\tNum Exp\t\tGalaxies\tNet Prob")
for t in sortedTiles:

    total_slews += 1
    total_exps += t.num_exposures

    gals, prob, gal_ids = t.calculate_efficiency()

    all_gal_ids += gal_ids
    all_gals += gals
    all_prob += prob
    print("%s\t\t\t%s\t\t\t%s\t\t\t%0.6f" % (t.tile_num, t.num_exposures, gals, prob))
    # print("%s\t\t\t%s\t\t\t%s\t\t\t%0.8f" % (t.tile_num, t.num_exposures, gals, prob))

print("****************\n")
print("Total Galaxies %s/%s" % (all_gals,
                                np.sum([len(p.contained_galaxies)
                                        for pi, p in northern_AAOmega_pixel_dict.items()])))
print("Total Prob: %0.2f" % all_prob)
total_hours = (total_exps * 40 + total_slews * 15) / 60.
total_nights = total_hours / 10.0
print("Total Hours: %0.2f; Total 11-hour nights: %0.2f" % (total_hours, total_nights))

# endregion


if localization_plots:

    def create_contour_polygon(percentile, all_map_pix, detector):
        percentile_str = "%0.0fth" % (percentile * 100)
        percentile_cuttoff = percentile
        percentile_index = 0

        print("Find index for %s..." % percentile_str)
        cum_prob = 0.0
        for i in range(len(all_map_pix)):
            cum_prob += all_map_pix[i].prob
            percentile_index = i

            if (cum_prob >= percentile_cuttoff):
                break
        print("... %s" % percentile_index)

        net_polygon = []
        for p in all_map_pix[0:percentile_index]:
            net_polygon += p.query_polygon
        joined_poly = unary_union(net_polygon)

        # Fix any seams
        eps = 0.00001
        merged_poly = []
        smoothed_poly = joined_poly.buffer(eps, 1, join_style=JOIN_STYLE.mitre).buffer(-eps, 1,
                                                                                       join_style=JOIN_STYLE.mitre)

        try:
            test_iter = iter(smoothed_poly)
            merged_poly = smoothed_poly
        except TypeError as te:
            merged_poly.append(smoothed_poly)

        print("Number of sub-polygons in `merged_%s_poly`: %s" % (percentile_str, len(merged_poly)))
        sql_poly = SQL_Polygon(merged_poly, detector)
        return sql_poly


    # /Users/davecoulter/Dropbox/UCSC/GWSearch/teglon/Events/S190814bv/Candidates
    candidate_file_path = "../Events/S190814bv/Candidates/paper_candidates.csv"

    # Get configured Detectors
    detector_select = "SELECT id, Name, Deg_width, Deg_width, Deg_radius, Area, MinDec, MaxDec FROM Detector"
    detector_result = query_db([detector_select])[0]

    detectors = []
    for d in detector_result:
        if d[1] is not None and d[2] is not None:
            detectors.append(Detector(d[1], float(d[2]), float(d[3]), detector_id=int(d[0])))
        else:
            detectors.append(Detector(d[1], None, None, detector_radius_deg=float(d[4]), detector_id=int(d[0])))


    # Get all observed tiles for all configured detectors
    observed_tile_select = '''
                SELECT 
                    id,
                    Detector_id, 
                    FieldName, 
                    RA, 
                    _Dec, 
                    EBV, 
                    N128_SkyPixel_id, 
                    Band_id, 
                    MJD, 
                    Exp_Time, 
                    Mag_Lim, 
                    HealpixMap_id 
                FROM
                    ObservedTile 
                WHERE
                    HealpixMap_id = %s and 
                    Detector_id = %s 
            '''
    observed_tiles = {}
    for d in detectors:
        ot_result = query_db([observed_tile_select % (healpix_map_id, d.id)])[0]
        observed_tiles[d.name] = [
            Tile(float(ot[3]),
                 float(ot[4]),
                 d.deg_width,
                 d.deg_height,
                 1024,
                 tile_id=int(ot[0])) for ot in ot_result]

        print("Loaded %s %s tiles..." % (len(observed_tiles[d.name]), d.name))

    # load candidates if they exist
    candidates = []
    with open(candidate_file_path, 'r') as csvfile:

        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)  # skip header

        for row in csvreader:
            name = row[0]
            ra = float(row[1])
            dec = float(row[2])
            flag1 = bool(int(row[3]))
            flag2 = bool(int(row[4]))
            is_keck = bool(int(row[5]))

            # append tuple
            candidates.append((name, coord.SkyCoord(ra, dec, unit=(u.deg, u.deg)), flag1, flag2, is_keck))
    print("Number of candidates: %s" % len(candidates))

    select_pix = '''
                SELECT 
                    running_prob.id, 
                    running_prob.HealpixMap_id, 
                    running_prob.Pixel_Index, 
                    running_prob.Prob, 
                    running_prob.Distmu, 
                    running_prob.Distsigma, 
                    running_prob.Mean, 
                    running_prob.Stddev, 
                    running_prob.Norm, 
                    running_prob.N128_SkyPixel_id, 
                    running_prob.cum_prob 
                FROM 
                    (SELECT 
                        hp_prob.id, 
                        hp_prob.HealpixMap_id, 
                        hp_prob.Pixel_Index,
                        hp_prob.Prob, 
                        hp_prob.Distmu, 
                        hp_prob.Distsigma, 
                        hp_prob.Mean, 
                        hp_prob.Stddev, 
                        hp_prob.Norm, 
                        hp_prob.N128_SkyPixel_id, 
                        SUM(hp_prob.Prob) OVER(ORDER BY hp_prob.Prob DESC) AS cum_prob 
                    FROM 
                        (SELECT 
                            hp.id, 
                            hp.HealpixMap_id, 
                            hp.Pixel_Index,
                            hp.Prob, 
                            hp.Distmu, 
                            hp.Distsigma, 
                            hp.Mean, 
                            hp.Stddev, 
                            hp.Norm, 
                            hp.N128_SkyPixel_id 
                        FROM HealpixPixel hp 
                        WHERE hp.HealpixMap_id = %s 
                        ORDER BY
                            hp.Prob DESC) hp_prob
                        GROUP BY
                            hp_prob.id, 
                            hp_prob.HealpixMap_id, 
                            hp_prob.Pixel_Index,
                            hp_prob.Prob, 
                            hp_prob.Distmu, 
                            hp_prob.Distsigma, 
                            hp_prob.Mean, 
                            hp_prob.Stddev, 
                            hp_prob.Norm, 
                            hp_prob.N128_SkyPixel_id 
                        ) running_prob 
                WHERE 
                    running_prob.cum_prob <= 0.90 
            '''

    print("Selecting map pixels...")
    map_pix_result = query_db([select_pix % healpix_map_id])[0]
    print("...done")

    print("Building pixel elements...")
    map_pix = [Pixel_Element(int(mp[2]), map_nside, float(mp[3]), pixel_id=int(mp[0])) for mp in
               map_pix_result]
    map_pix_sorted = sorted(map_pix, key=lambda x: x.prob, reverse=True)
    print("...done")

    sql_50_poly = create_contour_polygon(0.5, map_pix_sorted, detectors[0])
    sql_95_poly = create_contour_polygon(0.95, map_pix_sorted, detectors[0])

    #
    # cutoff_50th = 0.5
    cutoff_95th = 0.9
    # index_50th = 0
    index_95th = 0
    #
    # print("Find index for 50th...")
    # cum_prob = 0.0
    # for i in range(len(map_pix_sorted)):
    #     cum_prob += map_pix_sorted[i].prob
    #     index_50th = i
    #
    #     if (cum_prob >= cutoff_50th):
    #         break
    #
    # print("... %s" % index_50th)
    #
    print("Find index for 95th...")
    cum_prob = 0.0
    for i in range(len(map_pix_sorted)):
        cum_prob += map_pix_sorted[i].prob
        index_95th = i

        if (cum_prob >= cutoff_95th):
            break
    print("... %s" % index_95th)
    #
    # print("Build multipolygons...")
    # net_50_polygon = []
    # for p in map_pix_sorted[0:index_50th]:
    #     net_50_polygon += p.query_polygon
    # joined_50_poly = unary_union(net_50_polygon)
    #
    # # Fix any seams
    # eps = 0.00001
    # merged_50_poly = []
    # smoothed_50_poly = joined_50_poly.buffer(eps, 1, join_style=JOIN_STYLE.mitre).buffer(-eps, 1,
    #                                                                                      join_style=JOIN_STYLE.mitre)
    # try:
    #     test_iter = iter(smoothed_50_poly)
    #     merged_50_poly = smoothed_50_poly
    # except TypeError as te:
    #     merged_50_poly.append(smoothed_50_poly)
    #
    # print("Number of sub-polygons in `merged_50_poly`: %s" % len(merged_50_poly))
    # sql_50_poly = SQL_Polygon(merged_50_poly, detectors[0])
    #
    # net_90_polygon = []
    # for p in map_pix_sorted[0:index_90th]:
    #     net_90_polygon += p.query_polygon
    # joined_90_poly = unary_union(net_90_polygon)
    #
    # # Fix any seams
    # merged_90_poly = []
    # smoothed_90_poly = joined_90_poly.buffer(eps, 1, join_style=JOIN_STYLE.mitre).buffer(-eps, 1,
    #                                                                                      join_style=JOIN_STYLE.mitre)
    # try:
    #     test_iter = iter(smoothed_90_poly)
    #     merged_90_poly = smoothed_90_poly
    # except TypeError as te:
    #     merged_90_poly.append(smoothed_90_poly)
    #
    # print("Number of sub-polygons in `merged_90_poly`: %s" % len(merged_90_poly))
    # sql_90_poly = SQL_Polygon(merged_90_poly, detectors[0])
    # print("... done.")

    # sql_50_poly = None
    # with open('sql50.pkl', 'rb') as handle:
    #     sql_50_poly = pickle.load(handle)
    #
    # sql_90_poly = None
    # with open('sql90.pkl', 'rb') as handle:
    #     sql_90_poly = pickle.load(handle)

    fig = plt.figure(figsize=(21, 10), dpi=1000)
    ax1 = fig.add_subplot(121)

    m = Basemap(projection='stere',
                lon_0=15.0,
                lat_0=-20.0,
                llcrnrlat=-35.0,
                urcrnrlat=-19.5,
                llcrnrlon=8.0,
                urcrnrlon=25.0)

    # Scale colormap
    pix_90 = map_pix_sorted[0:index_95th]
    pixel_probs = [p.prob for p in pix_90]
    min_prob = np.min(pixel_probs)
    max_prob = np.max(pixel_probs)

    print("min prob: %s" % min_prob)
    print("max prob: %s" % max_prob)


    norm = colors.Normalize(min_prob, max_prob)

    clrs = {
        "SWOPE": (230.0 / 256.0, 159 / 256.0, 0),
        "NICKEL": (0, 114.0 / 256.0, 178.0 / 256.0),
        "THACHER": (0, 158.0 / 256.0, 115.0 / 256.0),
        "MOSFIRE": (204.0 / 256.0, 121.0 / 256.0, 167.0 / 256.0),
        "SINISTRO": (218.0 / 256.0, 35.0 / 256.0, 35.0 / 256.0),
        "KAIT": (86.0 / 256.0, 180.0 / 256.0, 233.0 / 256.0)
    }
    # clrs = {
    #     "SWOPE": "gold",
    #     "NICKEL": "tomato",
    #     "THACHER": "limegreen",
    #     "MOSFIRE": (204.0 / 256.0, 121.0 / 256.0, 167.0 / 256.0),
    #     "SINISTRO": (218.0 / 256.0, 35.0 / 256.0, 35.0 / 256.0),
    #     "KAIT": "lightcyan"
    # }

    # Plot SWOPE, then THACHER, then NICKEL, then MOSFIRE
    x1, y1 = m(0, 0)
    m.plot(x1, y1, marker='s', markeredgecolor="k", markerfacecolor=clrs["SWOPE"], markersize=20, label="Swope",
           linestyle='None')
    m.plot(x1, y1, marker='s', markeredgecolor="k", markerfacecolor=clrs["SINISTRO"], markersize=18,
           label="LCOGT", linestyle='None')
    m.plot(x1, y1, marker='s', markeredgecolor="k", markerfacecolor=clrs["THACHER"], markersize=16, label="Thacher",
           linestyle='None')
    m.plot(x1, y1, marker='s', markeredgecolor="k", markerfacecolor=clrs["NICKEL"], markersize=14, label="Nickel",
           linestyle='None')
    m.plot(x1, y1, marker='s', markeredgecolor="k", markerfacecolor=clrs["KAIT"], markersize=12, label="KAIT",
           linestyle='None')
    m.plot(x1, y1, marker='s', markeredgecolor="k", markerfacecolor=clrs["MOSFIRE"], markersize=10, label="MOSFIRE",
           linestyle='None')

    # This will probably get washed out by no transparanecy for EPS figures...
    print("Plotting (%s) `pixels`..." % len(pix_90))
    for i, p in enumerate(pix_90):
        p.plot(m, ax1, facecolor=plt.cm.Greys(norm(p.prob)), edgecolor='None', linewidth=0.5,
               alpha=norm(p.prob) * 0.8)

    tile_opacity = 0.1
    print("Plotting Tiles for: %s (%s)" % ("SWOPE", clrs["SWOPE"]))
    for i, t in enumerate(observed_tiles["SWOPE"]):
        t.plot(m, ax1, edgecolor=clrs["SWOPE"], facecolor=clrs["SWOPE"], linewidth=0.25, alpha=tile_opacity)
        t.plot(m, ax1, edgecolor='k', facecolor="None", linewidth=0.25, alpha=1.0)

    print("Plotting Tiles for: %s (%s)" % ("SINISTRO", clrs["SINISTRO"]))
    for i, t in enumerate(observed_tiles["SINISTRO"]):
        t.plot(m, ax1, edgecolor=clrs["SINISTRO"], facecolor=clrs["SINISTRO"], linewidth=0.25, alpha=tile_opacity)
        t.plot(m, ax1, edgecolor='k', facecolor="None", linewidth=0.25, alpha=1.0, zorder=9999)

    print("Plotting Tiles for: %s (%s)" % ("THACHER", clrs["THACHER"]))
    for i, t in enumerate(observed_tiles["THACHER"]):
        t.plot(m, ax1, edgecolor=clrs["THACHER"], facecolor=clrs["THACHER"], linewidth=0.25, alpha=tile_opacity)
        t.plot(m, ax1, edgecolor='k', facecolor="None", linewidth=0.25, alpha=1.0)

    print("Plotting Tiles for: %s (%s)" % ("NICKEL", clrs["NICKEL"]))
    for i, t in enumerate(observed_tiles["NICKEL"]):
        t.plot(m, ax1, edgecolor=clrs["NICKEL"], facecolor=clrs["NICKEL"], linewidth=0.25, alpha=tile_opacity)
        t.plot(m, ax1, edgecolor='k', facecolor="None", linewidth=0.25, alpha=1.0)

    print("Plotting Tiles for: %s (%s)" % ("KAIT", clrs["KAIT"]))
    for i, t in enumerate(observed_tiles["KAIT"]):
        t.plot(m, ax1, edgecolor=clrs["KAIT"], facecolor=clrs["KAIT"], linewidth=0.25, alpha=1.0, zorder=9999)
        t.plot(m, ax1, edgecolor='k', facecolor="None", linewidth=0.25, alpha=1.0, zorder=9999)

    print("Plotting Tiles for: %s (%s)" % ("MOSFIRE", clrs["MOSFIRE"]))
    for i, t in enumerate(observed_tiles["MOSFIRE"]):
        t.plot(m, ax1, edgecolor=clrs["MOSFIRE"], facecolor=clrs["MOSFIRE"], linewidth=0.25, alpha=1.0, zorder=9999)
        t.plot(m, ax1, edgecolor='k', facecolor="None", linewidth=0.25, alpha=1.0, zorder=9999)



    print("Plotting SQL Multipolygons")

    # with open('sql50.pkl', 'wb') as handle:
    # 	pickle.dump(sql_50_poly, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('sql90.pkl', 'wb') as handle:
    # 	pickle.dump(sql_90_poly, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sql_50_poly.plot(m, ax1, edgecolor='black', linewidth=2.0, facecolor='None')
    sql_95_poly.plot(m, ax1, edgecolor='black', linewidth=2.0, facecolor='None')

    # Plotted off the map so that the legend will have a line item

    for c in candidates:
        x, y = m(c[1].ra.degree, c[1].dec.degree)

        mkrclr = 'gray'
        if c[2] and c[3]:  # passes both Charlie and Ryan cuts
            mkrclr = 'lime'
        m.plot(x, y, marker='o', markeredgecolor='k', markerfacecolor=mkrclr, markersize=5.0)

    x1, y1 = m(0, 0)
    m.plot(x1, y1, marker='.', linestyle="None", markeredgecolor="k", markerfacecolor="lime", markersize=20,
           label="Viable candidate")

    x2, y2 = m(0, 0)
    m.plot(x2, y2, marker='.', linestyle="None", markeredgecolor="k", markerfacecolor="grey", markersize=20,
           label="Excluded")


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

    parallels = np.arange(-90., 90., 10.)
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
        # a = coord.Angle(tick_obj, unit=u.deg)
        # print(a.hms)
        print(tick_obj)
        for text_obj in ra_ticks[tick_obj][1]:
            if tick_obj in RA_label_dict:
                text_obj.set_text(RA_label_dict[tick_obj])
                text_obj.set_size(sm_label_size)

    # sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Greys)
    # sm.set_array([])  # can be an empty list
    #
    # tks = np.linspace(min_prob, max_prob, 6)
    # # tks = np.logspace(np.log10(min_prob), np.log10(max_prob), 11)
    # tks_strings = []
    #
    # for t in tks:
    #     tks_strings.append('%0.2f' % (t * 100))
    #
    # cb = fig.colorbar(sm, ax=ax1, ticks=tks, orientation='vertical', fraction=0.04875, pad=0.02,
    #                   alpha=0.80)  # 0.08951
    # cb.ax.set_yticklabels(tks_strings, fontsize=16)
    # cb.set_label("2D Pixel Probability", fontsize=label_size, labelpad=9.0)
    #
    # cb.ax.tick_params(width=2.0, length=6.0)
    #
    # cb.outline.set_linewidth(2.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2.0)

    ax1.invert_xaxis()
    ax1.legend(loc='upper left', fontsize=sm_label_size, borderpad=0.35, handletextpad=0.0, labelspacing=0.4)

    ax1.set_ylabel(r'$\mathrm{Declination}$', fontsize=label_size, labelpad=36)
    ax1.set_xlabel(r'$\mathrm{Right\;Ascension}$', fontsize=label_size, labelpad=30)



















    ax2 = fig.add_subplot(122)
    m2 = Basemap(projection='stere',
                lon_0=15.0,
                lat_0=-20.0,
                llcrnrlat=-35.5,
                urcrnrlat=-19.5,
                llcrnrlon=8.0,
                urcrnrlon=25.0)

    all_prob = []
    for pix_index, p in northern_AAOmega_pixel_dict.items():
        all_prob.append(p.prob)

    min_gal_weighted_prob = np.min(all_prob)
    max_gal_weighted_prob = np.max(all_prob)
    n = colors.Normalize(min_gal_weighted_prob, max_gal_weighted_prob)



    clrs = {
        "SWOPE": (230.0 / 256.0, 159 / 256.0, 0),
        "NICKEL": (0, 114.0 / 256.0, 178.0 / 256.0),
        "THACHER": (0, 158.0 / 256.0, 115.0 / 256.0),
        "MOSFIRE": (204.0 / 256.0, 121.0 / 256.0, 167.0 / 256.0),
        "SINISTRO": (218.0 / 256.0, 35.0 / 256.0, 35.0 / 256.0),
        "KAIT": (86.0 / 256.0, 180.0 / 256.0, 233.0 / 256.0)
    }
    # x1, y1 = m(0, 0)
    # m.plot(x1, y1, marker='s', markeredgecolor="k", markerfacecolor=clrs["SWOPE"], markersize=20, label="Swope",
    #        linestyle='None')
    x1, y1 = m2(0, 0)
    m2.plot(x1, y1, marker='s', markeredgecolor="k", markerfacecolor=clrs["NICKEL"], markersize=20,
            label="2 x 40 min", linestyle='None')
    m2.plot(x1, y1, marker='s', markeredgecolor="k", markerfacecolor=clrs["SINISTRO"], markersize=20,
            label="3 x 40 min", linestyle='None')



    for pix_index, p in northern_AAOmega_pixel_dict.items():
        clr = plt.cm.Greys(n(p.prob))
        p.plot(m2, ax2, edgecolor="None", facecolor=clr, linewidth="1.0", alpha=1.0)


    sql_50_poly.plot(m2, ax2, edgecolor='black', linewidth=2.0, facecolor='None')
    sql_95_poly.plot(m2, ax2, edgecolor='gray', linewidth=2.0, facecolor='None')

    for t in aaomega_tiles:
        if t.num_exposures == 3:
            t.plot(m2, ax2, edgecolor=clrs["SINISTRO"], facecolor="None", linewidth="2.0")

    for t in aaomega_tiles:

        if t.num_exposures == 2:
            t.plot(m2, ax2, edgecolor=clrs["NICKEL"], facecolor="None", linewidth="2.0")



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
    dec_ticks = m2.drawparallels(parallels, labels=[0, 1, 0, 0])
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
    ra_ticks = m2.drawmeridians(meridians, labels=[0, 0, 0, 1])

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

    sm = plt.cm.ScalarMappable(norm=n, cmap=plt.cm.Greys)
    sm.set_array([])  # can be an empty list

    tks = np.linspace(min_gal_weighted_prob, max_gal_weighted_prob, 5)
    # tks = np.logspace(np.log10(min_prob), np.log10(max_prob), 11)
    tks_strings = []

    for t in tks:
        tks_strings.append('%0.2f' % (t * 100))

    cb = fig.colorbar(sm, ax=ax2, ticks=tks, orientation='vertical', fraction=0.04875, pad=0.02,
                      alpha=0.80)  # 0.08951
    cb.ax.set_yticklabels(tks_strings, fontsize=16)
    cb.set_label("2D Pixel Probability", fontsize=label_size, labelpad=9.0)

    cb.ax.tick_params(width=2.0, length=6.0)

    cb.outline.set_linewidth(2.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(2.0)

    ax2.invert_xaxis()

    ax2.set_ylabel(r'$\mathrm{Declination}$', fontsize=label_size, labelpad=36)
    ax2.set_xlabel(r'$\mathrm{Right\;Ascension}$', fontsize=label_size, labelpad=30)

    ax2.legend(loc='upper left', fontsize=sm_label_size, borderpad=0.35, handletextpad=0.0, labelspacing=0.4)









    fig.savefig('Proposal_Localization.eps', bbox_inches='tight')  # ,dpi=840
    plt.close('all')
    print("... Done.")



if mag_dist_and_fractional_completeness:

    cosmo_high = LambdaCDM(H0=20.0, Om0=0.27, Ode0=0.73)
    cosmo_low = LambdaCDM(H0=140.0, Om0=0.27, Ode0=0.73)

    northern_95th_pixel_ids = path_format.format(ps1_strm_dir, "northern_95th_pixel_ids.txt")
    southern_95th_pixel_ids = path_format.format(ps1_strm_dir, "southern_95th_pixel_ids.txt")
    _50th_pixel_id_file = path_format.format(ps1_strm_dir, "50th_pixel_ids.txt")

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

    _50th_pixel_ids = []
    with open(_50th_pixel_id_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        next(csvreader)  # skip header

        for row in csvreader:
            id = row[0]
            _50th_pixel_ids.append(id)

    pixel_result_50 = query_db([pixel_select % ",".join(_50th_pixel_ids)])[0]
    print("Total NSIDE=1024 pixels in 50th: %s" % len(pixel_result_50))

    map_pix_north = []
    map_pix_north_dict = {}
    map_pix_50_dict = {}

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
        map_pix_north_dict[index] = p

    for m in pixel_result_50:
        pix_id = int(m[0])
        index = int(m[2])
        prob = float(m[3])
        dist = float(m[7])
        stddev = float(m[8])
        p = Pixel_Element(index, map_nside, prob, pixel_id=pix_id, mean_dist=dist, stddev_dist=stddev)

        map_pix_50_dict[index] = p

    for m in pixel_result_south:
        pix_id = int(m[0])
        index = int(m[2])
        prob = float(m[3])
        dist = float(m[7])
        stddev = float(m[8])
        p = Pixel_Element(index, map_nside, prob, pixel_id=pix_id, mean_dist=dist, stddev_dist=stddev)
        map_pix_south.append(p)
        all_map_pix.append(p)

    # region Compute Volume Information for the North where we have PS1 information
    map_pix_dist = np.asarray([mp.mean_dist for mp in map_pix_north])
    map_pix_dist_far = np.asarray([mp.mean_dist + 2.0 * mp.stddev_dist for mp in map_pix_north])
    map_pix_dist_near = np.asarray([mp.mean_dist - 2.0 * mp.stddev_dist for mp in map_pix_north])

    map_pix_z_limits = {}
    for mp in map_pix_north:
        max_dist = mp.mean_dist + 2.0 * mp.stddev_dist
        min_dist = mp.mean_dist - 2.0 * mp.stddev_dist

        # Stretch range with cosmologies with H0 ranging from 20 to 140...
        min_z = z_at_value(cosmo_high.luminosity_distance, min_dist * u.Mpc)
        max_z = z_at_value(cosmo_low.luminosity_distance, max_dist * u.Mpc)

        if mp.id not in map_pix_z_limits:
            map_pix_z_limits[mp.id] = (min_z, max_z, min_dist, max_dist)

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
            synth_B2 = float(row[4])
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
                pg = (ps1_galaxy_id, gaia_ra, gaia_dec, synth_B1, synth_B2, z_phot0, z_photErr, PS1_z_dist,
                      ps1_hp_id, N128_SkyPixel_id, prob_Galaxy, kron_g, kron_r, kron_i, kron_z, kron_y)
                ps1_galaxies.append(pg)

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

    two_cuts_r = []
    two_cuts_r_50th = []
    two_cuts_r_95th = []
    for i in range(len(prob_Gal)):
        p = prob_Gal[i]
        r = r_kron_mags[i]
        z = zs[i]

        if p >= 0.9 and z <= 0.2:

            two_cuts_r.append(r)

            gal = ps1_galaxies[i]
            ra = gal[1]
            dec = gal[2]
            c = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg))
            index = hp.ang2pix(1024, 0.5 * np.pi - c.dec.radian, c.ra.radian)

            if index in map_pix_north_dict:
                two_cuts_r_95th.append(gal)

            if index in map_pix_50_dict:
                two_cuts_r_50th.append(gal)

    print("Number of galaxies with prob_Galaxy >= 0.9 AND z <= 0.2: %s" % len(two_cuts_r))
    print("Number of galaxies in `two_cuts_r_95th`: %s" % len(two_cuts_r_95th))
    print("Number of galaxies in `two_cuts_r_50th`: %s" % len(two_cuts_r_50th))

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

    #### Figuring out the percentage of the full sample that we think we can get by percentile membership
    h = 0.7
    phi = 1.6e-2 * h ** 3  # +/- 0.3 Mpc^-3
    a = -1.07  # +/- 0.07
    L_B_star = 1.2e+10 / h ** 2  # +/- 0.1

    _95_denominator_gals = [g for g_id, g in northern_95th_AAOmega_galaxy_dict_by_galaxy_id.items()]
    _50_denominator_gals = []
    for g in _95_denominator_gals:
        if g.pix_index in _50_AAOmega_pixel_dict:
            _50_denominator_gals.append(g)


    _95_numerator_gals = []
    _50_numerator_gals = []

    for gi in all_gal_ids:
        gal = northern_95th_AAOmega_galaxy_dict_by_galaxy_id[gi]

        if gal.pix_index in _50_AAOmega_pixel_dict:
            _50_numerator_gals.append(gal)

        if gal.pix_index in northern_95th_AAOmega_galaxy_dict_by_pixel_index:
            _95_numerator_gals.append(gal)

    _95th_denominator_luminosities = []
    for g in _95_denominator_gals:
        synth_B2 = g.synth_B
        z_dist = cosmo.luminosity_distance(g.z).value

        L_Sun__L_star = 10 ** (-0.4 * ((synth_B2 - (5 * np.log10(z_dist * 1e+6) - 5)) - 5.48)) / L_B_star
        _95th_denominator_luminosities.append(L_Sun__L_star)

    _50th_denominator_luminosities = []
    for g in _50_denominator_gals:
        synth_B2 = g.synth_B
        z_dist = cosmo.luminosity_distance(g.z).value

        L_Sun__L_star = 10 ** (-0.4 * ((synth_B2 - (5 * np.log10(z_dist * 1e+6) - 5)) - 5.48)) / L_B_star
        _50th_denominator_luminosities.append(L_Sun__L_star)


    _95th_numerator_luminosities = []
    for g in _95_numerator_gals:
        synth_B2 = g.synth_B
        z_dist = cosmo.luminosity_distance(g.z).value

        L_Sun__L_star = 10 ** (-0.4 * ((synth_B2 - (5 * np.log10(z_dist * 1e+6) - 5)) - 5.48)) / L_B_star
        _95th_numerator_luminosities.append(L_Sun__L_star)

    _50th_numerator_luminosities = []
    for g in _50_numerator_gals:
        synth_B2 = g.synth_B
        z_dist = cosmo.luminosity_distance(g.z).value

        L_Sun__L_star = 10 ** (-0.4 * ((synth_B2 - (5 * np.log10(z_dist * 1e+6) - 5)) - 5.48)) / L_B_star
        _50th_numerator_luminosities.append(L_Sun__L_star)

    y_95_denominator, binEdges_95_denominator = np.histogram(np.log10(_95th_denominator_luminosities), bins=np.linspace(-6.0, 2.0, 45))
    bincenters_95_denominator = 0.5 * (binEdges_95_denominator[1:] + binEdges_95_denominator[:-1])

    y_50_denominator, binEdges_50_denominator = np.histogram(np.log10(_50th_denominator_luminosities), bins=np.linspace(-6.0, 2.0, 45))
    bincenters_50_denominator = 0.5 * (binEdges_50_denominator[1:] + binEdges_50_denominator[:-1])

    y_95_numerator, binEdges_95_numerator = np.histogram(np.log10(_95th_numerator_luminosities), bins=np.linspace(-6.0, 2.0, 45))
    bincenters_95_numerator = 0.5 * (binEdges_95_numerator[1:] + binEdges_95_numerator[:-1])

    y_50_numerator, binEdges_50_numerator = np.histogram(np.log10(_50th_numerator_luminosities), bins=np.linspace(-6.0, 2.0, 45))
    bincenters_50_numerator = 0.5 * (binEdges_50_numerator[1:] + binEdges_50_numerator[:-1])

    _95_comp_bins = []
    _50_comp_bins = []
    _95_comp_ratios = []
    _50_comp_ratios = []

    log_LB_LStar_efficiency_thresh = -1.3439626432899594
    for i, bs in enumerate(bincenters_95_denominator):
        _95_denominator_interval = y_95_denominator[i]
        _50_denominator_interval = y_50_denominator[i]
        _95_numerator_interval = y_95_numerator[i]
        _50_numerator_interval = y_50_numerator[i]

        _95_comp_ratio = 0
        if _95_denominator_interval > 0 and _95_numerator_interval > 0:
            if bs < log_LB_LStar_efficiency_thresh:
                _95_comp_ratio = (0.7*_95_numerator_interval) / _95_denominator_interval
            else:
                _95_comp_ratio = _95_numerator_interval / _95_denominator_interval

        _50_comp_ratio = 0
        if _50_denominator_interval > 0 and _50_numerator_interval > 0:
            if bs < log_LB_LStar_efficiency_thresh:
                _50_comp_ratio = (0.7*_50_numerator_interval) / _50_denominator_interval
            else:
                _50_comp_ratio = _50_numerator_interval / _50_denominator_interval


        if _95_denominator_interval > 0 and bs > -5:
            _95_comp_bins.append(bs)
            _95_comp_ratios.append(_95_comp_ratio)
        elif _95_denominator_interval == 0 and bs <= -5:
            _95_comp_bins.append(bs)
            _95_comp_ratios.append(_95_comp_ratio)

        if _50_denominator_interval > 0 and bs > -4:
            _50_comp_bins.append(bs)
            _50_comp_ratios.append(_50_comp_ratio)
        elif _50_denominator_interval == 0 and bs <= -4:
            _50_comp_bins.append(bs)
            _50_comp_ratios.append(_50_comp_ratio)

    from scipy.signal import savgol_filter

    x1 = np.linspace(np.min(_50_comp_bins), np.max(_50_comp_bins), 100)
    itp1 = interp1d(_50_comp_bins, _50_comp_ratios, kind='linear')
    window_size1, poly_order1 = 5, 3
    # window_size, poly_order = 3, 2
    yy_sg1 = savgol_filter(itp1(x1), window_size1, poly_order1)

    x2 = np.linspace(np.min(_95_comp_bins), np.max(_95_comp_bins), 100)
    itp2 = interp1d(_95_comp_bins, _95_comp_ratios, kind='linear')
    # window_size2, poly_order2 = 7, 5
    window_size2, poly_order2 = 7, 3
    # window_size2, poly_order2 = 21, 3

    yy_sg2 = savgol_filter(itp2(x2), window_size2, poly_order2)






    fig = plt.figure(figsize=(21, 10), dpi=1000)
    # ax1 = fig.add_subplot(221)
    # n1, bins1, patches1 = ax1.hist(zs, histtype='step', bins=np.linspace(0, 1, 20), color="black")
    # ax1.set_xlabel("Photo z")
    # ax1.set_ylabel("Count")
    #
    # ax2 = fig.add_subplot(222)
    # n2, bins2, patches2 = ax2.hist(prob_Gal, histtype='step', bins=np.linspace(0.7, 1, 20), color="dodgerblue")
    # ax2.set_xlabel("prob_Galaxy")
    # ax2.set_ylabel("Count")

    # ax3 = fig.add_subplot(223)
    ax3 = fig.add_subplot(121)

    # left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]
    # ax4 = fig.add_axes([left, bottom, width, height])

    # n3, bins3, patches3 = ax3.hist(r_kron_mags, histtype='step', bins=np.linspace(14, 22, 20), color="k")

    # n3, bins3, patches3 = ax3.hist(r_kron_mags, histtype='step', bins=np.linspace(14, 22, 17), color="k")
    n3, bins3, patches3 = ax3.hist(two_cuts_r, histtype='step', bins=np.linspace(14, 22, 17), color="k")

    # ax1.hist(all_cuts_z, histtype='stepfilled', bins=bins1, color='black')
    # ax2.hist(all_cuts_p, histtype='stepfilled', bins=bins2, color='dodgerblue')
    # ax3.hist(all_cuts_r, histtype='stepfilled', bins=bins3, color='red')

    # ps1_galaxies_by_mag_bin[_20_0_20_5] = ([], "red")
    # ps1_galaxies_by_mag_bin[_20_5_21_0] = ([], "red")
    # ps1_galaxies_by_mag_bin[_21_0_21_5] = ([], "green")
    # ps1_galaxies_by_mag_bin[_21_5_22_0] = ([], "blue")

    all_cuts_arr = np.asarray(all_cuts_r)

    bright_i = np.where((all_cuts_arr < 20.5))[0]
    bright_r = all_cuts_arr[bright_i]
    ax3.hist(bright_r, histtype='stepfilled', bins=bins3, color='red', label="Exp Time: 40 min")
    ax3.hist(bright_r, histtype='step', bins=bins3, color='k')

    mid_i = np.where((all_cuts_arr >= 20.5) & (all_cuts_arr < 21.0))[0]
    mid_r = all_cuts_arr[mid_i]
    ax3.hist(mid_r, histtype='stepfilled', bins=bins3, color='green', label="Exp Time: 80 min")
    ax3.hist(mid_r, histtype='step', bins=bins3, color='k')

    dim_i = np.where((all_cuts_arr >= 21.0) & (all_cuts_arr < 22.0))[0]
    dim_r = all_cuts_arr[dim_i]
    ax3.hist(dim_r, histtype='stepfilled', bins=bins3, color='blue', label="Exp Time: 120 min")
    ax3.hist(dim_r, histtype='step', bins=bins3, color='k')


    ax3.set_xlabel(r"$\mathrm{r_{kron}}$ [mag]", fontsize=24)
    ax3.set_ylabel("Number of Galaxies", fontsize=24)
    ax3.tick_params(axis='both', which='major', labelsize=24, length=8.0, width=2)
    ax3.legend(loc='upper left', fontsize=18, borderpad=0.35, handletextpad=0.0, labelspacing=0.4)

    # for key in osDES_keylist:
    #
    #     if key[1] == Q4:
    #         continue
    #     if key[0] not in [_20_5_21_0, _21_0_21_5, _21_5_22_0]:
    #         continue
    #
    #     model = ozDES_models[key[0]][key[1]]
    #     exp_time = ozDES_data[key[0]][key[1]][0]
    #     success = np.asarray(ozDES_data[key[0]][key[1]][1]) * 100.0
    #
    #     # find time to >= 80%
    #     model_success = model(model_time)
    #     thresh = 0.75
    #     thresh_txt = thresh * 100
    #     time_for_bin_to_thresh_success = next(model_time[i] for i, s in enumerate(model_success) if s >= thresh)
    #
    #     # addendum = ""
    #     # if key[0] == _20_0_20_5:
    #     #     pass
    #     # elif key[0] == _20_5_21_0:
    #     #     addendum += "; 80 min => %0.0f%%" % float(model(80.0) * 100.0)
    #     # elif key[0] == _21_0_21_5:
    #     #     addendum += "; 120 min => %0.0f%%" % float(model(120.0) * 100.0)
    #     # elif key[0] == _21_5_22_0:
    #     #     addendum += "; 160 min => %0.0f%%" % float(model(160.0) * 100.0)
    #
    #     # Plot data
    #     if key[4] != "":
    #         lbl = key[4] + "; %0.0f min => %0.0f%%" % (time_for_bin_to_thresh_success, thresh_txt) + addendum
    #         ax4.plot(exp_time, success, color=key[2], linestyle=key[3], label=lbl, alpha=1.0)
    #     else:
    #         ax4.plot(exp_time, success, color=key[2], linestyle=key[3], alpha=1.0)
    #
    #     # plot model -- only good until 250 seconds
    #     ax4.plot(model_time, model_success * 100, color=key[2], linestyle='--', alpha=0.3)
    #     ax4.set_xlim([35, 130])
    #     ax4.set_ylim([40, 100])
    #
    # ax4.vlines(40.0, 30, 103, colors='k', linestyles='--', label="Min ExpTime: 40 min")
    # ax4.set_xlabel("Exposure time (minutes)")
    # ax4.set_ylabel("Redshift Completeness (%)")


    ax2 = fig.add_subplot(122)

    # ax.plot(_95_comp_bins, _95_comp_ratios, '+', color='red')
    # ax.plot(_50_comp_bins, _50_comp_ratios, '+', color='red')
    ax2.plot(x1, yy_sg1, '-', color='orange', label="50th percentile", linewidth=4)
    ax2.plot(x2, yy_sg2, '-', color='green', label="95th percentile", linewidth=4)

    ax2.set_xlim([-6.0, 2.0])
    ax2.set_xticks([-6, -5, -4, -3, -2, -1, 0, 1])
    ax2.set_xlabel(r"Log($L_B/L^{*}_{B}$)", fontsize=24)
    ax2.set_ylabel("Redshift Completness", fontsize=24)
    ax2.tick_params(axis='both', which='major', labelsize=24, length=8.0, width=2)
    ax2.legend(loc='upper left', fontsize=18, borderpad=0.35, handletextpad=0.0, labelspacing=0.4)


    fig.savefig("Proposal_sample_stats.eps", bbox_inches='tight')
    plt.close('all')



if H0_calc:


    def get_confidence_intervals(x_arr, y_arr):

        # print("***")
        # print(y_arr)
        # print("---")

        delta_input = x_arr[1] - x_arr[0]
        threshold_16 = 0.16
        threshold_50 = 0.50
        threshold_84 = 0.84

        running_prob = 0.0
        index_of_16 = -1
        index_of_50 = -1
        index_of_84 = -1
        found_16 = False
        found_50 = False

        for i, p in enumerate(y_arr):

            running_prob += p * delta_input
            if running_prob >= threshold_16 and not found_16:
                found_16 = True
                index_of_16 = i

            if running_prob >= threshold_50 and not found_50:
                found_50 = True
                index_of_50 = i

            if running_prob >= threshold_84:
                index_of_84 = i
                break

        if index_of_16 == -1 or index_of_50 == -1 or index_of_84 == -1:
            print("\n\n************************************")
            print(x_arr)
            print(y_arr)
            print(i)
            print(index_of_16)
            print(index_of_50)
            print(index_of_84)
            raise Exception("Could not find indices!")

        median_xval = x_arr[index_of_50]
        xval_lower_bound = x_arr[index_of_16]
        xval_upper_bound = x_arr[index_of_84]
        frac_measurement = 100 * (xval_upper_bound - xval_lower_bound) / (2 * median_xval)


        print("Area under 1-sigma curve: %s" % trapz(y_arr[index_of_16:index_of_84], x_arr[index_of_16:index_of_84]))

        return xval_lower_bound, index_of_16, median_xval, index_of_50, xval_upper_bound, index_of_84, frac_measurement

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    start = time.time()

    # Global variables
    c = 2.998e+5    # km/s
    D0 = 267.       # Mpc
    D0_err = 52.    # Mpc

    D0_dist = norm(loc=D0, scale=D0_err)
    print("Original Distance, Err: %s +/- %s Mpc" % (D0, D0_err))

    # northern_95th_AAOmega_galaxy_dict_by_galaxy_id = {}
    # northern_95th_AAOmega_galaxy_dict_by_pixel_index = {}
    # northern_AAOmega_pixel_dict[index] = p

    spec_z = []
    for gi in all_gal_ids:
        gal = northern_95th_AAOmega_galaxy_dict_by_galaxy_id[gi]
        spec_z.append(gal.z)

    print("min z: %s" % min(spec_z))
    print("max z: %s" % max(spec_z))

    spec_z = np.asarray([0.108100, 0.144100, 0.189700, 0.181100, 0.199100, 0.0841000, 0.193900, 0.151700])
    spec_z_weight = np.asarray([0.347369, 0.379116, 0.780225, 0.415930, 0.546455, 0.619116, 0.816442, 0.642831])

    total_weight = sum(spec_z_weight)
    relative_weights = spec_z_weight/total_weight

    model_z_err = 1000/c

    z_dists = []
    for i, z in enumerate(spec_z):
        z_dist = norm(loc=z, scale=model_z_err)
        z_dists.append((z_dist, relative_weights[i]))

    z_input = np.linspace(0, 0.2, 500)
    z_sum = np.zeros(len(z_input))
    for z_tuple in z_dists:
        dist = z_tuple[0]
        prob = z_tuple[1]

        result = prob * dist.pdf(z_input)
        z_sum += result

    z_norm = z_sum / trapz(z_sum, z_input)  # normalize

    csz = np.cumsum(z_norm)
    cszn = csz/csz[-1]





    # samples = []
    # for i in range(11000):
    #     ri = random.random()
    #     idx = find_nearest(cszn, ri)
    #     samples.append(z_input[idx])

    # fig = plt.figure(figsize=(10, 5), dpi=800)
    # ax1 = fig.add_subplot(121)
    # ax1.plot(z_input, z_norm)
    # ax1.hist(samples, density=True, histtype='stepfilled', alpha=0.2)
    #
    # ax2 = fig.add_subplot(122)
    # ax2.plot(z_input, cszn)
    #
    # fig.savefig("ryan_z_dist_test.png", bbox_inches='tight')
    # plt.close('all')


    z = []
    z_weight = []
    z_err = []
    total_prob = 0
    for pix_index, pixel in northern_AAOmega_pixel_dict.items():
        for gal_id, galaxy in pixel.contained_galaxies.items():
            total_prob += galaxy.prob_fraction

    print("Total prob: %s" % total_prob)

    for pix_index, pixel in northern_AAOmega_pixel_dict.items():
        for gal_id, galaxy in pixel.contained_galaxies.items():
            scaled_prob = galaxy.prob_fraction / total_prob
            z_weight.append(scaled_prob)
            z.append(galaxy.z)
            z_err.append(galaxy.z_photErr)

    z_mean = np.average(z, weights=z_weight)
    print("Weighted mean z: %s" % z_mean)

    spec_z_err = 4.2e-4
    H0_phot_distributions = []
    H0_spec_distributions = []
    for pix_index, pixel in northern_AAOmega_pixel_dict.items():
        for gal_id, galaxy in pixel.contained_galaxies.items():

            prob = galaxy.prob_fraction

            photo_z = galaxy.z
            photo_z_err = galaxy.z_photErr
            H0_phot_i = (c * photo_z) / D0
            H0_phot_i_err = np.sqrt(photo_z_err ** 2 * (c / D0) ** 2 + D0_err ** 2 * (-(c * photo_z) / D0 ** 2) ** 2)
            H0_phot_dist = norm(loc=H0_phot_i, scale=H0_phot_i_err)


            if not (H0_phot_i < 20 or H0_phot_i > 150):
                H0_phot_distributions.append((H0_phot_dist, prob/photo_z**3))



            # get spec z
            ri = random.random()
            idx = find_nearest(cszn, ri)
            spec_z = z_input[idx]
            H0_spec_i = (c * spec_z) / D0
            H0_spec_i_err = np.sqrt(spec_z_err ** 2 * (c / D0) ** 2 + D0_err ** 2 * (-(c * spec_z) / D0 ** 2) ** 2)
            H0_spec_dist = norm(loc=H0_spec_i, scale=H0_spec_i_err)

            if not (H0_spec_i < 20 or H0_spec_i > 150):
                H0_spec_distributions.append((H0_spec_dist, prob/spec_z**3))





    H0_input = np.linspace(20, 150, 500)
    H0_phot_sum = np.zeros(len(H0_input))
    for h0_tuple in H0_phot_distributions:
        dist = h0_tuple[0]
        prob = h0_tuple[1]

        # n = trapz(dist.pdf(H0_input), H0_input)
        # result = prob * (dist.pdf(H0_input)/n)
        result = prob * dist.pdf(H0_input)

        H0_phot_sum += result

    H0_spec_sum = np.zeros(len(H0_input))
    for h0_tuple in H0_spec_distributions:
        dist = h0_tuple[0]
        prob = h0_tuple[1]

        # n = trapz(dist.pdf(H0_input), H0_input)
        # result = prob * (dist.pdf(H0_input) / n)

        result = prob * dist.pdf(H0_input)
        H0_spec_sum += result

    H0_phot_norm = H0_phot_sum / trapz(H0_phot_sum, H0_input)  # normalize
    H0_spec_norm = H0_spec_sum / trapz(H0_spec_sum, H0_input)  # normalize

    phot_min_H0, index_phot_min_H0, phot_median_H0, index_phot_median_H0, phot_max_H0, \
    index_phot_max_H0, percent_phot_measurement = get_confidence_intervals(H0_input, H0_phot_norm)

    spec_min_H0, index_spec_min_H0, spec_median_H0, index_spec_median_H0, spec_max_H0, \
    index_spec_max_H0, percent_spec_measurement = get_confidence_intervals(H0_input, H0_spec_norm)

    y_cutoff = np.min(H0_phot_norm)

    fig = plt.figure(figsize=(10, 10), dpi=800)
    ax1 = fig.add_subplot(111)

    ax1.plot(H0_input, H0_phot_norm, color='k', linewidth=2)
    ax1.vlines(phot_min_H0, y_cutoff, H0_phot_norm[index_phot_min_H0], colors='k', linestyles=':')
    ax1.vlines(phot_median_H0, y_cutoff, H0_phot_norm[index_phot_median_H0], colors='k', linestyles='-',
               label=r"Photo-z H0=%0.2f$^{+%0.2f}_{-%0.2f}$ (%0.2f)%%" %
                     (phot_median_H0, (phot_median_H0 - phot_min_H0),
                      (phot_max_H0 - phot_median_H0), percent_phot_measurement))
    ax1.vlines(phot_max_H0, y_cutoff, H0_phot_norm[index_phot_max_H0], colors='k', linestyles=':')




    ax1.plot(H0_input, H0_spec_norm, color='r', linewidth=2)
    ax1.vlines(spec_min_H0, y_cutoff, H0_spec_norm[index_spec_min_H0], colors='r', linestyles=':')
    ax1.vlines(spec_median_H0, y_cutoff, H0_spec_norm[index_spec_median_H0], colors='r', linestyles='-',
               label=r"Spec-z H0=%0.2f$^{+%0.2f}_{-%0.2f}$ (%0.2f)%%" %
                     (spec_median_H0, (spec_median_H0 - spec_min_H0),
                      (spec_max_H0 - spec_median_H0), percent_spec_measurement))
    ax1.vlines(spec_max_H0, y_cutoff, H0_spec_norm[index_spec_max_H0], colors='r', linestyles=':')

    ax1.set_ylabel(r"P($\mathrm{H_0}$)", fontsize=24)
    ax1.set_xlabel(r"$\mathrm{H_0}$ [km s$^-1$ Mpc$^-1$]", fontsize=24)

    ax1.tick_params(axis='both', which='major', labelsize=18, length=8.0, width=2)

    ax1.set_ylim(ymin=y_cutoff)
    ax1.legend()

    fig.savefig("Proposal_H0_calc.eps", bbox_inches='tight')
    plt.close('all')

    end = time.time()
    duration = (end - start)
    print("\n********* start DEBUG ***********")
    print("Execution time: %s" % duration)
    print("********* end DEBUG ***********\n")