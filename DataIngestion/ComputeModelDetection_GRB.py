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
# GW190814_t_0 = 58709.882824224536  # time of GW190814 merger
GW190814_t_0 = 58598.346134259256  # time of GW190425 merger

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

# Multiprocessing methods...
def initial_z(pixel_data):
    cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

    pix_index = pixel_data[0]
    mean_dist = pixel_data[1]

    d = Distance(mean_dist, u.Mpc)
    z = d.compute_z(cosmology=cosmo)

    return (pix_index, z)

def integrate_pixel(pixel_data):

    reverse_band_mapping_new = {
        "SDSS g": "sdss_g",
        "SDSS r": "sdss_r",
        "SDSS i": "sdss_i",
        "Clear": "Clear"
    }

    pix_key = pixel_data[0]
    pix_index = pix_key[0]
    model_key = pix_key[1]

    mean_dist = pixel_data[1]
    dist_sigma = pixel_data[2]
    mwe = pixel_data[3]
    prob_2D = pixel_data[4]
    band = pixel_data[5]
    mjd = pixel_data[6]
    lim_mag = pixel_data[7]
    model_func_dict = pixel_data[8]
    delta_mjd = pixel_data[9]

    f_band = model_func_dict[reverse_band_mapping_new[band]]
    abs_in_band = f_band(delta_mjd)

    # compute distance upper bound, given the limiting magnitude:
    pwr = (lim_mag - abs_in_band + 5.0 - mwe) / 5.0 - 6.0
    d_upper = 10 ** pwr
    prob_to_detect = 0.0

    if d_upper > 0.0:
        # cdf = lambda d: norm.cdf(d, mean_dist, dist_sigma)
        # dist_norm = cdf(np.inf) - cdf(0.0) # truncating integral at d = 0 Mpc
        # prob_to_detect = (prob_2D/dist_norm) * (cdf(d_upper) - cdf(0.0))

        prob_to_detect = prob_2D * (0.5 * erf((d_upper - mean_dist) / (np.sqrt(2) * dist_sigma)) - \
                         0.5 * erf(-mean_dist / (np.sqrt(2) * dist_sigma)))

    new_pix_key = (pix_index, model_key, band, mjd)
    return (new_pix_key, prob_to_detect)

class Teglon:

    def add_options(self, parser=None, usage=None, config=None):
        import optparse
        if parser == None:
            parser = optparse.OptionParser(usage=usage, conflict_handler="resolve")

        parser.add_option('--gw_id', default="", type="str", help='LIGO superevent name, e.g. `S190425z`.')

        parser.add_option('--healpix_file', default="", type="str",
                          help='Healpix filename. Used with `gw_id` to identify unique map.')

        parser.add_option('--healpix_dir', default='../Events/{GWID}', type="str",
                          help='Directory for where to look for the healpix file.')

        parser.add_option('--model_output_dir', default="../Events/{GWID}/ModelDetection", type="str",
                          help='Directory for where to output processed models.')

        parser.add_option('--num_cpu', default="6", type="int",
                          help='Number of CPUs to use for multiprocessing')

        parser.add_option('--sub_dir', default="", type="str",
                          help='GRB Model sub directory (for batching)')

        parser.add_option('--batch_dir', default="", type="str",
                          help='GRB Model sub sub directory (for batching... lol)')

        return (parser)

    def main(self):
        print("Processes to use: %s" % self.options.num_cpu)

        # region Sanity/Parameter checks
        prep_start = time.time()

        is_error = False

        # Parameter checks
        if self.options.gw_id == "":
            is_error = True
            print("GWID is required.")

        if self.options.healpix_file == "":
            is_error = True
            print("Healpix file is required.")

        if is_error:
            print("Exiting...")
            return 1

        formatted_healpix_dir = self.options.healpix_dir
        if "{GWID}" in formatted_healpix_dir:
            formatted_healpix_dir = formatted_healpix_dir.replace("{GWID}", self.options.gw_id)

        formatted_model_output_dir = self.options.model_output_dir
        if "{GWID}" in formatted_model_output_dir:
            formatted_model_output_dir = formatted_model_output_dir.replace("{GWID}", self.options.gw_id)

        hpx_path = "%s/%s" % (formatted_healpix_dir, self.options.healpix_file)
        base_grb_model_path = "../GRBModels/%s" % self.options.sub_dir
        grb_model_path = base_grb_model_path + "/%s" % self.options.batch_dir

        grb_model_files = []
        for file_index, file in enumerate(os.listdir(grb_model_path)):

            if file.endswith(".dat"):
                grb_model_files.append("%s/%s" % (grb_model_path, file))

        if len(grb_model_files) <= 0:
            is_error = True
            print("There are no models to process.")

        # Check if the above files exist...
        if not os.path.exists(hpx_path):
            is_error = True
            print("Healpix file `%s` does not exist." % hpx_path)

        if is_error:
            print("Exiting...")
            return 1
        # endregion

        # region Convienence Dictionaries
        # Band abbreviation, band_id mapping
        band_mapping_new = {
            "sdss_g": "SDSS g",
            "sdss_r": "SDSS r",
            "sdss_i": "SDSS i",
            "Clear": "Clear"
        }

        reverse_band_mapping_new = {
            "SDSS g": "sdss_g",
            "SDSS r": "sdss_r",
            "SDSS i": "sdss_i",
            "Clear": "Clear"
        }

        detector_mapping = {
            "s": "SWOPE",
            "t": "THACHER",
            "a": "ANDICAM",
            "n": "NICKEL",
            "m": "MOSFIRE",
            "k": "KAIT",
            "si": "SINISTRO"
        }
        # endregion

        # region Load Serialized Sky Pixels, EBV, and Models from Disk
        # LOADING NSIDE 128 SKY PIXELS AND EBV INFORMATION
        print("\nLoading NSIDE 128 pixels...")
        nside128 = 128
        N128_dict = None
        with open('N128_dict.pkl', 'rb') as handle:
            N128_dict = pickle.load(handle)
        del handle

        print("\nLoading existing EBV...")
        ebv = None
        with open('ebv.pkl', 'rb') as handle:
            ebv = pickle.load(handle)

        models = {}
        for index, mf in enumerate(grb_model_files):
            model_table = Table.read(mf, format='ascii.ecsv')
            mask = model_table['time'] >= 0.0

            model_time = np.asarray(model_table['time'][mask])
            g = np.asarray(model_table['sdss_g'][mask])
            r = np.asarray(model_table['sdss_r'][mask])
            i = np.asarray(model_table['sdss_i'][mask])
            clear = np.asarray(model_table['Clear'][mask])
            # clear = np.asarray(model_table['sdss_r'][mask])

            model_props = model_table.meta['comment']
            model_type = model_props[0].split("=")[1]
            E = float(model_props[1].split("=")[1])
            n = float(model_props[2].split("=")[1])
            theta_obs = float(model_props[3].split("=")[1])

            # ########################################
            # # Hack to test linear models. Remove. DC
            # model_props = model_table.meta['comment']
            # M = float(model_props[0].split("=")[1])
            # dM = float(model_props[1].split("=")[1])
            # ########################################

            base_name = os.path.basename(mf)
            print("Loading `%s`" % base_name)

            # Get interpolation function for each Light Curve
            f_g = interp1d(model_time, g, fill_value="extrapolate")
            f_r = interp1d(model_time, r, fill_value="extrapolate")
            f_i = interp1d(model_time, i, fill_value="extrapolate")
            f_clear = interp1d(model_time, clear, fill_value="extrapolate")

            models[(E, n, theta_obs)] = {
                'sdss_g': f_g,
                'sdss_r': f_r,
                'sdss_i': f_i,
                'Clear': f_clear
            }

            # ########################################
            # # Hack to test linear models. Remove. DC
            # models[(M, dM)] = {
            #     'time': np.asarray([]),
            #     'sdss_g': f_g,
            #     'sdss_r': f_r,
            #     'sdss_i': f_i,
            #     'Clear': f_clear
            # }
            # ########################################
        # endregion

        # region Get Map, Bands and initialize pixels.
        # Get Map ID
        print("\nLoading Healpix Map...")
        healpix_map_select = "SELECT id, NSIDE FROM HealpixMap WHERE GWID = '%s' and Filename = '%s'"
        healpix_map_id = int(query_db([healpix_map_select % (self.options.gw_id, self.options.healpix_file)])[0][0][0])
        healpix_map_nside = int(
            query_db([healpix_map_select % (self.options.gw_id, self.options.healpix_file)])[0][0][1])

        # Get Bands
        print("\nLoading Configured Bands...")
        band_select = "SELECT id, Name, F99_Coefficient FROM Band"
        bands = query_db([band_select])[0]
        band_dict_by_name = {}
        band_dict_by_id = {}
        for b in bands:
            b_id = int(b[0])
            b_name = b[1]
            b_coeff = float(b[2])

            band_dict_by_name[b_name] = (b_id, b_name, b_coeff)
            band_dict_by_id[b_id] = (b_id, b_name, b_coeff)

        print("\nRetrieving distinct, imaged map pixels")
        map_pixel_select = '''
        SELECT 
            DISTINCT hp.id, 
            hp.HealpixMap_id, 
            hp.Pixel_Index, 
            hp.Prob, 
            hp.Distmu, 
            hp.Distsigma, 
            hp.Distnorm, 
            hp.Mean, 
            hp.Stddev, 
            hp.Norm, 
            sp.Pixel_Index as N128_Pixel_Index 
        FROM 
            HealpixPixel hp 
        JOIN ObservedTile_HealpixPixel ot_hp on ot_hp.HealpixPixel_id = hp.id 
        JOIN ObservedTile ot on ot.id = ot_hp.ObservedTile_id 
        JOIN SkyPixel sp on sp.id = hp.N128_SkyPixel_id 
        WHERE
            ot.HealpixMap_id = %s and 
            ot.Mag_Lim IS NOT NULL 
        '''

        q = map_pixel_select % healpix_map_id
        map_pixels = query_db([q])[0]
        print("Retrieved %s map pixels..." % len(map_pixels))

        # Initialize map pix dict for later access
        map_pixel_dict_new = OrderedDict()

        class Pixel_Synopsis():
            def __init__(self, mean_dist, dist_sigma, prob_2D, pixel_index, N128_index, pix_ebv):  # z
                self.mean_dist = mean_dist
                self.dist_sigma = dist_sigma
                self.forced_norm = 0.0

                self.prob_2D = prob_2D
                self.pixel_index = pixel_index
                self.N128_index = N128_index
                self.pix_ebv = pix_ebv
                self.z = 0.0

                # From the tiles that contain this pixel
                # band:value
                self.measured_bands = []
                self.lim_mags = OrderedDict()
                self.delta_mjds = OrderedDict()

                # From the model (only select the bands that have been imaged)
                self.A_lambda = OrderedDict()  # band:value

                # Final calculation
                # model:band:value
                self.best_integrated_probs = OrderedDict()

            def __str__(self):
                return str(self.__dict__)

        count_bad_pixels = 0

        initial_integrands = []
        for p in map_pixels:
            mean_dist = float(p[7])
            dist_sigma = float(p[8])
            prob_2D = float(p[3])
            pixel_index = int(p[2])
            N128_pixel_index = int(p[10])
            pix_ebv = ebv[N128_pixel_index]

            if mean_dist == 0.0:
                # distance did not converge for this pixel. pass...
                print("Bad Index: %s" % pixel_index)
                count_bad_pixels += 1
                continue

            p_new = Pixel_Synopsis(
                mean_dist,
                dist_sigma,
                prob_2D,
                pixel_index,
                N128_pixel_index,
                pix_ebv)

            initial_integrands.append((p_new.pixel_index, p_new.mean_dist, p_new.dist_sigma))
            map_pixel_dict_new[pixel_index] = p_new

        print("Starting z-cosmo pool (%s distances)..." % len(initial_integrands))
        it1 = time.time()

        pool1 = mp.Pool(processes=self.options.num_cpu, maxtasksperchild=100)
        resolved_z = pool1.imap_unordered(initial_z, initial_integrands, chunksize=1000)

        pool1.close()
        pool1.join()
        del pool1

        for rz in resolved_z:
            map_pixel_dict_new[rz[0]].z = rz[1]
        it2 = time.time()

        print("... finished z-cosmo pool: %s [seconds]" % (it2 - it1))

        print("\nMap pixel dict complete. %s bad pixels." % count_bad_pixels)
        # endregion

        # region Get Detectors
        detectors = []
        print("\nLoading Swope...")
        detector_select = "SELECT id, Name, Deg_width, Deg_width, Deg_radius, Area, MinDec, MaxDec FROM Detector WHERE `Name`='%s'"
        dr = query_db([detector_select % 'SWOPE'])[0][0]
        swope = Detector(dr[1], float(dr[2]), float(dr[2]), detector_id=int(dr[0]))
        detectors.append(swope)

        print("\nLoading Thacher...")
        dr = query_db([detector_select % 'THACHER'])[0][0]
        thacher = Detector(dr[1], float(dr[2]), float(dr[2]), detector_id=int(dr[0]))
        detectors.append(thacher)

        print("\nLoading Nickel...")
        dr = query_db([detector_select % 'NICKEL'])[0][0]
        nickel = Detector(dr[1], float(dr[2]), float(dr[2]), detector_id=int(dr[0]))
        detectors.append(nickel)

        print("\nLoading KAIT...")
        dr = query_db([detector_select % 'KAIT'])[0][0]
        kait = Detector(dr[1], float(dr[2]), float(dr[2]), detector_id=int(dr[0]))
        detectors.append(kait)

        print("\nLoading SINISTRO...")
        dr = query_db([detector_select % 'SINISTRO'])[0][0]
        sinistro = Detector(dr[1], float(dr[2]), float(dr[2]), detector_id=int(dr[0]))
        detectors.append(sinistro)
        # endregion

        # region Load Tiles
        # Get and instantiate all observed tiles
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
                Detector_id = %s and 
                Mag_Lim IS NOT NULL 
        '''

        observed_tiles = []

        print("\nLoading Swope's Observed Tiles...")
        ot_result = query_db([observed_tile_select % (healpix_map_id, swope.id)])[0]
        for ot in ot_result:
            t = Tile(float(ot[3]), float(ot[4]), swope.deg_width, swope.deg_height, healpix_map_nside,
                     tile_id=int(ot[0]))
            t.field_name = ot[2]
            t.mjd = float(ot[8])
            t.mag_lim = float(ot[10])
            t.band_id = int(ot[7])

            observed_tiles.append(t)
        print("Loaded %s %s tiles..." % (len(ot_result), swope.name))

        print("\nLoading Nickel's Observed Tiles...")
        ot_result = query_db([observed_tile_select % (healpix_map_id, nickel.id)])[0]
        for ot in ot_result:
            t = Tile(float(ot[3]), float(ot[4]), nickel.deg_width, nickel.deg_height, healpix_map_nside,
                     tile_id=int(ot[0]))
            t.field_name = ot[2]
            t.mjd = float(ot[8])
            t.mag_lim = float(ot[10])
            t.band_id = int(ot[7])

            observed_tiles.append(t)
        print("Loaded %s %s tiles..." % (len(ot_result), nickel.name))

        print("\nLoading Thacher's Observed Tiles...")
        ot_result = query_db([observed_tile_select % (healpix_map_id, thacher.id)])[0]
        for ot in ot_result:
            t = Tile(float(ot[3]), float(ot[4]), thacher.deg_width, thacher.deg_height, healpix_map_nside,
                     tile_id=int(ot[0]))
            t.field_name = ot[2]
            t.mjd = float(ot[8])
            t.mag_lim = float(ot[10])
            t.band_id = int(ot[7])

            observed_tiles.append(t)
        print("Loaded %s %s tiles..." % (len(ot_result), thacher.name))

        print("\nLoading KAIT's Observed Tiles...")
        ot_result = query_db([observed_tile_select % (healpix_map_id, kait.id)])[0]
        for ot in ot_result:
            t = Tile(float(ot[3]), float(ot[4]), kait.deg_width, kait.deg_height, healpix_map_nside, tile_id=int(ot[0]))
            t.field_name = ot[2]
            t.mjd = float(ot[8])
            t.mag_lim = float(ot[10])
            t.band_id = int(ot[7])
            observed_tiles.append(t)
        print("Loaded %s %s tiles..." % (len(ot_result), kait.name))

        print("\nLoading SINISTRO's Observed Tiles...")
        ot_result = query_db([observed_tile_select % (healpix_map_id, sinistro.id)])[0]
        for ot in ot_result:
            t = Tile(float(ot[3]), float(ot[4]), sinistro.deg_width, sinistro.deg_height, healpix_map_nside,
                     tile_id=int(ot[0]))
            t.field_name = ot[2]
            t.mjd = float(ot[8])
            t.mag_lim = float(ot[10])
            t.band_id = int(ot[7])
            observed_tiles.append(t)
        print("Loaded %s %s tiles..." % (len(ot_result), sinistro.name))
        # endregion

        print("Getting Detector-Band pairs...")
        detector_band_result = query_db([
            '''SELECT 
                DISTINCT d.Name as Detector, b.Name as Band
            FROM Detector d 
            JOIN ObservedTile ot on ot.Detector_id = d.id
            JOIN Band b on b.id = ot.Band_id
            WHERE ot.HealpixMap_id = %s''' % healpix_map_id
        ])

        print("\nUpdating pixel `delta_mjds` and `lim_mags`...")
        # region Initialize models
        # For each tile:
        #   we want the MJD of observation, and add that to the list of a pixels' MJD collection.
        #   we want the limiting mag, add that to the list of a pixel's lim mag collection
        #   we want to correct the delta_mjd for time dilation
        for t in observed_tiles:
            pix_indices = t.enclosed_pixel_indices

            for i in pix_indices:

                # get band from id...
                band = band_dict_by_id[t.band_id]
                band_name = band[1]

                # Some pixels are omitted because their distance information did not converge
                if i not in map_pixel_dict_new:
                    continue

                pix_synopsis_new = map_pixel_dict_new[i]

                if band_name not in pix_synopsis_new.measured_bands:
                    pix_synopsis_new.measured_bands.append(band_name)
                    pix_synopsis_new.delta_mjds[band_name] = {}
                    pix_synopsis_new.lim_mags[band_name] = {}

                # time-dilate the delta_mjd, which is used to get the Abs Mag LC point
                pix_synopsis_new.delta_mjds[band_name][t.mjd] = (t.mjd - GW190814_t_0) / (1.0 + pix_synopsis_new.z)
                pix_synopsis_new.lim_mags[band_name][t.mjd] = (t.mag_lim)

        print("\nInitializing %s models..." % len(models))
        # endregion

        print("\nUpdating pixel `A_lambda`...")
        # region Update pixel MWE
        # Set pixel `A_lambda`
        for pix_index, pix_synopsis in map_pixel_dict_new.items():
            for model_param_tuple, model_dict in models.items():
                for model_col in model_dict.keys():
                    if model_col in band_mapping_new:

                        band = band_dict_by_name[band_mapping_new[model_col]]
                        band_id = band[0]
                        band_name = band[1]
                        band_coeff = band[2]

                        if band_name in pix_synopsis.measured_bands:
                            if band_name not in pix_synopsis.A_lambda:
                                pix_a_lambda = pix_synopsis.pix_ebv * band_coeff
                                pix_synopsis.A_lambda[band_name] = pix_a_lambda

        prep_end = time.time()
        print("Prep time: %s [seconds]" % (prep_end - prep_start))
        # endregion

        compute_start = time.time()
        print("\nUpdating `map_pixel_dict_new`...")
        count = 0
        for pix_index, pix_synopsis in map_pixel_dict_new.items():
            for band in pix_synopsis.measured_bands:
                for model_param_tuple, model_dict in models.items():
                    pixel_delta_mjd = pix_synopsis.delta_mjds[band]

                    for i, (mjd, delta_mjd) in enumerate(pixel_delta_mjd.items()):
                        if model_param_tuple not in pix_synopsis.best_integrated_probs:
                            pix_synopsis.best_integrated_probs[model_param_tuple] = {}
                        try:
                            pix_synopsis.best_integrated_probs[model_param_tuple][band][mjd] = 0.0
                        except:
                            pix_synopsis.best_integrated_probs[model_param_tuple][band] = {mjd: 0.0}
            count += 1
            if count % 1000 == 0:
                print("Processed: %s" % count)
        compute_end = time.time()
        print("Update `map_pixel_dict_new` time: %s [seconds]" % (compute_end - compute_start))



        # with open('%s/%s_models.pkl' % (formatted_model_output_dir, self.options.gw_id), 'wb') as handle:
        #     pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # with open('%s/%s_map_pixel_dict_new.pkl' % (formatted_model_output_dir, self.options.gw_id), 'wb') as handle:
        #     pickle.dump(map_pixel_dict_new, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        #
        # raise Exception("Stop")
        #
        # models = None
        # map_pixel_dict_new = None
        # with open('%s/%s_models.pkl' % (formatted_model_output_dir, self.options.gw_id), 'rb') as handle:
        #     models = pickle.load(handle)
        #
        # with open('%s/%s_map_pixel_dict_new.pkl' % (formatted_model_output_dir, self.options.gw_id), 'rb') as handle:
        #     map_pixel_dict_new = pickle.load(handle)


        # Compose integrands
        print("Building integrands...")
        integrands_start = time.time()
        pixels_to_integrate = []
        for model_key, model_func_dict in models.items():
            for pix_index, pix_synopsis in map_pixel_dict_new.items():
                for band in pix_synopsis.measured_bands:
                    mwe = pix_synopsis.A_lambda[band]
                    prob_2D = pix_synopsis.prob_2D
                    mean_dist = pix_synopsis.mean_dist
                    dist_sigma = pix_synopsis.dist_sigma

                    for i, (mjd, delta_mjd) in enumerate(pix_synopsis.delta_mjds[band].items()):
                        pix_key = (pix_index, model_key)
                        pixels_to_integrate.append((
                                pix_key,
                                mean_dist,
                                dist_sigma,
                                mwe,
                                prob_2D,
                                band,
                                mjd,
                                pix_synopsis.lim_mags[band][mjd],
                                model_func_dict,
                                delta_mjd
                        ))
        integrands_end = time.time()
        print("Building integrands time: %s [seconds]" % (integrands_end - integrands_start))

        print("Integrating...")
        mp_start = time.time()
        pool3 = mp.Pool(processes=self.options.num_cpu, maxtasksperchild=500)
        integrated_pixels = pool3.imap_unordered(integrate_pixel,
                                                 pixels_to_integrate,
                                                 chunksize=5000)

        for ip in integrated_pixels:
            pix_key = ip[0]
            pix_index = pix_key[0]
            model_param_tuple = pix_key[1]
            band = pix_key[2]
            mjd = pix_key[3]
            prob = ip[1]
            map_pixel_dict_new[pix_index].best_integrated_probs[model_param_tuple][band][mjd] = prob
        mp_end = time.time()

        pool3.close()
        pool3.join()
        del pool3
        print("Integration time: %s [seconds]" % (mp_end - mp_start))
        # endregion












        # region Serialization
        # Finally, get the highest valued integration, and sum
        running_sums = {}  # model:band:value

        for pix_index, pix_synopsis in map_pixel_dict_new.items():
            for band in pix_synopsis.measured_bands:
                for model_param_tuple, model_dict in models.items():

                    pix_max = 0.0
                    if model_param_tuple not in running_sums:
                        running_sums[model_param_tuple] = {}

                    try:
                        t = running_sums[model_param_tuple][band]
                    except:
                        running_sums[model_param_tuple][band] = 0.0

                    probs = []
                    for mjd, integrated_prob in pix_synopsis.best_integrated_probs[model_param_tuple][band].items():
                        probs.append(integrated_prob)

                    pix_max = np.max(probs)
                    running_sums[model_param_tuple][band] += pix_max

        # # Report to stdout the results per model...
        # for model_param_tuple, band_dict in running_sums.items():
        #     print("\nIntegrated prob to detect model (E=%s, n=%s, theta_obs=%s)" % model_param_tuple)
        #
        #     for band, running_sum in band_dict.items():
        #         print("\t%s: %s" % (band, running_sum))

        # ########################################
        # # Hack to test linear models. Remove. DC
        # for model_param_tuple, band_dict in running_sums.items():
        #     print("\nIntegrated prob to detect model (M=%s, dM=%s)" % model_param_tuple)
        #
        #     for band, running_sum in band_dict.items():
        #         print("\t%s: %s" % (band, running_sum))
        # ########################################

        # Serialize results: for every pixel, just get the highest prob
        running_sums2 = {}
        for model_param_tuple, model_dict in models.items():
            running_sums2[model_param_tuple] = 0.0

            for pix_index, pix_synopsis in map_pixel_dict_new.items():
                pix_max = 0.0
                probs = []
                for band in pix_synopsis.measured_bands:
                    for mjd, integrated_prob in pix_synopsis.best_integrated_probs[model_param_tuple][band].items():
                        probs.append(integrated_prob)

                pix_max = np.max(probs)
                running_sums2[model_param_tuple] += pix_max

        cols = ['E', 'n', 'theta_obs', 'Prob']
        dtype = ['f8', 'f8', 'f8', 'f8']

        # ########################################
        # # Hack to test linear models. Remove. DC
        # cols = ['M', 'dM', 'Prob']
        # dtype = ['f8', 'f8', 'f8']
        # ########################################
        result_table = Table(dtype=dtype, names=cols)

        for model_param_tuple, prob in running_sums2.items():

            # print("\nCombined Integrated prob to detect model (E=%s, n=%s, theta_obs=%s)" % model_param_tuple)
            # print("\t%s" % prob)
            result_table.add_row([model_param_tuple[0], model_param_tuple[1], model_param_tuple[2], prob])

            # ########################################
            # # Hack to test linear models. Remove. DC
            # print("\nCombined Integrated prob to detect model (M=%s, dM=%s)" % model_param_tuple)
            # print("\t%s" % prob)
            # result_table.add_row([model_param_tuple[0], model_param_tuple[1], prob])
            # ########################################


        result_table.write("%s/Detection_Results_%s_%s.prob" % (formatted_model_output_dir,
                                                                self.options.sub_dir,
                                                                self.options.batch_dir),
                           overwrite=True, format='ascii.ecsv')







        # endregion

if __name__ == "__main__":
    useagestring = """python ComputeModelDetection.py [options]

Example with healpix_dir defaulted to 'Events/<gwid>' amd model_dir to 'Events/{GWID}/Models':
python ComputeModelDetection.py --gw_id <gwid> --healpix_file <filename>

Assumes .dat files with Astropy ECSV format: 
https://docs.astropy.org/en/stable/api/astropy.io.ascii.Ecsv.html

"""
    start = time.time()

    teglon = Teglon()
    parser = teglon.add_options(usage=useagestring)
    options, args = parser.parse_args()
    teglon.options = options

    teglon.main()

    end = time.time()
    duration = (end - start)
    print("\n********* start DEBUG ***********")
    print("Teglon `ComputeModelDetection_GRB` execution time: %s" % duration)
    print("********* end DEBUG ***********\n")
