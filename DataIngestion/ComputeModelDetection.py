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


# Set up dustmaps config
config["data_dir"] = "./"

# Generate all pixel indices
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
GW190814_t_0 = 58709.882824224536  # time of GW190814 merger


def get_healpix_pixel_id(galaxy_info):
    phi, theta = np.radians(float(galaxy_info[8])), 0.5 * np.pi - np.radians(float(galaxy_info[9]))

    # map NSIDE is last argument of galaxy_info
    # return the galaxy_id with the pixel index in the NSIDE of the healpix map
    return (galaxy_info[0], hp.ang2pix(int(galaxy_info[-1]), theta, phi))


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

        return (parser)

    def main(self):

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
        model_path = "../Models"

        model_files = []
        for file in os.listdir(model_path):
            if file.endswith(".dat"):
                model_files.append("%s/%s" % (model_path, file))

        if len(model_files) <= 0:
            is_error = True
            print("There are no models to process.")

        # Check if the above files exist...
        if not os.path.exists(hpx_path):
            is_error = True
            print("Healpix file `%s` does not exist." % hpx_path)

        if is_error:
            print("Exiting...")
            return 1

        # CONVENIENCE DICTIONARIES
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
        for mf in model_files:
            model_table = Table.read(mf, format='ascii.ecsv')
            mask = model_table['time'] > 0.0

            time = np.asarray(model_table['time'][mask])
            g = np.asarray(model_table['sdss_g'][mask])
            r = np.asarray(model_table['sdss_r'][mask])
            i = np.asarray(model_table['sdss_i'][mask])
            clear = np.asarray(model_table['Clear'][mask])

            model_props = model_table.meta['comment']
            mass = float(re.findall("\d+\.\d+", model_props[1])[0])
            velocity = float(re.findall("\d+\.\d+", model_props[2])[0])

            factor = float(re.findall("(\d+\.?\d?)[e+\-]", model_props[3])[0])
            exp = float(re.findall("[e+\-](-+\d+\.?\d+)", model_props[3])[0])
            Xlan = factor * 10 ** (exp)

            base_name = os.path.basename(mf)
            print("Loading `%s`" % base_name)
            models[(base_name, mass, velocity, Xlan)] = {
                'time': time,
                'sdss_g': g,
                'sdss_r': r,
                'sdss_i': i,
                'Clear': clear
            }

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

        # map_pixel_dict_old = OrderedDict()
        class Pixel_Synopsis():
            def __init__(self, mean_dist, dist_sigma, prob_2D, pixel_index, N128_index, pix_ebv, z):
                self.mean_dist = mean_dist
                self.dist_sigma = dist_sigma
                self.prob_2D = prob_2D
                self.pixel_index = pixel_index
                self.N128_index = N128_index
                self.pix_ebv = pix_ebv
                self.z = z

                self.distance_arr = None
                self.distance_modulus_arr = None
                self.distance_probs = None

                # From the tiles that contain this pixel
                # band:value
                self.measured_bands = []
                self.lim_mags = OrderedDict()
                self.delta_mjds = OrderedDict()

                # From the model (only select the bands that have been imaged)
                self.A_lambda = OrderedDict()  # band:value

                # model:arr
                self.model_observer_time_arr_new = OrderedDict()
                # self.model_observer_time_arr_old = None

                # Computed based on model + tile + pixel info
                # band:value
                # self.app_mag_model = {} # x-val
                # self.app_mag_model_prob = {} # y-val

                # Final calculation
                # model:band:value
                self.best_integrated_probs = OrderedDict()

            def __str__(self):
                return str(self.__dict__)

        count_bad_pixels = 0
        for p in map_pixels:
            mean_dist = float(p[7])
            dist_sigma = float(p[8])
            prob_2D = float(p[3])
            pixel_index = int(p[2])
            N128_pixel_index = int(p[10])
            pix_ebv = ebv[N128_pixel_index]

            d = Distance(mean_dist, u.Mpc)
            if mean_dist == 0.0:
                # distance did not converge for this pixel. pass...
                print("Bad Index: %s" % pixel_index)
                count_bad_pixels += 1
                continue

            z = d.compute_z(cosmology=cosmo)

            p_new = Pixel_Synopsis(
                mean_dist,
                dist_sigma,
                prob_2D,
                pixel_index,
                N128_pixel_index,
                pix_ebv,
                z)

            min_dist = mean_dist - 5.0 * dist_sigma
            if min_dist <= 0.0:
                min_dist = 0.001
            max_dist = mean_dist + 5.0 * dist_sigma

            distance_arr = np.linspace(min_dist, max_dist, 100)
            distance_modulus_arr = 5.0 * np.log10(distance_arr * 1e+6) - 5.0
            distance_probs = 1.0 / np.sqrt(2.0 * np.pi * dist_sigma ** 2) * np.exp(
                -1.0 * (distance_arr - mean_dist) ** 2 / (2 * dist_sigma ** 2))

            p_new.distance_arr = distance_arr
            p_new.distance_modulus_arr = distance_modulus_arr
            p_new.distance_probs = distance_probs

            map_pixel_dict_new[pixel_index] = p_new

        print("\nMap pixel dict complete. %s bad pixels." % count_bad_pixels)

        # Get Detectors
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
        # For each tile:
        # 	we want the MJD of observation, and add that to the list of a pixels' MJD collection.
        # 	we want the limiting mag, add that to the list of a pixel's lim mag collection
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

                pix_synopsis_new.delta_mjds[band_name][t.mjd] = (t.mjd - GW190814_t_0)
                pix_synopsis_new.lim_mags[band_name][t.mjd] = (t.mag_lim)

        print("\nInitializing %s models..." % len(models))

        print("\nUpdating pixel `A_lambda` and `model_observer_time_arr`...")
        # Set pixel `model_observer_time_arr` and `A_lambda`
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

                time_dilation = 1.0 + pix_synopsis.z
                pix_synopsis.model_observer_time_arr_new[model_param_tuple] = model_dict["time"] * time_dilation

        # NEW Do the calculation...
        print("\nIntegrating total model probs...")
        count = 0
        for pix_index, pix_synopsis in map_pixel_dict_new.items():

            for band in pix_synopsis.measured_bands:

                for model_param_tuple, model_dict in models.items():

                    model_abs_mags = model_dict[reverse_band_mapping_new[band]]
                    pixel_delta_mjd = pix_synopsis.delta_mjds[band]

                    for i, (mjd, delta_mjd) in enumerate(pixel_delta_mjd.items()):

                        # get the corresponding abs mag for the time of the observation
                        pix_abs_mag = np.interp(delta_mjd, pix_synopsis.model_observer_time_arr_new[model_param_tuple],
                                                model_abs_mags)
                        # print("pix_abs_mag: %s" % pix_abs_mag)

                        # compute the distribution in apparent mag
                        pix_app_mag = np.asarray(pix_synopsis.distance_modulus_arr) + pix_abs_mag + \
                                      pix_synopsis.A_lambda[band]

                        # re-normalize this distribution to sum to the pixel 2D prob
                        # SIMPS (y, x)
                        app_mag_norm = simps(pix_synopsis.distance_probs, pix_app_mag)
                        renorm_pix_app_mag_prob = np.asarray(
                            (pix_synopsis.prob_2D / app_mag_norm) * pix_synopsis.distance_probs)

                        # Integrate the app mag distribution from arbitrarily bright to the limiting magnitude
                        f_interp = lambda x: np.interp(x, pix_app_mag, renorm_pix_app_mag_prob)
                        lower_bound = np.min(pix_app_mag)
                        upper_bound = pix_synopsis.lim_mags[band][mjd]

                        if model_param_tuple not in pix_synopsis.best_integrated_probs:
                            pix_synopsis.best_integrated_probs[model_param_tuple] = {}

                        try:
                            pix_synopsis.best_integrated_probs[model_param_tuple][band][mjd] = 0.0
                        except:
                            pix_synopsis.best_integrated_probs[model_param_tuple][band] = {mjd: 0.0}

                        prob_to_detect = quad(f_interp, lower_bound, upper_bound)[0]

                        if not np.isnan(prob_to_detect) and prob_to_detect >= 0.0:
                            pix_synopsis.best_integrated_probs[model_param_tuple][band][mjd] = prob_to_detect

            count += 1
            if count % 1000 == 0:
                print("Processed: %s" % count)

        # NEW
        # Finally, get the highest valued integration, and sum
        running_sums = {}  # model:band:value
        # pixels_to_plot = {} # model:band:pixel

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

        # pixels_to_plot[] append(Pixel_Element(pix_index, healpix_map_nside, pix_max))

        for model_param_tuple, band_dict in running_sums.items():
            print("\nIntegrated prob to detect model by band `%s` (%s, %s, %s)" % model_param_tuple)

            for band, running_sum in band_dict.items():
                print("\t%s: %s" % (band, running_sum))

        ## Additional calculation -- for every pixel, just get the highest prob
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

        # Build ascii.ecsv formatted output
        cols = ['Model', 'Mass', 'Velocity', 'X_lan', 'Prob']
        dtype = ['S40', 'f8', 'f8', 'f8', 'f8']
        meta = ['{key} = {value}'.format(key=db[0], value=db[1]) for db in detector_band_result[0]]
        result_table = Table(dtype=dtype, names=cols)
        result_table.meta['comment'] = meta

        for model_param_tuple, prob in running_sums2.items():
            print("\nCombined Integrated prob to detect model `%s` (%s, %s, %s)" % model_param_tuple)
            print("\t%s" % prob)
            result_table.add_row(
                [model_param_tuple[0], model_param_tuple[1], model_param_tuple[2], model_param_tuple[3], prob])

        result_table.write("%s/Detection_Results.prob" % formatted_model_output_dir, overwrite=True, format='ascii.ecsv')

    # # Just for fun: Pixel prob map
    # fig = plt.figure(figsize=(10,10), dpi=1000)
    # ax = fig.add_subplot(111)

    # m = Basemap(projection='stere',
    # 		 lon_0=15.0,
    # 		 lat_0=-20.0,
    # 		 llcrnrlat=-35.0,
    # 		 urcrnrlat=-18.0,
    # 		 llcrnrlon=5.0,
    # 		 urcrnrlon=30.0)

    # log_lower_limit = 1e-8
    # pixel_probs = []
    # for p in pixels_to_plot:
    # 	if p.prob >= log_lower_limit:
    # 		pixel_probs.append(p.prob)
    # 	else:
    # 		pixel_probs.append(log_lower_limit)

    # # pixel_probs = [p.prob for p in pixels_to_plot]
    # min_prob = np.min(pixel_probs)
    # max_prob = np.max(pixel_probs)

    # print("min prob: %s" % min_prob)
    # print("max prob: %s" % max_prob)

    # # norm = colors.Normalize(min_prob, max_prob)
    # norm = colors.LogNorm(min_prob, max_prob)
    # print("Plotting (%s) `pixels`..." % len(pixels_to_plot))
    # for i,p in enumerate(pixels_to_plot):
    # 	p.plot(m, ax, facecolor=plt.cm.viridis(norm(pixel_probs[i])), edgecolor='None', linewidth=0.5, alpha=0.8)

    # for i,p in enumerate(pixels_to_plot):
    # 	p.plot(m, ax, facecolor='None', edgecolor='k', linewidth=0.25, alpha=1.0)

    # # draw meridians
    # meridians = np.arange(0.,360.,10.)
    # par = m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=18,zorder=-1,color='gray', linewidth=0.5)

    # # draw parallels
    # parallels = np.arange(-90.,90.,10.)
    # par = m.drawparallels(parallels,labels=[0,1,0,0],fontsize=18,zorder=-1,color='gray', linewidth=0.5, xoffset=230000)

    # sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    # sm.set_array([]) # can be an empty list

    # # tks = np.linspace(min_prob, max_prob, 11)
    # tks = np.logspace(np.log10(min_prob), np.log10(max_prob), 11)
    # tks_strings = []

    # for t in tks:
    # 	tks_strings.append('%0.3f' % (t*100))

    # cb = fig.colorbar(sm, ax=ax, ticks=tks, orientation='horizontal', fraction=0.08951, pad=0.05, alpha=0.80)
    # cb.ax.set_xticklabels(tks_strings, fontsize=14)
    # cb.set_label("% per Pixel", fontsize=14, labelpad=10.0)
    # cb.outline.set_linewidth(1.0)

    # for axis in ['top','bottom','left','right']:
    # 	ax.spines[axis].set_linewidth(2.0)

    # ax.invert_xaxis()

    # fig.savefig('CharlieTest2.png', bbox_inches='tight')
    # plt.close('all')
    # print("... Done.")

    # Plot r-band light curve with limits by telescope
    # 1. Use the avg distance from pixels
    # 	1.a Use this distance to perform time dilation
    # 2. Use the avg A_r from pixels

        # parser.add_option('--gw_id', default="", type="str", help='LIGO superevent name, e.g. `S190425z`.')
        #
        # parser.add_option('--healpix_file', default="", type="str",
        #                   help='Healpix filename. Used with `gw_id` to identify unique map.')
        #
        # parser.add_option('--healpix_dir', default='../Events/{GWID}', type="str",
        #                   help='Directory for where to look for the healpix file.')
        #
        # parser.add_option('--model_dir', default="../Events/{GWID}/Models", type="str",
        #                   help='Directory for where to look for the models to be processed.')


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
    print("Teglon `ComputeModelDetection` execution time: %s" % duration)
    print("********* end DEBUG ***********\n")
