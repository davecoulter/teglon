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

import sys

sys.path.append('../')

import os
import optparse

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

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
configFile = os.path.join(__location__, 'Settings.ini')

db_config = RawConfigParser()
db_config.read(configFile)

db_name = db_config.get('database', 'DATABASE_NAME')
db_user = db_config.get('database', 'DATABASE_USER')
db_pwd = db_config.get('database', 'DATABASE_PASSWORD')
db_host = db_config.get('database', 'DATABASE_HOST')
db_port = db_config.get('database', 'DATABASE_PORT')


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


class Teglon:

    def add_options(self, parser=None, usage=None, config=None):
        import optparse
        if parser == None:
            parser = optparse.OptionParser(usage=usage, conflict_handler="resolve")

        parser.add_option('--gw_id', default="", type="str",
                          help='LIGO superevent name, e.g. `S190425z` ')

        parser.add_option('--healpix_dir', default='../Events/{GWID}', type="str",
                          help='Directory for where to look for the healpix file.')

        parser.add_option('--healpix_file', default="", type="str", help='healpix filename.')

        parser.add_option('--s_band', default="r", type="str",
                          help='Filter for Swope tile extract to use for extinction calculation. Default: `r`. Available: (g, r, i)')

        parser.add_option('--t_band', default="r", type="str",
                          help='Filter for Thacher tile extract to use for extinction calculation. Default: `r`. Available: (g, r, i)')

        parser.add_option('--g_band', default="r", type="str",
                          help='Filter for Galaxies extract to use for extinction calculation. Default: `r`. Available: (g, r, i)')

        parser.add_option('--s_exp_time', default="60.0", type="float",
                          help='Exposure time (seconds) to write out for Swope. Default: `60.0`. Must be > 0.0')

        parser.add_option('--t_exp_time', default="120.0", type="float",
                          help='Exposure time (seconds) to write out for Thacher. Default: `120.0`. Must be > 0.0')

        parser.add_option('--g_exp_time', default="120.0", type="float",
                          help='Exposure time (seconds) to write out for Galaxies. Default: `120.0`. Must be > 0.0')

        parser.add_option('--extinct', default="0.5", type="float",
                          help='Extinction in mags in the specified band(s) to be less than. Default: 0.5 mag. Must be > 0.0')

        parser.add_option('--prob_type', default="4D", type="str",
                          help='''Probability type to consider. Default `4D`. Available: (4D, 2D)''')

        parser.add_option('--s_cum_prob', default="0.9", type="float",
                          help='Cumulative prob to cover in tiles for Swope. Default 0.9. Must be > 0.2 and < 0.95')

        parser.add_option('--t_cum_prob', default="0.9", type="float",
                          help='Cumulative prob to cover in tiles for Thacher. Default 0.9. Must be > 0.2 and < 0.95')

        parser.add_option('--num_gal', default="10000", type="int",
                          help='''Filters number of galaxies to return. All galaxies are ranked wrt 4D prob. The top 
                          `num_gal` galaxies are then returned. Specify `-1` to not filter results. Default: `10000` 
                          ''')

        parser.add_option('--galaxies_detector', default="NICKEL", type="string",
                          help='The telescope to default in the galaxies output')

        parser.add_option('--s_min_ra', default="-1.0", type="float",
                      help='''Swope: Optional, decimal format. If specified, `s_min_ra`, `s_max_ra`, `s_min_dec`, 
                      and `s_max_dec` are all required and supersede Swope dec limits. Tiles within this range 
                      will be selected <= to prob limit specified in `s_cum_prob`. `s_min_ra` must be >= 0.0 and < 
                      `s_max_ra`
                      ''')

        parser.add_option('--s_max_ra', default="-1.0", type="float",
                          help='''Swope: Optional, decimal format. If specified, `s_min_ra`, `s_max_ra`, `s_min_dec`, 
                          and `s_max_dec` are all required and supersede Swope dev limits. Tiles within this range will 
                          be selected <= to prob limit specified in `s_cum_prob`. `s_max_ra` must be <=360.0 and > 
                          `s_min_ra`
                          ''')

        parser.add_option('--s_min_dec', default="-1.0", type="float",
                          help='''Swope: Optional, decimal format. If specified, `s_min_ra`, `s_max_ra`, `s_min_dec`, 
                          and `s_max_dec` are all required and supersede Swope dec limits. Tiles within this range will 
                          be selected <= to prob limit specified in `s_cum_prob`. `s_min_dec` must be >= -90.0 and < 
                          `s_max_dec`
                          ''')

        parser.add_option('--s_max_dec', default="-1.0", type="float",
                          help='''Swope: Optional, decimal format. If specified, `s_min_ra`, `s_max_ra`, `s_min_dec`, 
                          and `s_max_dec` are all required and supersede Swope dec limits. Tiles within this range will 
                          be selected <= to prob limit specified in `s_cum_prob`. `s_max_dec` must be <= 90.0 and > 
                          `s_min_dec`
                          ''')

        parser.add_option('--t_min_ra', default="-1.0", type="float",
                          help='''Thacher: Optional, decimal format. If specified, `t_min_ra`, `t_max_ra`, `t_min_dec`, 
                          and `t_max_dec` are all required and supersede Thacher dec limits. Tiles within this range 
                          will be selected <= to prob limit specified in `t_cum_prob`. `t_min_ra` must be >= 0.0 and < 
                          `t_max_ra`
                          ''')

        parser.add_option('--t_max_ra', default="-1.0", type="float",
                          help='''Thacher: Optional, decimal format. If specified, `t_min_ra`, `t_max_ra`, `t_min_dec`, 
                          and `t_max_dec` are all required and supersede Thacher dec limits. Tiles within this range 
                          will be selected <= to prob limit specified in `t_cum_prob`. `t_max_ra` must be <=360.0 and > 
                          `t_min_ra`
                          ''')

        parser.add_option('--t_min_dec', default="-1.0", type="float",
                          help='''Thacher: Optional, decimal format. If specified, `t_min_ra`, `t_max_ra`, `t_min_dec`, 
                          and `t_max_dec` are all required and supersede Thacher dec limits. Tiles within this range 
                          will be selected <= to prob limit specified in `t_cum_prob`. `t_min_dec` must be >= -90.0 and 
                          < `t_max_dec`
                          ''')

        parser.add_option('--t_max_dec', default="-1.0", type="float",
                          help='''Thacher: Optional, decimal format. If specified, `t_min_ra`, `t_max_ra`, `t_min_dec`, 
                          and `t_max_dec` are all required and supersede Thacher dec limits. Tiles within this range 
                          will be selected <= to prob limit specified in `t_cum_prob`. `t_max_dec` must be <= 90.0 and > 
                          `t_min_dec`
                          ''')

        parser.add_option('--g_min_ra', default="-1.0", type="float",
                          help='''Galaxies: Optional, decimal format. If specified, `g_min_ra`, `g_max_ra`, `g_min_dec`, 
                          and `g_max_dec` are all required and filter the top `num_gal` galaxies. `g_min_ra` must be >= 
                          0.0 and < `g_max_ra`
                          ''')

        parser.add_option('--g_max_ra', default="-1.0", type="float",
                          help='''Galaxies: Optional, decimal format. If specified, `g_min_ra`, `g_max_ra`, `g_min_dec`, 
                                  and `g_max_dec` are all required and filter the top `num_gal` galaxies. `g_max_ra` 
                                  must be <= 360.0 and > `g_min_ra`
                                  ''')

        parser.add_option('--g_min_dec', default="-1.0", type="float",
                          help='''Galaxies: Optional, decimal format. If specified, `g_min_ra`, `g_max_ra`, `g_min_dec`, 
                                  and `g_max_dec` are all required and filter the top `num_gal` galaxies. `g_min_dec` 
                                  must be >= -90.0 and < `g_max_dec`
                                  ''')

        parser.add_option('--g_max_dec', default="-1.0", type="float",
                          help='''Galaxies: Optional, decimal format. If specified, `g_min_ra`, `g_max_ra`, `g_min_dec`, 
                                  and `g_max_dec` are all required and filter the top `num_gal` galaxies. `g_max_dec` 
                                  must be <= 90.0 and > `g_min_dec`
                                  ''')

        return (parser)

    def main(self):

        t1 = time.time()

        # Valid prob types
        _4D = "4D"
        _2D = "2D"

        band_mapping = {
            "g": "SDSS g",
            "r": "SDSS r",
            "i": "SDSS i"
        }

        is_error = False

        # Parameter sanity checks
        if self.options.gw_id == "":
            is_error = True
            print("GWID is required.")

        formatted_healpix_dir = self.options.healpix_dir
        if "{GWID}" in formatted_healpix_dir:
            formatted_healpix_dir = formatted_healpix_dir.replace("{GWID}", self.options.gw_id)

        if self.options.healpix_file == "":
            is_error = True
            print("You must specify which healpix file to process.")

        if self.options.s_band not in band_mapping:
            is_error = True
            print("Invalid Swope band selection. Available bands: %s" % band_mapping.keys())

        if self.options.t_band not in band_mapping:
            is_error = True
            print("Invalid Thacher band selection. Available bands: %s" % band_mapping.keys())

        if self.options.g_band not in band_mapping:
            is_error = True
            print("Invalid Galaxy band selection. Available bands: %s" % band_mapping.keys())

        if self.options.s_exp_time <= 0.0:
            is_error = True
            print("Swope exposure time must be > 0.0 seconds")

        if self.options.t_exp_time <= 0.0:
            is_error = True
            print("Thacher exposure time must be > 0.0 seconds")

        if self.options.g_exp_time <= 0.0:
            is_error = True
            print("Galaxy exposure time must be > 0.0 seconds")

        if self.options.extinct <= 0.0:
            is_error = True
            print("Extinction must be a valid float > 0.0")

        if not (self.options.prob_type == _4D or self.options.prob_type == _2D):
            is_error = True
            print("Prob type must either be `4D` or `2D`")

        if self.options.s_cum_prob > 0.95 or self.options.s_cum_prob < 0.20:
            is_error = True
            print("Swope cumulative prob must be between 0.2 and 0.95")

        if self.options.t_cum_prob > 0.95 or self.options.t_cum_prob < 0.20:
            is_error = True
            print("Thacher cumulative prob must be between 0.2 and 0.95")

        is_Swope_box_query = False
        if self.options.s_min_ra != -1 and \
                self.options.s_max_ra != -1 and \
                self.options.s_min_dec != -1 and \
                self.options.s_max_dec != -1:
            # if any of these are non-default values, then we will validate them all
            is_Swope_box_query = True

            if self.options.s_min_ra < 0.0 or self.options.s_min_ra >= self.options.s_max_ra:
                is_error = True
                print("Swope Min RA must be >= 0.0 and be < Swope Max RA")

            if self.options.s_max_ra > 360.0 or self.options.s_max_ra <= self.options.s_min_ra:
                is_error = True
                print("Swope Max RA must be <= 360.0 and be > Swope Min RA")

            if self.options.s_min_dec < -90.0 or self.options.s_min_dec >= self.options.s_max_dec:
                is_error = True
                print("Swope Min Dec must be >= -90.0 and be < Swope Max Dec")

            if self.options.s_max_dec > 90.0 or self.options.s_max_dec <= self.options.s_min_dec:
                is_error = True
                print("Swope Max Dec must be <= 90.0 and be > Swope Min Dec")

        is_Thacher_box_query = False
        if self.options.t_min_ra != -1 and \
                self.options.t_max_ra != -1 and \
                self.options.t_min_dec != -1 and \
                self.options.t_max_dec != -1:
            # if any of these are non-default values, then we will validate them all
            is_Thacher_box_query = True

            if self.options.t_min_ra < 0.0 or self.options.t_min_ra >= self.options.t_max_ra:
                is_error = True
                print("Thacher Min RA must be >= 0.0 and be < Thacher Max RA")

            if self.options.t_max_ra > 360.0 or self.options.t_max_ra <= self.options.t_min_ra:
                is_error = True
                print("Thacher Max RA must be <= 360.0 and be > Thacher Min RA")

            if self.options.t_min_dec < -90.0 or self.options.t_min_dec >= self.options.t_max_dec:
                is_error = True
                print("Thacher Min Dec must be >= -90.0 and be < Thacher Max Dec")

            if self.options.t_max_dec > 90.0 or self.options.t_max_dec <= self.options.t_min_dec:
                is_error = True
                print("Thacher Max Dec must be <= 90.0 and be > Thacher Min Dec")

        is_Galaxy_box_query = False
        if self.options.g_min_ra != -1 and \
                self.options.g_max_ra != -1 and \
                self.options.g_min_dec != -1 and \
                self.options.g_max_dec != -1:
            # if any of these are non-default values, then we will validate them all
            is_Galaxy_box_query = True

            if self.options.g_min_ra < 0.0 or self.options.g_min_ra >= self.options.g_max_ra:
                is_error = True
                print("Galaxy Min RA must be >= 0.0 and be < Galaxy Max RA")

            if self.options.g_max_ra > 360.0 or self.options.g_max_ra <= self.options.g_min_ra:
                is_error = True
                print("Galaxy Max RA must be <= 360.0 and be > Galaxy Min RA")

            if self.options.g_min_dec < -90.0 or self.options.g_min_dec >= self.options.g_max_dec:
                is_error = True
                print("Galaxy Min Dec must be >= -90.0 and be < Galaxy Max Dec")

            if self.options.g_max_dec > 90.0 or self.options.g_max_dec <= self.options.g_min_dec:
                is_error = True
                print("Galaxy Max Dec must be <= 90.0 and be > Galaxy Min Dec")

        if is_error:
            print("Exiting...")
            return 1

        hpx_path = "%s/%s" % (formatted_healpix_dir, self.options.healpix_file)
        s_band_name = band_mapping[self.options.s_band]
        t_band_name = band_mapping[self.options.t_band]
        g_band_name = band_mapping[self.options.g_band]

        # Get Map ID
        healpix_map_select = "SELECT id, RescaledNSIDE FROM HealpixMap WHERE GWID = '%s' and Filename = '%s'"
        healpix_map_id = int(query_db([healpix_map_select % (self.options.gw_id, self.options.healpix_file)])[0][0][0])

        band_select = "SELECT id, Name, F99_Coefficient FROM Band WHERE `Name`='%s'"
        s_band_result = query_db([band_select % s_band_name])[0][0]
        s_band_F99 = float(s_band_result[2])

        t_band_result = query_db([band_select % t_band_name])[0][0]
        t_band_F99 = float(t_band_result[2])

        g_band_result = query_db([band_select % g_band_name])[0][0]
        g_band_F99 = float(g_band_result[2])

        SWOPE = "SWOPE"
        THACHER = "THACHER"
        GALAXY = self.options.galaxies_detector

        detector_select_by_name = "SELECT id, Name, Deg_width, Deg_height, Deg_radius, Area, MinDec, MaxDec FROM Detector WHERE Name='%s'"
        s_detector_result = query_db([detector_select_by_name % SWOPE])[0][0]
        s_detector_id = int(s_detector_result[0])
        t_detector_result = query_db([detector_select_by_name % THACHER])[0][0]
        t_detector_id = int(t_detector_result[0])

        # 4D NON-BOX QUERY
        # REPLACEMENT PARAMETERS: (band_F99, detector_id, healpix_map_id, self.options.cum_prob, band_F99,
        # self.options.extinct)
        tile_select_4D = '''
        SELECT 
            running_tile_prob.id as static_tile_id,  
            st.FieldName, 
            st.RA, 
            st._Dec, 
            running_tile_prob.net_prob, 
            running_tile_prob.mean_pixel_dist, 
            running_tile_prob.EBV*%s as A_lambda, 
            running_tile_prob.cum_prob 
        FROM 
            (SELECT 
                aggregate_tile.id, 
                aggregate_tile.Detector_id, 
                aggregate_tile.net_prob, 
                aggregate_tile.mean_pixel_dist, 
                aggregate_tile.EBV, 
                SUM(aggregate_tile.net_prob) OVER(ORDER BY SUM(aggregate_tile.net_prob) DESC) AS cum_prob 
             FROM 
                (SELECT 
                    st.id, 
                    st.Detector_id, 
                    SUM(hpc.NetPixelProb) as net_prob, 
                    AVG(hp.Mean) as mean_pixel_dist, 
                    sp_ebv.EBV 
                FROM 
                    StaticTile st 
                JOIN StaticTile_HealpixPixel st_hp on st_hp.StaticTile_id = st.id 
                JOIN HealpixPixel_Completeness hpc on hpc.HealpixPixel_id = st_hp.HealpixPixel_id 
                JOIN HealpixPixel hp on hp.id = hpc.HealpixPixel_id 
                JOIN SkyPixel_EBV sp_ebv on sp_ebv.N128_SkyPixel_id = st.N128_SkyPixel_id 
                JOIN Detector d on d.id = st.Detector_id 
                WHERE d.id = %s and hp.HealpixMap_id = %s 
                GROUP BY 
                    st.id, 
                    st.Detector_id, 
                    sp_ebv.EBV) aggregate_tile 
            GROUP BY 
                aggregate_tile.id, 
                aggregate_tile.Detector_id, 
                aggregate_tile.net_prob, 
                aggregate_tile.mean_pixel_dist, 
                aggregate_tile.EBV 
            ORDER BY 
                aggregate_tile.net_prob DESC) as running_tile_prob 
        JOIN StaticTile st on st.id = running_tile_prob.id 
        JOIN Detector d on d.id = running_tile_prob.Detector_id 
        WHERE 
            running_tile_prob.cum_prob <= %s AND 
            running_tile_prob.EBV*%s <= %s AND 
            st._Dec BETWEEN d.MinDec AND d.MaxDec 
        ORDER BY 
            running_tile_prob.net_prob DESC 
        '''

        # 2D NON-BOX QUERY
        # REPLACEMENT PARAMETERS: (band_F99, detector_id, healpix_map_id, self.options.cum_prob, band_F99,
        # self.options.extinct)
        tile_select_2D = '''
        SELECT 
            running_tile_prob.id as static_tile_id,  
            st.FieldName, 
            st.RA, 
            st._Dec, 
            running_tile_prob.net_prob, 
            running_tile_prob.mean_pixel_dist, 
            running_tile_prob.EBV*%s as A_lambda, 
            running_tile_prob.cum_prob 
        FROM 
            (SELECT 
                aggregate_tile.id, 
                aggregate_tile.Detector_id, 
                aggregate_tile.net_prob, 
                aggregate_tile.mean_pixel_dist, 
                aggregate_tile.EBV, 
                SUM(aggregate_tile.net_prob) OVER(ORDER BY SUM(aggregate_tile.net_prob) DESC) AS cum_prob 
             FROM 
                (SELECT 
                    st.id, 
                    st.Detector_id, 
                    SUM(hp.prob) as net_prob, 
                    AVG(hp.Mean) as mean_pixel_dist, 
                    sp_ebv.EBV 
                FROM 
                    StaticTile st 
                JOIN StaticTile_HealpixPixel st_hp on st_hp.StaticTile_id = st.id  
                JOIN HealpixPixel hp on hp.id = st_hp.HealpixPixel_id 
                JOIN SkyPixel_EBV sp_ebv on sp_ebv.N128_SkyPixel_id = st.N128_SkyPixel_id 
                JOIN Detector d on d.id = st.Detector_id 
                WHERE d.id = %s and hp.HealpixMap_id = %s 
                GROUP BY 
                    st.id, 
                    st.Detector_id, 
                    sp_ebv.EBV) aggregate_tile 
            GROUP BY 
                aggregate_tile.id, 
                aggregate_tile.Detector_id, 
                aggregate_tile.net_prob, 
                aggregate_tile.mean_pixel_dist, 
                aggregate_tile.EBV 
            ORDER BY 
                aggregate_tile.net_prob DESC) as running_tile_prob 
        JOIN StaticTile st on st.id = running_tile_prob.id 
        JOIN Detector d on d.id = running_tile_prob.Detector_id 
        WHERE 
            running_tile_prob.cum_prob <= %s AND 
            running_tile_prob.EBV*%s <= %s AND 
            st._Dec BETWEEN d.MinDec AND d.MaxDec 
        ORDER BY 
            running_tile_prob.net_prob DESC 
    '''

        # 4D BOX QUERY
        # REPLACEMENT PARAMETERS: (band_F99, detector_id, healpix_map_id, self.options.cum_prob, band_F99,
        # self.options.extinct, s/t_min_ra, s/t_max_ra, s/t_min_dec, s/t_max_dec)
        box_tile_select_4D = '''
        SELECT 
            running_tile_prob.id as static_tile_id,  
            st.FieldName, 
            st.RA, 
            st._Dec, 
            running_tile_prob.net_prob, 
            running_tile_prob.mean_pixel_dist, 
            running_tile_prob.EBV*%s as A_lambda, 
            running_tile_prob.cum_prob 
        FROM 
            (SELECT 
                aggregate_tile.id, 
                aggregate_tile.Detector_id, 
                aggregate_tile.net_prob, 
                aggregate_tile.mean_pixel_dist, 
                aggregate_tile.EBV, 
                SUM(aggregate_tile.net_prob) OVER(ORDER BY SUM(aggregate_tile.net_prob) DESC) AS cum_prob 
             FROM 
                (SELECT 
                    st.id, 
                    st.Detector_id, 
                    SUM(hpc.NetPixelProb) as net_prob, 
                    AVG(hp.Mean) as mean_pixel_dist, 
                    sp_ebv.EBV 
                FROM 
                    StaticTile st 
                JOIN StaticTile_HealpixPixel st_hp on st_hp.StaticTile_id = st.id 
                JOIN HealpixPixel_Completeness hpc on hpc.HealpixPixel_id = st_hp.HealpixPixel_id 
                JOIN HealpixPixel hp on hp.id = hpc.HealpixPixel_id 
                JOIN SkyPixel_EBV sp_ebv on sp_ebv.N128_SkyPixel_id = st.N128_SkyPixel_id 
                JOIN Detector d on d.id = st.Detector_id 
                WHERE d.id = %s and hp.HealpixMap_id = %s 
                GROUP BY 
                    st.id, 
                    st.Detector_id, 
                    sp_ebv.EBV) aggregate_tile 
            GROUP BY 
                aggregate_tile.id, 
                aggregate_tile.Detector_id, 
                aggregate_tile.net_prob, 
                aggregate_tile.mean_pixel_dist, 
                aggregate_tile.EBV 
            ORDER BY 
                aggregate_tile.net_prob DESC) as running_tile_prob 
        JOIN StaticTile st on st.id = running_tile_prob.id 
        JOIN Detector d on d.id = running_tile_prob.Detector_id 
        WHERE 
            running_tile_prob.cum_prob <= %s AND 
            running_tile_prob.EBV*%s <= %s AND  
            st.RA BETWEEN %s AND %s AND 
            st._Dec BETWEEN %s AND %s 
        ORDER BY 
            running_tile_prob.net_prob DESC 
        '''

        # 2D BOX QUERY
        # REPLACEMENT PARAMETERS: (band_F99, detector_id, healpix_map_id, self.options.cum_prob, band_F99,
        # self.options.extinct, s/t_min_ra, s/t_max_ra, s/t_min_dec, s/t_max_dec)
        box_tile_select_2D = '''
        SELECT 
            running_tile_prob.id as static_tile_id,  
            st.FieldName, 
            st.RA, 
            st._Dec, 
            running_tile_prob.net_prob, 
            running_tile_prob.mean_pixel_dist, 
            running_tile_prob.EBV*%s as A_lambda, 
            running_tile_prob.cum_prob 
        FROM 
            (SELECT 
                aggregate_tile.id, 
                aggregate_tile.Detector_id, 
                aggregate_tile.net_prob, 
                aggregate_tile.mean_pixel_dist, 
                aggregate_tile.EBV, 
                SUM(aggregate_tile.net_prob) OVER(ORDER BY SUM(aggregate_tile.net_prob) DESC) AS cum_prob 
             FROM 
                (SELECT 
                    st.id, 
                    st.Detector_id, 
                    SUM(hp.Prob) as net_prob, 
                    AVG(hp.Mean) as mean_pixel_dist, 
                    sp_ebv.EBV 
                FROM 
                    StaticTile st 
                JOIN StaticTile_HealpixPixel st_hp on st_hp.StaticTile_id = st.id  
                JOIN HealpixPixel hp on hp.id = st_hp.HealpixPixel_id 
                JOIN SkyPixel_EBV sp_ebv on sp_ebv.N128_SkyPixel_id = st.N128_SkyPixel_id 
                JOIN Detector d on d.id = st.Detector_id 
                WHERE d.id = %s and hp.HealpixMap_id = %s 
                GROUP BY 
                    st.id, 
                    st.Detector_id, 
                    sp_ebv.EBV) aggregate_tile 
            GROUP BY 
                aggregate_tile.id, 
                aggregate_tile.Detector_id, 
                aggregate_tile.net_prob, 
                aggregate_tile.mean_pixel_dist, 
                aggregate_tile.EBV 
            ORDER BY 
                aggregate_tile.net_prob DESC) as running_tile_prob 
        JOIN StaticTile st on st.id = running_tile_prob.id 
        JOIN Detector d on d.id = running_tile_prob.Detector_id 
        WHERE 
            running_tile_prob.cum_prob <= %s AND 
            running_tile_prob.EBV*%s <= %s AND  
            st.RA BETWEEN %s AND %s AND 
            st._Dec BETWEEN %s AND %s 
        ORDER BY 
            running_tile_prob.net_prob DESC 
        '''

        galaxies_select = '''
        SELECT 
            gd2.id, 
            gd2.Name_GWGC, 
            gd2.Name_HyperLEDA, 
            gd2.Name_2MASS, 
            gd2.RA, 
            gd2._Dec, 
            gd2.z_dist, 
            gd2.B, 
            gd2.K, 
            hp_gd2_w.GalaxyProb 
        FROM 
            GalaxyDistance2 gd2 
        JOIN HealpixPixel_GalaxyDistance2 hp_gd2 on hp_gd2.GalaxyDistance2_id = gd2.id 
        JOIN HealpixPixel_GalaxyDistance2_Weight hp_gd2_w on hp_gd2_w.HealpixPixel_GalaxyDistance2_id = hp_gd2.id 
        JOIN HealpixPixel_Completeness hpc on hpc.HealpixPixel_id = hp_gd2.HealpixPixel_id 
        WHERE 
            hpc.HealpixMap_id = %s 
        '''

        box_galaxies_select = '''
        SELECT 
            gd2.id, 
            gd2.Name_GWGC, 
            gd2.Name_HyperLEDA, 
            gd2.Name_2MASS, 
            gd2.RA, 
            gd2._Dec, 
            gd2.z_dist, 
            gd2.B, 
            gd2.K, 
            hp_gd2_w.GalaxyProb 
        FROM 
            GalaxyDistance2 gd2 
        JOIN HealpixPixel_GalaxyDistance2 hp_gd2 on hp_gd2.GalaxyDistance2_id = gd2.id 
        JOIN HealpixPixel_GalaxyDistance2_Weight hp_gd2_w on hp_gd2_w.HealpixPixel_GalaxyDistance2_id = hp_gd2.id 
        JOIN HealpixPixel_Completeness hpc on hpc.HealpixPixel_id = hp_gd2.HealpixPixel_id 
        WHERE 
            hpc.HealpixMap_id = %s AND 
            gd2.RA BETWEEN %s AND %s AND 
            gd2._Dec BETWEEN %s AND %s 
        '''

        box_file_formatter = "%s/%s_%s_%s_%s_box.csv"
        non_box_file_formatter = "%s/%s_%s_%s_%s.csv"
        box_galaxies_formatter = "%s/%s_%s_box.csv"
        non_box_galaxies_formatter = "%s/%s_%s.csv"

        s_formatted_output_path = ""
        t_formatted_output_path = ""
        g_formatted_output_path = ""
        r_formatted_output_path = "%s/%s_%s.reg" % (formatted_healpix_dir, "REGION",
                                                    self.options.healpix_file.replace(",", "_"))
        g_select_to_execute = ""
        s_select_to_execute = ""
        t_select_to_execute = ""

        if is_Galaxy_box_query:
            g_formatted_output_path = box_galaxies_formatter % (formatted_healpix_dir, "GALAXIES",
                                                                       self.options.healpix_file.replace(",", "_"))
            g_select_to_execute = box_galaxies_select % (healpix_map_id, self.options.g_min_ra, self.options.g_max_ra,
                                                         self.options.g_min_dec, self.options.g_max_dec)
        else:
            g_formatted_output_path = non_box_galaxies_formatter % (formatted_healpix_dir, "GALAXIES",
                                                                       self.options.healpix_file.replace(",", "_"))
            g_select_to_execute = galaxies_select % healpix_map_id

        if is_Swope_box_query:
            s_formatted_output_path = box_file_formatter % (formatted_healpix_dir, SWOPE, self.options.prob_type,
                                                            self.options.s_cum_prob,
                                                            self.options.healpix_file.replace(",", "_"))
            if self.options.prob_type == _4D:
                s_select_to_execute = box_tile_select_4D % (s_band_F99, s_detector_id, healpix_map_id,
                                                            self.options.s_cum_prob, s_band_F99, self.options.extinct,
                                                            self.options.s_min_ra, self.options.s_max_ra,
                                                            self.options.s_min_dec, self.options.s_max_dec)
            else:
                s_select_to_execute = box_tile_select_2D % (s_band_F99, s_detector_id, healpix_map_id,
                                                            self.options.s_cum_prob, s_band_F99, self.options.extinct,
                                                            self.options.s_min_ra, self.options.s_max_ra,
                                                            self.options.s_min_dec, self.options.s_max_dec)
        else:
            s_formatted_output_path = non_box_file_formatter % (formatted_healpix_dir, SWOPE, self.options.prob_type,
                                                                self.options.s_cum_prob,
                                                                self.options.healpix_file.replace(",", "_"))
            if self.options.prob_type == _4D:
                s_select_to_execute = tile_select_4D % (s_band_F99, s_detector_id, healpix_map_id,
                                                        self.options.s_cum_prob, s_band_F99, self.options.extinct)
            else:
                s_select_to_execute = tile_select_2D % (s_band_F99, s_detector_id, healpix_map_id,
                                                        self.options.s_cum_prob, s_band_F99, self.options.extinct)

        if is_Thacher_box_query:
            t_formatted_output_path = box_file_formatter % (formatted_healpix_dir, THACHER, self.options.prob_type,
                                                            self.options.t_cum_prob,
                                                            self.options.healpix_file.replace(",", "_"))
            if self.options.prob_type == _4D:
                t_select_to_execute = box_tile_select_4D % (t_band_F99, t_detector_id, healpix_map_id,
                                                            self.options.t_cum_prob, t_band_F99, self.options.extinct,
                                                            self.options.t_min_ra, self.options.t_max_ra,
                                                            self.options.t_min_dec, self.options.t_max_dec)
            else:
                t_select_to_execute = box_tile_select_2D % (t_band_F99, t_detector_id, healpix_map_id,
                                                            self.options.t_cum_prob, t_band_F99, self.options.extinct,
                                                            self.options.t_min_ra, self.options.t_max_ra,
                                                            self.options.t_min_dec, self.options.t_max_dec)
        else:
            t_formatted_output_path = non_box_file_formatter % (formatted_healpix_dir, THACHER, self.options.prob_type,
                                                                self.options.t_cum_prob,
                                                                self.options.healpix_file.replace(",", "_"))
            if self.options.prob_type == _4D:
                t_select_to_execute = tile_select_4D % (t_band_F99, t_detector_id, healpix_map_id,
                                                        self.options.t_cum_prob, t_band_F99, self.options.extinct)
            else:
                t_select_to_execute = tile_select_2D % (t_band_F99, t_detector_id, healpix_map_id,
                                                        self.options.t_cum_prob, t_band_F99, self.options.extinct)

        s_tile_result = query_db([s_select_to_execute])[0]
        t_tile_result = query_db([t_select_to_execute])[0]
        galaxies_result = query_db([g_select_to_execute])[0]

        # Sort/limit galaxies_result
        sorted_galaxies_result = sorted(galaxies_result, key=lambda x: float(x[9]), reverse=True)
        galaxies_to_write = []
        if self.options.num_gal > 0:
            galaxies_to_write = sorted_galaxies_result[0:self.options.num_gal]

        def GetSexigesimalString(c):
            ra = c.ra.hms
            dec = c.dec.dms

            ra_string = "%02d:%02d:%05.2f" % (ra[0], ra[1], ra[2])
            if dec[0] >= 0:
                dec_string = "+%02d:%02d:%05.2f" % (dec[0], np.abs(dec[1]), np.abs(dec[2]))
            else:
                dec_string = "%03d:%02d:%05.2f" % (dec[0], np.abs(dec[1]), np.abs(dec[2]))

            # Python has a -0.0 object. If the deg is this (because object lies < 60 min south), the string formatter will drop the negative sign
            if c.dec < 0.0 and dec[0] == 0.0:
                dec_string = "-00:%02d:%05.2f" % (np.abs(dec[1]), np.abs(dec[2]))
            return (ra_string, dec_string)

        # E.g. Non box:
        #   <directory>/SWOPE_4D_0.9_LALInference_0.fits.gz.csv
        # Box:
        #   <directory>/SWOPE_2D_0.8_bayestar.fits.gz_box.csv


        with open(s_formatted_output_path, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)

            cols = []
            cols.append('# FieldName')
            cols.append('FieldRA')
            cols.append('FieldDec')
            cols.append('Telscope')
            cols.append('Filter')
            cols.append('ExpTime')
            cols.append('Priority')
            cols.append('Status')
            csvwriter.writerow(cols)

            for i, row in enumerate(s_tile_result):
                c = coord.SkyCoord(row[2], row[3], unit=(u.deg, u.deg))
                coord_str = GetSexigesimalString(c)

                cols = []

                cols.append(row[1])
                cols.append(coord_str[0])
                cols.append(coord_str[1])
                cols.append(SWOPE)
                cols.append(self.options.s_band)
                cols.append(str(self.options.s_exp_time))
                cols.append(row[4])
                cols.append('False')
                csvwriter.writerow(cols)

            print("Done writing out %s" % SWOPE)

        with open(t_formatted_output_path, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)

            cols = []
            cols.append('# FieldName')
            cols.append('FieldRA')
            cols.append('FieldDec')
            cols.append('Telscope')
            cols.append('Filter')
            cols.append('ExpTime')
            cols.append('Priority')
            cols.append('Status')
            csvwriter.writerow(cols)

            for i, row in enumerate(t_tile_result):
                c = coord.SkyCoord(row[2], row[3], unit=(u.deg, u.deg))
                coord_str = GetSexigesimalString(c)

                cols = []

                cols.append(row[1])
                cols.append(coord_str[0])
                cols.append(coord_str[1])
                cols.append(THACHER)
                cols.append(self.options.t_band)
                cols.append(str(self.options.t_exp_time))
                cols.append(row[4])
                cols.append('False')
                csvwriter.writerow(cols)

            print("Done writing out %s" % THACHER)

        with open(g_formatted_output_path,'w') as csvfile:

            csvwriter = csv.writer(csvfile)

            cols = []
            cols.append('# FieldName')
            cols.append('FieldRA')
            cols.append('FieldDec')
            cols.append('Telscope')
            cols.append('Filter')
            cols.append('ExpTime')
            cols.append('Priority')
            cols.append('Status')
            csvwriter.writerow(cols)

            for i, row in enumerate(galaxies_to_write):

                c = coord.SkyCoord(row[4], row[5], unit=(u.deg, u.deg))
                coord_str = GetSexigesimalString(c)

                cols = []

                Name_GWGC = row[1]
                Name_HyperLEDA = row[2]
                Name_2MASS = row[3]

                field_name = ""
                if Name_GWGC is not None:
                    field_name = Name_GWGC
                elif Name_HyperLEDA is not None:
                    field_name = "LEDA" + Name_HyperLEDA
                elif Name_2MASS is not None:
                    field_name = Name_2MASS

                if field_name == "":
                    raise("No field name!")

                cols.append(field_name)
                cols.append(coord_str[0])
                cols.append(coord_str[1])
                cols.append(GALAXY)
                cols.append(self.options.g_band)
                cols.append(str(self.options.g_exp_time))
                cols.append(row[9])
                cols.append('False')
                csvwriter.writerow(cols)

            print("Done w/ Galaxies")

        with open(r_formatted_output_path,'w') as csvfile:

            csvfile.write("# Region file format: DS9 version 4.0 global\n\n")
            csvfile.write("global color=lightgreen\n")
            csvfile.write("ICRS\n")

            for i, row in enumerate(galaxies_to_write):

                Name_GWGC = row[1]
                Name_HyperLEDA = row[2]
                Name_2MASS = row[3]

                field_name = ""
                if Name_GWGC is not None:
                    field_name = Name_GWGC
                elif Name_HyperLEDA is not None:
                    field_name = "LEDA" + Name_HyperLEDA
                elif Name_2MASS is not None:
                    field_name = Name_2MASS

                if field_name == "":
                    raise("No field name!")

                csvfile.write('circle(%s,%s,120") # width=4 text="%s"\n' % (row[4], row[5], field_name))

            print("Done w/ Region File")

        t2 = time.time()
        print("\n********* start DEBUG ***********")
        print("Extract Tiles execution time: %s" % (t2 - t1))
        print("********* end DEBUG ***********\n")


if __name__ == "__main__":
    useagestring = """python ExtractTiles.py [options]

Example non-box command line:
python ExtractTiles.py   --gw_id <gw id> --healpix_file <file name> 
                    --s_band r --s_exp_time 60 --s_cum_prob 0.90 
                    --t_band r --t_exp_time 120 --t_cum_prob 0.75 
                    --extinct 0.5 --prob_type 4D

Example box command line:
python ExtractTiles.py   --gw_id <gw id> --healpix_file <file name> 
                    --s_band r --s_exp_time 60 --s_cum_prob 0.90 
                    --s_min_ra 180.0 --s_max_ra 270.0 --s_min_dec -65.0 --s_max_dec 25.0 
                    --t_band r --t_exp_time 120 --t_cum_prob 0.75 
                    --t_min_ra 90.0 --t_max_ra 115.0 --t_min_dec -20.0 --t_max_dec 90.0 
                    --extinct 0.5 --prob_type 4D
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
    print("Teglon `ExtractTiles` execution time: %s" % duration)
    print("********* end DEBUG ***********\n")
