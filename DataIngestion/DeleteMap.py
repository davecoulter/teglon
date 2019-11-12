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

import astropy_healpix as ah
from astropy.table import Table

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


def delete_rows(delete_query, id_tuples_to_delete):
    # Start with
    batch_size = 10000
    i = 0
    j = batch_size
    k = len(id_tuples_to_delete)

    print("\nLength of records to DELETE: %s" % k)
    print("DELETE batch size: %s" % j)
    print("Starting loop...")

    number_of_deletes = k // batch_size
    if k % batch_size > 0:
        number_of_deletes += 1

    delete_num = 1
    # payload = []
    while j < k:
        t1 = time.time()
        print("%s:%s" % (i, j))

        id_string = ",".join([str(id_tup[0]) for id_tup in id_tuples_to_delete[i:j]])
        query_db([delete_query % id_string], commit=True)
        i = j
        j += batch_size

        t2 = time.time()

        print("\n********* start DEBUG ***********")
        print("DELETE %s/%s complete - execution time: %s" % (delete_num, number_of_deletes, (t2 - t1)))
        print("********* end DEBUG ***********\n")

        delete_num += 1

    t1 = time.time()
    print("%s:%s" % (i, k))

    id_string = ",".join([str(id_tup[0]) for id_tup in id_tuples_to_delete[i:k]])
    query_db([delete_query % id_string], commit=True)

    t2 = time.time()

    print("\n********* start DEBUG ***********")
    print("DELETE %s/%s complete - execution time: %s" % (delete_num, number_of_deletes, (t2 - t1)))
    print("********* end DEBUG ***********\n")

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

        parser.add_option('--healpix_map_id', default="-1", type="int",
                          help='''The integer primary key for the map to remove.''')

        return (parser)

    def main(self):
        is_error = False

        if self.options.healpix_map_id < 0:
            is_error = True
            print("HealpixMap_id is required.")

        healpix_map_select = '''
            SELECT GWID, URL, Filename FROM HealpixMap WHERE id = '%s'
        '''

        healpix_pixel_select = '''
            SELECT
                id
            FROM
                HealpixPixel
            WHERE
                HealpixMap_id = %s
        '''

        healpix_completeness_select = '''
            SELECT 
                id 
            FROM 
                HealpixPixel_Completeness 
            WHERE HealpixPixel_id IN (%s)
        '''

        healpix_pixel_galaxy_distance_select = '''
            SELECT 
                id 
            FROM 
                HealpixPixel_GalaxyDistance2 
            WHERE 
                HealpixPixel_id IN (%s)
        '''

        healpix_pixel_galaxy_distance_weight_select = '''
            SELECT 
                id 
            FROM 
                HealpixPixel_GalaxyDistance2_Weight  
            WHERE 
                HealpixPixel_GalaxyDistance2_id IN (%s)
        '''

        observed_tile_healpix_pixel_select = '''
            SELECT 
                id 
            FROM 
                ObservedTile_HealpixPixel 
            WHERE 
                HealpixPixel_id IN (%s) 
        '''

        observed_tile_select = '''
            SELECT 
                id 
            FROM 
                ObservedTile  
            WHERE 
                HealpixMap_id = %s   
        '''

        static_tile_healpix_pixel_select = '''
            SELECT 
                id 
            FROM 
                StaticTile_HealpixPixel 
            WHERE 
                HealpixPixel_id IN (%s) 
        '''

        healpix_completeness_delete = '''
            DELETE FROM HealpixPixel_Completeness 
            WHERE id IN (%s) AND id > 0 
        '''

        healpix_pixel_galaxy_distance_weight_delete = '''
            DELETE FROM HealpixPixel_GalaxyDistance2_Weight 
            WHERE id IN (%s) AND id > 0 
        '''

        healpix_pixel_galaxy_distance_delete = '''
            DELETE FROM HealpixPixel_GalaxyDistance2 
            WHERE id IN (%s) AND id > 0 
        '''

        observed_tile_healpix_pixel_delete = '''
            DELETE FROM ObservedTile_HealpixPixel 
            WHERE id IN (%s) AND id > 0 
        '''

        observed_tile_delete = '''
            DELETE FROM ObservedTile  
            WHERE id IN (%s) AND id > 0 
        '''

        static_tile_healpix_pixel_delete = '''
            DELETE FROM StaticTile_HealpixPixel 
            WHERE id IN (%s) AND id > 0 
        '''

        healpix_pixel_delete = '''
            DELETE FROM HealpixPixel 
            WHERE id IN (%s) AND id > 0 
        '''

        healpix_map_delete = '''
            DELETE FROM HealpixMap    
            WHERE id = %s 
        '''

        observed_tile_result = []
        healpix_pixel_result = []
        healpix_pixel_id_string = ""
        healpix_map_result = query_db([healpix_map_select % self.options.healpix_map_id])[0]
        if len(healpix_map_result) > 0:
            healpix_pixel_result = query_db([healpix_pixel_select % self.options.healpix_map_id])[0]
            healpix_pixel_id_string = ",".join([str(hp[0]) for hp in healpix_pixel_result])
            observed_tile_result = query_db([observed_tile_select % self.options.healpix_map_id])[0]

        observed_tile_healpix_pixel_result = []
        static_tile_healpix_pixel_result = []
        healpix_completeness_result = []
        healpix_pixel_galaxy_distance_result = []
        if len(healpix_pixel_result) > 0:
            observed_tile_healpix_pixel_result = query_db([observed_tile_healpix_pixel_select % healpix_pixel_id_string])[0]
            static_tile_healpix_pixel_result = query_db([static_tile_healpix_pixel_select % healpix_pixel_id_string])[0]
            healpix_completeness_result = query_db([healpix_completeness_select % healpix_pixel_id_string])[0]
            healpix_pixel_galaxy_distance_result = query_db([healpix_pixel_galaxy_distance_select % healpix_pixel_id_string])[0]

        healpix_pixel_galaxy_distance_weight_result = []
        if len(healpix_pixel_galaxy_distance_result) > 0:
            healpix_pixel_galaxy_distance_id_string = ",".join([str(hpgd[0]) for hpgd in healpix_pixel_galaxy_distance_result])
            healpix_pixel_galaxy_distance_weight_result = query_db([healpix_pixel_galaxy_distance_weight_select % healpix_pixel_galaxy_distance_id_string])[0]


        if len(healpix_map_result) > 0:
            print("Are you sure you want to delete:\n\tGWID: %s\n\tURL: %s\n\tFile: %s" % healpix_map_result[0])

        print('''Number of other records to be deleted:
            HealpixPixel_Completeness: %s
            HealpixPixel_GalaxyDistance2_Weight: %s
            HealpixPixel_GalaxyDistance2: %s
            ObservedTile_HealpixPixel: %s
            ObservedTile: %s
            StaticTile_HealpixPixel: %s
            HealpixPixel: %s
                ''' % (len(healpix_completeness_result),
                       len(healpix_pixel_galaxy_distance_weight_result),
                       len(healpix_pixel_galaxy_distance_result),
                       len(observed_tile_healpix_pixel_result),
                       len(observed_tile_result),
                       len(static_tile_healpix_pixel_result),
                       len(healpix_pixel_result)))

        if len(healpix_completeness_result) > 0:
            print("DELETE %s HealpixPixel_Completeness" % len(healpix_completeness_result))
            delete_rows(healpix_completeness_delete, healpix_completeness_result)

        if len(healpix_pixel_galaxy_distance_weight_result) > 0:
            print("DELETE %s HealpixPixel_GalaxyDistance2_Weight" % len(healpix_pixel_galaxy_distance_weight_result))
            delete_rows(healpix_pixel_galaxy_distance_weight_delete, healpix_pixel_galaxy_distance_weight_result)

        if len(healpix_pixel_galaxy_distance_result) > 0:
            print("DELETE %s HealpixPixel_GalaxyDistance2" % len(healpix_pixel_galaxy_distance_result))
            delete_rows(healpix_pixel_galaxy_distance_delete, healpix_pixel_galaxy_distance_result)

        if len(observed_tile_healpix_pixel_result) > 0:
            print("DELETE %s ObservedTile_HealpixPixel" % len(observed_tile_healpix_pixel_result))
            delete_rows(observed_tile_healpix_pixel_delete, observed_tile_healpix_pixel_result)

        if len(observed_tile_result) > 0:
            print("DELETE %s ObservedTile" % len(observed_tile_result))
            delete_rows(observed_tile_delete, observed_tile_result)

        if len(static_tile_healpix_pixel_result) > 0:
            print("DELETE %s StaticTile_HealpixPixel" % len(static_tile_healpix_pixel_result))
            delete_rows(static_tile_healpix_pixel_delete, static_tile_healpix_pixel_result)

        if len(healpix_pixel_result) > 0:
            print("DELETE %s HealpixPixel" % len(healpix_pixel_result))
            delete_rows(healpix_pixel_delete, healpix_pixel_result)

        if len(healpix_map_result) > 0:
            print("DELETE HealpixMap id=%s" % str(self.options.healpix_map_id))
            delete_map = healpix_map_delete % str(self.options.healpix_map_id)
            query_db([delete_map], commit=True)


if __name__ == "__main__":
    useagestring = """python DeleteMap.py [options]
    
python DeleteMap.py --healpix_map_id <database id>
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
    print("Teglon `DeleteMap` execution time: %s" % duration)
    print("********* end DEBUG ***********\n")


