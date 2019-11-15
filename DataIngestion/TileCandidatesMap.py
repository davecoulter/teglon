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

        db = my.connect(host=db_host, user=db_user, passwd=db_pwd, db=db_name, port=3306)
        cursor = db.cursor()

        for q in query_list:
            cursor.execute(q)

            if commit: # used for updates, etc
                db.commit()

            streamed_results = []
            print("\tfetching results...")
            while True:
                r = cursor.fetchmany(1000000)
                count = len(r)
                streamed_results += r
                size_in_mb = sys.getsizeof(streamed_results)/1.0e+6

                print("\t\tfetched: %s; current length: %s; running size: %0.3f MB" % (count, len(streamed_results), size_in_mb))

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

    number_of_queries = len(query_list)//batch_size
    if len(query_list) % batch_size > 0:
        number_of_queries += 1

    query_num = 1
    payload = []
    while jj < kk:
        t1 = time.time()

        print("%s:%s" % (ii,jj))
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

    print("\n%s:%s" % (ii,kk))

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

    number_of_inserts = len(insert_data)//batch_size
    if len(insert_data) % batch_size > 0:
        number_of_inserts += 1

    insert_num = 1
    payload = []
    while j < k:
        t1 = time.time()

        print("%s:%s" % (i,j))
        payload = insert_data[i:j]

        if insert_records(insert_statement, payload):
            i = j
            j += batch_size
        else:
            raise("Error inserting batch! Exiting...")

        t2 = time.time()

        print("\n********* start DEBUG ***********")
        print("INSERT %s/%s complete - execution time: %s" % (insert_num, number_of_inserts, (t2 - t1)))
        print("********* end DEBUG ***********\n")

        insert_num += 1

    print("Out of loop...")

    t1 = time.time()

    print("\n%s:%s" % (i,k))

    payload = insert_data[i:k]
    if not insert_records(insert_statement, payload):
        raise("Error inserting batch! Exiting...")

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
    phi, theta = np.radians(float(galaxy_info[8])), 0.5*np.pi - np.radians(float(galaxy_info[9]))

    # map NSIDE is last argument of galaxy_info
    # return the galaxy_id with the pixel index in the NSIDE of the healpix map
    return (galaxy_info[0], hp.ang2pix(int(galaxy_info[-1]), theta, phi))

class Teglon:

    def add_options(self, parser=None, usage=None, config=None):
        import optparse
        if parser == None:
            parser = optparse.OptionParser(usage=usage, conflict_handler="resolve")


        parser.add_option('--gw_id', default="", type="str",
                        help='LIGO superevent name, e.g. `S190425z` ')

        parser.add_option('--healpix_dir', default='../Events/{GWID}', type="str",help='Directory for where to look for the healpix file.')

        parser.add_option('--candidate_dir', default='../Events/{GWID}/Candidates', type="str",
                          help='Directory for where to look for the healpix file.')

        parser.add_option('--healpix_file', default="", type="str", help='Healpix filename.')

        parser.add_option('--candidate_file', default="", type="str", help='Filename of candidates to cross-reference.')

        return(parser)

    def main(self):

        healpix_map_select = "SELECT id, RescaledNSIDE FROM HealpixMap WHERE GWID = '%s' and Filename = '%s'"
        healpix_map_id = -1
        healpix_map_nside = -1

        is_error = False

        if self.options.healpix_file == "":
            is_error = True
            print("You must specify which healpix file to process.")

        if self.options.candidate_file == "":
            is_error = True
            print("You must specify which candidate file to process.")

        # Parameter checks
        if self.options.gw_id == "":
            is_error = True
            print("GWID is required.")
        else:
            if self.options.healpix_file != "":
                try:
                    # Get Map ID
                    map_result = query_db([healpix_map_select % (self.options.gw_id, self.options.healpix_file)])[0][0]
                    healpix_map_id = map_result[0]
                    healpix_map_nside = map_result[1]
                except Error as e:
                    print("\nMap lookup error:")
                    print(e)
                    is_error = True

        if healpix_map_id < 0 or healpix_map_nside < 0:
            is_error = True
            print("Could not find GW event based on --gw_id=%s and --healpix_file=%s." % (self.options.gw_id,
                                                                                          self.options.healpix_file))


        formatted_healpix_dir = self.options.healpix_dir
        formatted_candidate_dir = self.options.candidate_dir
        if "{GWID}" in formatted_healpix_dir:
            formatted_healpix_dir = formatted_healpix_dir.replace("{GWID}", self.options.gw_id)

        if "{GWID}" in formatted_candidate_dir:
            formatted_candidate_dir = formatted_candidate_dir.replace("{GWID}", self.options.gw_id)


        hpx_path = "%s/%s" % (formatted_healpix_dir, self.options.healpix_file)
        candidate_file_path = "%s/%s" % (formatted_candidate_dir, self.options.candidate_file)

        target_file_name = candidate_file_path.split('/')[-1].split(".")[0]
        print("Target File name: %s" % target_file_name)

        resolved_candidate_file_path = "%s/%s" % (formatted_candidate_dir, target_file_name + "_resolved.txt")

        if is_error:
            print("Exiting...")
            return 1

        tile_select = '''
            SELECT id, Detector_id, FieldName, RA, _Dec, Coord, Poly, EBV, N128_SkyPixel_id 
            FROM StaticTile 
            WHERE Detector_id = %s and 
                st_contains(Poly, ST_GeomFromText('POINT(%s %s)', 4326));
        '''

        _2d_prob_percentile = '''
            SELECT SUM(hp.Prob)
            FROM HealpixPixel hp
            WHERE
                hp.HealpixMap_id = %s AND
                hp.Prob >= 
                    (SELECT _hp.Prob 
                    FROM HealpixPixel _hp
                    WHERE _hp.Pixel_Index = %s and _hp.HealpixMap_id = %s)
            ORDER BY hp.Prob DESC;
        '''

        _4d_prob_percentile = '''
            SELECT SUM(hpc.NetPixelProb)
            FROM HealpixPixel_Completeness hpc
            WHERE 
                hpc.HealpixMap_id = %s AND 
                hpc.NetPixelProb >= 
                (SELECT _hpc.NetPixelProb 
                FROM HealpixPixel _hp
                JOIN HealpixPixel_Completeness _hpc on _hpc.HealpixPixel_id = _hp.id
                WHERE _hp.Pixel_Index = %s AND _hp.HealpixMap_id = %s)
            ORDER BY hpc.NetPixelProb DESC;

        '''

        SWOPE_id = 1
        THACHER_id = 3
        candidates = []
        with open(candidate_file_path,'r') as csvfile:

            csvreader = csv.reader(csvfile, delimiter=',',skipinitialspace=True)
            next(csvreader)

            for row in csvreader:

                name = row[0]
                ra = row[1]
                dec = row[2]
                c = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg))

                print("Processing `%s`" % name)

                swope_tile = query_db([tile_select % (SWOPE_id, c.dec.degree, c.ra.degree - 180.0)])[0][0]
                thacher_tile = query_db([tile_select % (THACHER_id, c.dec.degree, c.ra.degree - 180.0)])[0][0]

                pixel_index = hp.ang2pix(int(healpix_map_nside), 0.5*np.pi - c.dec.radian, c.ra.radian)
                print("\tPixel Index: %s" % pixel_index)

                _2d = query_db([_2d_prob_percentile % (healpix_map_id, pixel_index, healpix_map_id)])[0][0][0]
                _4d = query_db([_4d_prob_percentile % (healpix_map_id, pixel_index, healpix_map_id)])[0][0][0]

                t = (name, ra, dec, c.ra.degree, c.dec.degree, healpix_map_nside, pixel_index, swope_tile[2], thacher_tile[2], _2d, _4d)
                candidates.append(t)


        with open(resolved_candidate_file_path,'w') as csvfile:
            csvwriter = csv.writer(csvfile)

            cols = []
            cols.append('# Name')
            cols.append('RA')
            cols.append('Dec')
            cols.append('RA_deg')
            cols.append('Dec_deg')
            cols.append('Map_NSIDE')
            cols.append('Pixel_Index')
            cols.append('Swope_Tile')
            cols.append('Thacher_Tile')
            cols.append('2D_Percentile')
            cols.append('Teglon_Percentile')
            csvwriter.writerow(cols)

            for i, row in enumerate(candidates):
                csvwriter.writerow(row)

        print("Done.")



if __name__ == "__main__":

    useagestring="""python TileCandidatesMap.py [options]

Example with healpix_dir defaulted to 'Events/<gwid>':
python TileCandidatesMap.py --gw_id <gwid> --healpix_file <filename> --candidate_file <filename>

INPUT FORMAT for `candidate_file`

Name, RA (decimal), DEC (decimal)
2019nmd,12.87085,-22.47137778
...

"""

    start = time.time()

    teglon = Teglon()
    parser = teglon.add_options(usage=useagestring)
    options,  args = parser.parse_args()
    teglon.options = options

    teglon.main()

    end = time.time()
    duration = (end - start)
    print("\n********* start DEBUG ***********")
    print("Teglon `TileCandidatesMap` execution time: %s" % duration)
    print("********* end DEBUG ***********\n")


