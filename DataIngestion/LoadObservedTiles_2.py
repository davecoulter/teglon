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

        parser.add_option('--tile_dir', default="../Events/{GWID}/ObservedTiles", type="str",
                          help='Directory for where to look for observed tiles to import.')

        parser.add_option('--tile_file', default="", type="str", help='File that contains the tile observations.')

        parser.add_option('--tele', default="", type="str",
                          help='Telescope abbreviation that `tile_file` corresponds to.')

        return (parser)

    def main(self):

        # Band abbreviation, band_id mapping
        band_mapping = {
            "g": "SDSS g",
            "r": "SDSS r",
            "i": "SDSS i",
            "Clear": "Clear",
            "J": "UKIRT J"
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

        is_error = False

        # Parameter checks
        if self.options.gw_id == "":
            is_error = True
            print("GWID is required.")

        if self.options.healpix_file == "":
            is_error = True
            print("Healpix file is required.")

        if self.options.tile_file == "":
            is_error = True
            print("Tile file is required.")

        if self.options.tele == "":
            is_error = True
            print("Telescope abbreviation is required.")

        if is_error:
            print("Exiting...")
            return 1

        formatted_healpix_dir = self.options.healpix_dir
        if "{GWID}" in formatted_healpix_dir:
            formatted_healpix_dir = formatted_healpix_dir.replace("{GWID}", self.options.gw_id)

        formatted_tile_dir = self.options.tile_dir
        if "{GWID}" in formatted_tile_dir:
            formatted_tile_dir = formatted_tile_dir.replace("{GWID}", self.options.gw_id)

        hpx_path = "%s/%s" % (formatted_healpix_dir, self.options.healpix_file)
        tile_path = "%s/%s" % (formatted_tile_dir, self.options.tile_file)

        # Check if the above files exist...
        if not os.path.exists(hpx_path):
            is_error = True
            print("Healpix file `%s` does not exist." % hpx_path)

        if not os.path.exists(tile_path):
            is_error = True
            print("Tile file `%s` does not exist." % tile_path)

        if self.options.tele not in detector_mapping:
            is_error = True
            print("Unknown telescope abbreviation: %s " % self.options.tele)

        # TODO: STANDARDIZE FILE FORMATTING
        # God damn you Charlie, you beautiful son of a bitch.

        # tile_files = []
        # for file in os.listdir(formatted_tile_dir):
        # 	if file.endswith(".dat"):
        # 		tile_files.append("%s/%s" % (formatted_tile_dir, file))

        # if len(tile_files) <= 0:
        # 	is_error = True
        # 	print("There are no tiles to process.")

        if is_error:
            print("Exiting...")
            return 1

        print("\tLoading NSIDE 128 pixels...")
        nside128 = 128
        N128_dict = None
        with open('N128_dict.pkl', 'rb') as handle:
            N128_dict = pickle.load(handle)
        del handle

        print("\tLoading existing EBV...")
        ebv = None
        with open('ebv.pkl', 'rb') as handle:
            ebv = pickle.load(handle)

        # Get Map ID
        healpix_map_select = "SELECT id, NSIDE FROM HealpixMap WHERE GWID = '%s' and Filename = '%s'"
        healpix_map_id = int(query_db([healpix_map_select % (self.options.gw_id, self.options.healpix_file)])[0][0][0])
        healpix_map_nside = int(
            query_db([healpix_map_select % (self.options.gw_id, self.options.healpix_file)])[0][0][1])

        print("Get map pixel")
        map_pixel_select = "SELECT id, HealpixMap_id, Pixel_Index, Prob, Distmu, Distsigma, Distnorm, Mean, Stddev, Norm, N128_SkyPixel_id FROM HealpixPixel WHERE HealpixMap_id = %s;"
        q = map_pixel_select % healpix_map_id
        map_pixels = query_db([q])[0]
        print("Retrieved %s map pixels..." % len(map_pixels))

        # Initialize map pix dict for later access
        map_pixel_dict = {}
        for p in map_pixels:
            map_pixel_dict[int(p[2])] = p

        band_select = "SELECT id, Name, F99_Coefficient FROM Band WHERE `Name`='%s'"
        detector_select_by_name = "SELECT id, Name, Deg_width, Deg_height, Deg_radius, Area, MinDec, MaxDec FROM Detector WHERE Name='%s'"

        # print("Processing %s tiles" % len(tile_files))
        # observed_tiles = []
        obs_tile_insert_data = []
        detectors = {}

        tele_name = detector_mapping[self.options.tele]
        detector_result = query_db([detector_select_by_name % tele_name])[0][0]
        detector = Detector(detector_result[1], float(detector_result[2]), float(detector_result[2]))
        detector.id = int(detector_result[0])
        detector.area = float(detector_result[5])

        if detector.name not in detectors:
            detectors[detector.name] = detector
        print("Processing `%s` for %s" % (tile_path, detector.name))

        # Iterate over lines of a tile
        with open(tile_path, 'r') as csvfile:
            # Read CSV lines
            csvreader = csv.reader(csvfile, delimiter=' ', skipinitialspace=True)
            for row in csvreader:

                file_name = row[0]
                field_name = row[1]
                ra = float(row[2])
                dec = float(row[3])
                mjd = float(row[4])
                band = row[5].strip()

                exp_time = None
                try:
                    exp_time = float(row[6])
                    if exp_time <= 0.0:
                        exp_time = None
                except:
                    pass

                mag_lim = None
                try:
                    mag_lim = float(row[7])
                except:
                    pass

                # Get Band_id
                band_map = band_mapping[band]
                band_results = query_db([band_select % band_map])[0][0]

                band_id = band_results[0]
                band_name = band_results[1]
                band_F99 = float(band_results[2])

                c = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg))
                n128_index = hp.ang2pix(nside128, 0.5 * np.pi - c.dec.radian, c.ra.radian)  # theta, phi
                n128_id = N128_dict[n128_index]

                t = Tile(c.ra.degree, c.dec.degree, detector.deg_width, detector.deg_height, int(healpix_map_nside))
                t.field_name = field_name
                t.N128_pixel_id = n128_id
                t.N128_pixel_index = n128_index
                t.mwe = ebv[n128_index] * band_F99
                t.mjd = mjd
                t.exp_time = exp_time
                t.mag_lim = mag_lim

                # observed_tiles.append(t)
                obs_tile_insert_data.append((
                    detector.id,
                    t.field_name,
                    t.ra_deg,
                    t.dec_deg,
                    "POINT(%s %s)" % (t.dec_deg, t.ra_deg - 180.0),  # Dec, RA order due to MySQL convention for lat/lon
                    t.query_polygon_string,
                    str(t.mwe),
                    t.N128_pixel_id,
                    band_id,
                    t.mjd,
                    t.exp_time,
                    t.mag_lim,
                    healpix_map_id))

        # iterate over tiles in tile_files
        # for tf in tile_files:

        # 	# Read Tile File CSV and get the telescope (by file naming convention)
        # 	file_name = tf.split("/")[-1]
        # 	tele_abbr = file_name.split("_")[0]
        # 	tele_name = detector_mapping[tele_abbr]

        # 	detector_result = query_db([detector_select_by_name % tele_name])[0][0]
        # 	detector = Detector(detector_result[1], float(detector_result[2]), float(detector_result[2]))
        # 	detector.id = int(detector_result[0])
        # 	detector.area = float(detector_result[5])

        # 	if detector.name not in detectors:
        # 		detectors[detector.name] = detector
        # 	print("Processing `%s` for %s" % (file_name, detector.name))

        # 	with open(tf,'r') as csvfile:

        # 		# Read CSV lines
        # 		csvreader = csv.reader(csvfile, delimiter=' ',skipinitialspace=True)

        # 		for row in csvreader:

        # 			file_name = row[0]
        # 			field_name = row[1]
        # 			ra = float(row[2])
        # 			dec = float(row[3])
        # 			# ra = row[2]
        # 			# dec = row[3]
        # 			exp_time = float(row[4])
        # 			mjd = float(row[5])
        # 			band = row[6]
        # 			mag_lim = float(row[7]) if row[7] != '___' else None
        # 			# mag_lim = None

        # 			# Get Band_id
        # 			band_map = band_mapping[band]
        # 			band_results = query_db([band_select % band_map])[0][0]

        # 			band_id = band_results[0]
        # 			band_name = band_results[1]
        # 			band_F99 = float(band_results[2])

        # 			c = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg))
        # 			# c = coord.SkyCoord(ra, dec, unit=(u.hour, u.deg))
        # 			n128_index = hp.ang2pix(nside128, 0.5*np.pi - c.dec.radian, c.ra.radian) # theta, phi
        # 			n128_id = N128_dict[n128_index]

        # 			t = Tile(c.ra.degree, c.dec.degree, detector.deg_width, detector.deg_height, int(healpix_map_nside))
        # 			t.field_name = field_name
        # 			t.N128_pixel_id = n128_id
        # 			t.N128_pixel_index = n128_index
        # 			t.mwe = ebv[n128_index]*band_F99
        # 			t.mjd = mjd
        # 			t.exp_time = exp_time
        # 			t.mag_lim = mag_lim

        # 			# observed_tiles.append(t)
        # 			obs_tile_insert_data.append((
        # 				detector.id,
        # 				t.field_name,
        # 				t.ra_deg,
        # 				t.dec_deg,
        # 				"POINT(%s %s)" % (t.dec_deg, t.ra_deg - 180.0),  # Dec, RA order due to MySQL convention for lat/lon
        # 				t.query_polygon_string,
        # 				str(t.mwe),
        # 				t.N128_pixel_id,
        # 				band_id,
        # 				t.mjd,
        # 				t.exp_time,
        # 				t.mag_lim,
        # 				healpix_map_id))

        insert_observed_tile = '''
			INSERT INTO 
				ObservedTile (Detector_id, FieldName, RA, _Dec, Coord, Poly, EBV, N128_SkyPixel_id, Band_id, MJD, Exp_Time, Mag_Lim, HealpixMap_id) 
			VALUES (%s, %s, %s, %s, ST_PointFromText(%s, 4326), ST_GEOMFROMTEXT(%s, 4326), %s, %s, %s, %s, %s, %s, %s)
		'''

        print("Inserting %s tiles..." % len(obs_tile_insert_data))
        batch_insert(insert_observed_tile, obs_tile_insert_data)
        print("Done...")

        # Associate map pixels with observed tiles
        print("Building observed tile-healpix map pixel relation...")

        obs_tile_select = '''
			SELECT id, Detector_id, FieldName, RA, _Dec, Coord, Poly, EBV, N128_SkyPixel_id, Band_id, MJD, Exp_Time, Mag_Lim, HealpixMap_id 
			FROM ObservedTile 
			WHERE Detector_id = %s and HealpixMap_id = %s 
		'''

        # Obtain the tile with id's so they can be used in later INSERTs
        tile_pixel_data = []
        for d_name, d in detectors.items():
            obs_tiles_for_detector = query_db([obs_tile_select % (d.id, healpix_map_id)])[0]

            print("Observed Tiles for %s: %s" % (d.name, len(obs_tiles_for_detector)))
            for otfd in obs_tiles_for_detector:

                ot_id = int(otfd[0])
                ra = float(otfd[3])
                dec = float(otfd[4])
                t = Tile(ra, dec, d.deg_width, d.deg_height, healpix_map_nside)
                t.id = ot_id

                for p in t.enclosed_pixel_indices:
                    tile_pixel_data.append((t.id, map_pixel_dict[p][0]))

        print("Length of tile_pixel_data: %s" % len(tile_pixel_data))

        tile_pixel_upload_csv = "%s/ObservedTiles/%s_tile_pixel_upload.csv" % (formatted_healpix_dir, detector.name)

        # Create CSV
        try:
            t1 = time.time()
            print("Creating `%s`" % tile_pixel_upload_csv)
            with open(tile_pixel_upload_csv, 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                for data in tile_pixel_data:
                    csvwriter.writerow(data)

            t2 = time.time()
            print("\n********* start DEBUG ***********")
            print("Tile-Pixel CSV creation execution time: %s" % (t2 - t1))
            print("********* end DEBUG ***********\n")
        except Error as e:
            print("Error in creating Tile-Pixel CSV:\n")
            print(e)
            print("\nExiting")
            return 1

        # clean up
        print("freeing `tile_pixel_data`...")
        del tile_pixel_data

        print("Bulk uploading `tile_pixel_data`...")
        ot_hp_upload_sql = """LOAD DATA LOCAL INFILE '%s' 
					INTO TABLE ObservedTile_HealpixPixel 
					FIELDS TERMINATED BY ',' 
					LINES TERMINATED BY '\n' 
					(ObservedTile_id, HealpixPixel_id);"""

        success = bulk_upload(ot_hp_upload_sql % tile_pixel_upload_csv)
        if not success:
            print("\nUnsuccessful bulk upload. Exiting...")
            return 1

        try:
            print("Removing `%s`..." % tile_pixel_upload_csv)
            os.remove(tile_pixel_upload_csv)

            print("... Done")
        except Error as e:
            print("Error in file removal")
            print(e)
            print("\nExiting")
            return 1


if __name__ == "__main__":
    useagestring = """python Generate_Tiles.py [options]

Example with healpix_dir defaulted to 'Events/<gwid>':
python LoadMap.py --gw_id <>gwid --healpix_file <filename>

Example with healpix_dir specified:
python LoadMap.py --gw_id <gwid> --healpix_dir Events/<directory name> --healpix_file <filename>
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
    print("Teglon `LoadObservedTiles` execution time: %s" % duration)
    print("********* end DEBUG ***********\n")
