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
build_map = True
build_pixels = True
build_tile_pixel_relation = True
build_galaxy_pixel_relation = True
build_completeness_func = True
build_galaxy_weights = True


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


initialize_start = time.time()

# Set up dustmaps config
config["data_dir"] = "./"

# Generate all pixel indices
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)


def initialize_tile(tile):
    tile.enclosed_pixel_indices
    # tile.query_polygon_string
    return tile


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

        parser.add_option('--gw_id', default="", type="str",
                          help='LIGO superevent name, e.g. `S190425z` ')

        parser.add_option('--healpix_dir', default='../Events/{GWID}', type="str",
                          help='Directory for where to look for the healpix file.')

        parser.add_option('--healpix_file', default="", type="str", help='healpix filename.')

        parser.add_option('--orig_res', action="store_true", default=False,
                          help='''Upload the healpix file at the native resolution (default is NSIDE = 256)''')

        return (parser)

    def main(self):

        healpix_map_select = "SELECT id, NSIDE FROM HealpixMap WHERE GWID = '%s' and Filename = '%s'"

        is_error = False

        # Parameter checks
        if self.options.gw_id == "":
            is_error = True
            print("GWID is required.")

        formatted_healpix_dir = self.options.healpix_dir
        formatted_candidates_dir = self.options.healpix_dir + "/Candidates"
        formatted_obs_tiles_dir = self.options.healpix_dir + "/ObservedTiles"
        formatted_model_dir = self.options.healpix_dir + "/ModelDetection"
        if "{GWID}" in formatted_healpix_dir:
            formatted_healpix_dir = formatted_healpix_dir.replace("{GWID}", self.options.gw_id)
            formatted_candidates_dir = formatted_candidates_dir.replace("{GWID}", self.options.gw_id)
            formatted_obs_tiles_dir = formatted_obs_tiles_dir.replace("{GWID}", self.options.gw_id)
            formatted_model_dir = formatted_model_dir.replace("{GWID}", self.options.gw_id)

        hpx_path = "%s/%s" % (formatted_healpix_dir, self.options.healpix_file)

        if self.options.healpix_file == "":
            is_error = True
            print("You must specify which healpix file to process.")

        map_check = query_db([healpix_map_select % (self.options.gw_id, self.options.healpix_file)])[0]
        if len(map_check) > 0:
            is_error = True
            print('''The combination of GWID `%s` and healpix file `%s` already exists in the db. Please choose a unique 
combination''' % (self.options.gw_id, self.options.healpix_file))

        if is_error:
            print("Exiting...")
            return 1

        ### Quantities that are required ###
        nside128 = 128
        nside256 = 256
        prob = None
        distmu = None
        distsigma = None
        distnorm = None
        header_gen = None
        map_nside = None
        healpix_map_id = None
        N128_dict = None
        map_pixel_dict = None

        if not build_map:
            print("Skipping build map...")
            print("\tLoading existing map...")
            orig_prob, orig_distmu, orig_distsigma, orig_distnorm, header_gen = hp.read_map(hpx_path, field=(0, 1, 2, 3), h=True)

            orig_npix = len(orig_prob)
            print("\tOriginal number of pix in '%s': %s" % (self.options.healpix_file, orig_npix))

            orig_map_nside = hp.npix2nside(orig_npix)

            # Check if NSIDE is > 256. If it is, and the --orig_res flag is not specified, rescale map to NSIDE 256
            if orig_map_nside > nside256 and not self.options.orig_res:
                print("Rescaling map to NSIDE = 256")
                rescaled_prob, rescaled_distmu, rescaled_distsigma, rescaled_distnorm = hp.ud_grade([orig_prob,
                                                                                                    orig_distmu,
                                                                                                    orig_distsigma,
                                                                                                    orig_distnorm],
                                                                                                    nside_out=nside256,
                                                                                                    order_in="RING",
                                                                                                    order_out="RING")
                rescaled_npix = len(rescaled_prob)
                print("\tRescaled number of pix in '%s': %s" % (self.options.healpix_file, rescaled_npix))

                rescaled_map_nside = hp.npix2nside(rescaled_npix)
                print("\tRescaled resolution (nside) of '%s': %s\n" % (self.options.healpix_file, rescaled_map_nside))

                original_pix_per_rescaled_pix = orig_npix / rescaled_npix
                print("Original pix per rescaled pix for %s" % original_pix_per_rescaled_pix)

                print("Renormalizing and initializing rescaled map...")
                rescaled_prob = rescaled_prob * original_pix_per_rescaled_pix

                prob = rescaled_prob
                distmu = rescaled_distmu
                distsigma = rescaled_distsigma
                distnorm = rescaled_distnorm
                map_nside = rescaled_map_nside
            else:
                prob = orig_prob
                distmu = orig_distmu
                distsigma = orig_distsigma
                distnorm = orig_distnorm
                map_nside = orig_map_nside

            print("Getting map id")
            healpix_map_id = query_db([healpix_map_select % (self.options.gw_id, self.options.healpix_file)])[0][0][0]
            print("Done map id")
        else:
            print("Building map...")
            # Create target directory if it doesn't already exist...
            try:
                os.mkdir(formatted_healpix_dir)
                print("\n\nDirectory ", formatted_healpix_dir, " Created ")

                # Create the `ObservedTiles` and `Candidates` subdirectories
                os.mkdir(formatted_candidates_dir)
                os.mkdir(formatted_obs_tiles_dir)
                os.mkdir(formatted_model_dir)
            except FileExistsError:
                print("\n\nDirectory ", formatted_healpix_dir, " already exists")

            # Get file -- ADD check and only get if you need to...
            try:
                print("Downloading `%s`..." % self.options.healpix_file)
                t1 = time.time()
                gw_file_formatter = "https://gracedb.ligo.org/apiweb/superevents/%s/files/%s"
                gw_file = gw_file_formatter % (self.options.gw_id, self.options.healpix_file)

                with urllib.request.urlopen(gw_file) as response:
                    with open(hpx_path, 'wb') as out_file:
                        shutil.copyfileobj(response, out_file)
                t2 = time.time()

                print("\n********* start DEBUG ***********")
                print("Downloading `%s` - execution time: %s" % (self.options.healpix_file, (t2 - t1)))
                print("********* end DEBUG ***********\n")

            except urllib.error.HTTPError as e:
                print("\nError:")
                print(e.code, gw_file)
                print("\n\tExiting...")
                return 1
            except urllib.error.URLError as e:
                print("\nError:")
                if hasattr(e, 'reason'):
                    print(e.reason, gw_file)
                elif hasattr(e, 'code'):
                    print(e.code, gw_file)
                print("\n\tExiting...")
                return 1
            except Error as e:
                print("\nError:")
                print(e)
                print("\n\tExiting...")
                return 1

            # Get event details from GraceDB page
            try:
                print("Scraping `%s` details..." % self.options.gw_id)
                t1 = time.time()
                gw_url_formatter = "https://gracedb.ligo.org/superevents/%s/view/"
                gw_url = gw_url_formatter % self.options.gw_id

                r = requests.get(gw_url)
                data = r.text
                soup = BeautifulSoup(data, "lxml")
                tds = soup.find("table", {"class": "superevent"}).find_all('tr')[1].find_all('td')

                FAR_hz = tds[3].text
                FAR_per_yr = tds[4].text
                t_start = tds[5].text
                t_0 = tds[6].text
                t_end = tds[7].text
                UTC_submission_time = parse(tds[8].text)

                t2 = time.time()
                print("\n********* start DEBUG ***********")
                print("Scraping `%s` details - execution time: %s" % (self.options.gw_id, (t2 - t1)))
                print("********* end DEBUG ***********\n")

            except urllib.error.HTTPError as e:
                print("\nError:")
                print(e.code, gw_file)
                print("\n\tExiting...")
                return 1
            except urllib.error.URLError as e:
                print("\nError:")
                if hasattr(e, 'reason'):
                    print(e.reason, gw_file)
                elif hasattr(e, 'code'):
                    print(e.code, gw_file)
                print("\n\tExiting...")
                return 1
            except Error as e:
                print("\nError:")
                print(e)
                print("\n\tExiting...")
                return 1

            # Unpack healpix file and insert map into db
            print("Unpacking '%s':%s..." % (self.options.gw_id, self.options.healpix_file))
            t1 = time.time()

            orig_prob, orig_distmu, orig_distsigma, orig_distnorm, header_gen = hp.read_map(hpx_path, field=(0, 1, 2, 3), h=True)

            header = dict(header_gen)
            orig_npix = len(orig_prob)
            print("\tOriginal number of pix in '%s': %s" % (self.options.healpix_file, orig_npix))

            sky_area = 4 * 180 ** 2 / np.pi
            area_per_orig_px = sky_area / orig_npix
            print("\tSky Area per original pix in '%s': %s [sq deg]" % (self.options.healpix_file, area_per_orig_px))

            orig_map_nside = hp.npix2nside(orig_npix)
            print("\tOriginal resolution (nside) of '%s': %s\n" % (self.options.healpix_file, orig_map_nside))


            # Check if NSIDE is > 256. If it is, and the --orig_res flag is not specified, rescale map to NSIDE 256
            if orig_map_nside > nside256 and not self.options.orig_res:
                print("Rescaling map to NSIDE = 256")
                rescaled_prob, rescaled_distmu, rescaled_distsigma, rescaled_distnorm = hp.ud_grade([orig_prob,
                                                                                                    orig_distmu,
                                                                                                    orig_distsigma,
                                                                                                    orig_distnorm],
                                                                                                    nside_out=nside256,
                                                                                                    order_in="RING",
                                                                                                    order_out="RING")
                rescaled_npix = len(rescaled_prob)
                print("\tRescaled number of pix in '%s': %s" % (self.options.healpix_file, rescaled_npix))

                area_per_rescaled_px = sky_area / rescaled_npix
                print("\tSky Area per rescaled pix in '%s': %s [sq deg]" % (self.options.healpix_file, area_per_rescaled_px))

                rescaled_map_nside = hp.npix2nside(rescaled_npix)
                print("\tRescaled resolution (nside) of '%s': %s\n" % (self.options.healpix_file, rescaled_map_nside))

                original_pix_per_rescaled_pix = orig_npix / rescaled_npix
                print("Original pix per rescaled pix for %s" % original_pix_per_rescaled_pix)

                print("Renormalizing and initializing rescaled map...")
                rescaled_prob = rescaled_prob * original_pix_per_rescaled_pix

                prob = rescaled_prob
                distmu = rescaled_distmu
                distsigma = rescaled_distsigma
                distnorm = rescaled_distnorm
                map_nside = rescaled_map_nside
            else:
                prob = orig_prob
                distmu = orig_distmu
                distsigma = orig_distsigma
                distnorm = orig_distnorm
                map_nside = orig_map_nside

            # Initialize with 0.0 prob to galaxies
            healpix_map_insert = "INSERT INTO HealpixMap (GWID, URL, Filename, NSIDE, t_0, SubmissionTime, NetProbToGalaxies, RescaledNSIDE) VALUES (%s,%s,%s,%s,%s,%s,%s,%s);"
            healpix_map_data = [(self.options.gw_id, gw_url, self.options.healpix_file, orig_map_nside, t_0,
                                 UTC_submission_time.strftime('%Y-%m-%d %H:%M:%S'), 0.0, map_nside)]

            print("\nInserting %s Healpix Map: %s ..." % (self.options.gw_id, self.options.healpix_file))
            insert_records(healpix_map_insert, healpix_map_data)
            healpix_map_id = query_db([healpix_map_select % (self.options.gw_id, self.options.healpix_file)])[0][0][0]
            print("...Done")

        if not build_pixels:
            print("Skipping pixels...")
            print("\tLoading NSIDE 128 pixels...")
            with open('N128_dict.pkl', 'rb') as handle:
                N128_dict = pickle.load(handle)
                del handle

            t1 = time.time()

            print("Get map pixel")
            map_pixel_select = "SELECT id, HealpixMap_id, Pixel_Index, Prob, Distmu, Distsigma, Distnorm, Mean, Stddev, Norm, N128_SkyPixel_id FROM HealpixPixel WHERE HealpixMap_id = %s;"
            map_pixels = query_db([map_pixel_select % healpix_map_id])[0]

            map_pixel_dict = {}
            for p in map_pixels:
                map_pixel_dict[int(p[2])] = p

            # clean up
            print("freeing `map_pixels`...")
            del map_pixels

            process = psutil.Process(os.getpid())
            mb = process.memory_info().rss / 1e+6
            print("Total mem usage: %0.3f [MB]" % mb)
            print("\n*****************************")

            t2 = time.time()
            print("\n********* start DEBUG ***********")
            print("Pixel select execution time: %s" % (t2 - t1))
            print("********* end DEBUG ***********\n")
        else:
            print("Building pixels...")
            # Process healpix pixels and bulk insert into db. To do this, `LOAD DATA LOCAL INFILE` must be enabled in MySQL.
            # The pixels are unpacked, associated with NSIDE 128 pixels, their LIGO distance distributions are resolved,
            # and finally they are written to the working directory as a CSV that is bulk uploaded to the db. Clean up the file
            # afterward.

            healpix_map_select = "SELECT id, NSIDE FROM HealpixMap WHERE GWID = '%s' and Filename = '%s'"
            healpix_map_id = query_db([healpix_map_select % (self.options.gw_id, self.options.healpix_file)])[0][0][0]

            # Associate these healpix pixels with our N128 Sky Pixels
            t1 = time.time()
            N128_select = "SELECT id, Pixel_Index FROM SkyPixel WHERE NSIDE = 128;"
            N128_result = query_db([N128_select])[0]
            print("Number of NSIDE 128 sky pixels: %s" % len(N128_result))

            N128_dict = {}
            for n128 in N128_result:
                N128_dict[int(n128[1])] = int(n128[0])

            # Clean up N128_result
            print("freeing `N128_result`...")
            del N128_result

            t2 = time.time()
            print("\n********* start DEBUG ***********")
            print("N128 select execution time: %s" % (t2 - t1))
            print("********* end DEBUG ***********\n")

            # Resolve pixel distance distribution. Clean output arrays prior to serialization...
            # MySQL cannot store the np.inf value returned by healpy. Get a value that's close but not too close!!
            t1 = time.time()
            max_double_value = 1.5E+308

            distmu[distmu > max_double_value] = max_double_value
            distsigma[distsigma > max_double_value] = max_double_value
            distnorm[distnorm > max_double_value] = max_double_value

            mean, stddev, norm = distance.parameters_to_moments(distmu, distsigma)

            mean[mean > max_double_value] = max_double_value
            stddev[stddev > max_double_value] = max_double_value
            norm[norm > max_double_value] = max_double_value

            theta, phi = hp.pix2ang(map_nside, range(len(prob)))
            N128_indices = hp.ang2pix(nside128, theta, phi)

            healpix_pixel_data = []
            for i, n128_i in enumerate(N128_indices):
                N128_pixel_id = N128_dict[n128_i]
                healpix_pixel_data.append((healpix_map_id, i, prob[i], distmu[i], distsigma[i], distnorm[i], mean[i],
                                           stddev[i], norm[i], N128_pixel_id))

            # Clean up
            print("freeing `theta`...")
            print("freeing `phi`...")
            print("freeing `N128_indices`...")
            del theta
            del phi
            del N128_indices

            t2 = time.time()
            print("\n********* start DEBUG ***********")
            print("Distance resolution execution time: %s" % (t2 - t1))
            print("********* end DEBUG ***********\n")

            # Create CSV, upload, and clean up CSV
            upload_csv = "%s/%s_bulk_upload.csv" % (formatted_healpix_dir, self.options.gw_id)
            try:
                t1 = time.time()
                print("Creating `%s`" % upload_csv)
                with open(upload_csv, 'w') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    for data in healpix_pixel_data:
                        csvwriter.writerow(data)

                t2 = time.time()
                print("\n********* start DEBUG ***********")
                print("CSV creation execution time: %s" % (t2 - t1))
                print("********* end DEBUG ***********\n")
            except Error as e:
                print("Error in creating CSV:\n")
                print(e)
                print("\nExiting")
                return 1

            t1 = time.time()
            upload_sql = """LOAD DATA LOCAL INFILE '%s' 
                    INTO TABLE HealpixPixel 
                    FIELDS TERMINATED BY ',' 
                    LINES TERMINATED BY '\n' 
                    (HealpixMap_id, Pixel_Index, Prob, Distmu, Distsigma, Distnorm, Mean, Stddev, Norm, N128_SkyPixel_id);"""

            success = bulk_upload(upload_sql % upload_csv)
            if not success:
                print("\nUnsuccessful bulk upload. Exiting...")
                return 1

            t2 = time.time()
            print("\n********* start DEBUG ***********")
            print("CSV upload execution time: %s" % (t2 - t1))

            try:
                print("Removing `%s`..." % upload_csv)
                os.remove(upload_csv)

                # Clean up
                print("freeing `healpix_pixel_data`...")
                del healpix_pixel_data

                print("... Done")
            except Error as e:
                print("Error in file removal")
                print(e)
                print("\nExiting")
                return 1

            t1 = time.time()
            map_pixel_select = "SELECT id, HealpixMap_id, Pixel_Index, Prob, Distmu, Distsigma, Distnorm, Mean, Stddev, Norm, N128_SkyPixel_id FROM HealpixPixel WHERE HealpixMap_id = %s"
            map_pixels = query_db([map_pixel_select % healpix_map_id])[0]
            print(len(map_pixels))
            map_pixel_dict = {}
            for p in map_pixels:
                map_pixel_dict[int(p[2])] = p

            # Clean up
            print("freeing `map_pixels`...")
            del map_pixels

            t2 = time.time()
            print("\n********* start DEBUG ***********")
            print("Pixel select execution time: %s" % (t2 - t1))
            print("********* end DEBUG ***********\n")

            with open('N128_dict.pkl', 'wb') as handle:
                pickle.dump(N128_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if not build_tile_pixel_relation:
            print("Skipping tile-pixel relation...")
        else:
            print("Building tile-pixel relation...")

            select_detector_id = "SELECT id, Name, Deg_width, Deg_height, Deg_radius, Area FROM Detector WHERE Name='%s'"
            tile_select = "SELECT id, Detector_id, FieldName, RA, _Dec, Coord, Poly, EBV, N128_SkyPixel_id FROM StaticTile WHERE Detector_id = %s "
            tile_pixel_upload_csv = "%s/%s_tile_pixel_upload.csv" % (formatted_healpix_dir, self.options.gw_id)

            ##### DO SWOPE ######
            # Get detector -> static tile rows
            swope_detector = query_db([select_detector_id % "SWOPE"])[0][0]
            swope_id = swope_detector[0]
            swope_static_tile_rows = query_db([tile_select % swope_id])[0]

            swope_tiles = []
            for r in swope_static_tile_rows:
                t = Tile(float(r[3]), float(r[4]), float(swope_detector[2]), float(swope_detector[3]), map_nside)
                t.id = int(r[0])
                t.mwe = float(r[7])
                swope_tiles.append(t)

            # clean up
            print("freeing `swope_static_tile_rows`...")
            del swope_static_tile_rows

            t1 = time.time()
            initialized_swope_tiles = None
            with mp.Pool() as pool:
                initialized_swope_tiles = pool.map(initialize_tile, swope_tiles)

            # clean up
            print("freeing `swope_tiles`...")
            del swope_tiles
            t2 = time.time()

            print("\n********* start DEBUG ***********")
            print("Swope Tile initialization execution time: %s" % (t2 - t1))
            print("********* end DEBUG ***********\n")

            # Insert Tile/Healpix pixel relations
            tile_pixel_data = []
            for t in initialized_swope_tiles:
                for p in t.enclosed_pixel_indices:
                    tile_pixel_data.append((t.id, map_pixel_dict[p][0]))

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

            ##### DO THACHER ######
            # Get detector -> static tile rows
            thacher_detector = query_db([select_detector_id % "THACHER"])[0][0]
            thacher_id = thacher_detector[0]
            thacher_static_tile_rows = query_db([tile_select % thacher_id])[0]

            thacher_tiles = []
            for r in thacher_static_tile_rows:
                t = Tile(float(r[3]), float(r[4]), float(thacher_detector[2]), float(thacher_detector[3]), map_nside)
                t.id = int(r[0])
                t.mwe = float(r[7])
                thacher_tiles.append(t)

            # clean up
            print("freeing `thacher_static_tile_rows`...")
            del thacher_static_tile_rows

            t1 = time.time()
            initialized_thacher_tiles = None
            with mp.Pool() as pool:
                initialized_thacher_tiles = pool.map(initialize_tile, thacher_tiles)

            # clean up
            print("freeing `thacher_tiles`...")
            del thacher_tiles
            t2 = time.time()

            print("\n********* start DEBUG ***********")
            print("Thacher Tile initialization execution time: %s" % (t2 - t1))
            print("********* end DEBUG ***********\n")

            # Insert Tile/Healpix pixel relations
            tile_pixel_data = []
            for t in initialized_thacher_tiles:
                for p in t.enclosed_pixel_indices:
                    tile_pixel_data.append((t.id, map_pixel_dict[p][0]))

            # Append to existing CSV, upload, and clean up CSV
            try:
                t1 = time.time()
                print("Appending `%s`" % tile_pixel_upload_csv)
                with open(tile_pixel_upload_csv, 'a') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    for data in tile_pixel_data:
                        csvwriter.writerow(data)

                t2 = time.time()
                print("\n********* start DEBUG ***********")
                print("Tile-Pixel CSV append execution time: %s" % (t2 - t1))
                print("********* end DEBUG ***********\n")
            except Error as e:
                print("Error in creating Tile-Pixel CSV:\n")
                print(e)
                print("\nExiting")
                return 1

            print("Bulk uploading Tile-Pixel...")
            t1 = time.time()
            st_hp_upload_sql = """LOAD DATA LOCAL INFILE '%s' 
                    INTO TABLE StaticTile_HealpixPixel 
                    FIELDS TERMINATED BY ',' 
                    LINES TERMINATED BY '\n' 
                    (StaticTile_id, HealpixPixel_id);"""

            success = bulk_upload(st_hp_upload_sql % tile_pixel_upload_csv)
            if not success:
                print("\nUnsuccessful bulk upload. Exiting...")
                return 1

            t2 = time.time()
            print("\n********* start DEBUG ***********")
            print("Tile-Pixel CSV upload execution time: %s" % (t2 - t1))

            try:
                print("Removing `%s`..." % tile_pixel_upload_csv)
                os.remove(tile_pixel_upload_csv)

                # clean up
                print("freeing `tile_pixel_data`...")
                del tile_pixel_data

                print("... Done")
            except Error as e:
                print("Error in file removal")
                print(e)
                print("\nExiting")
                return 1

        if not build_galaxy_pixel_relation:
            print("Skipping galaxy-pixel relation...")
        else:
            print("Building galaxy-pixel relation...")
            # 1. Select all Galaxies from GD2
            # 2. Resolve all Galaxies to a HealpixPixel index/id
            # 3. Store HealpixPixel_GalaxyDistance2 association
            t1 = time.time()
            galaxy_select = '''
                SELECT 	id, 
                        Galaxy_id, 
                        Distance_id, 
                        PGC, 
                        Name_GWGC, 
                        Name_HyperLEDA, 
                        Name_2MASS, 
                        Name_SDSS_DR12, 
                        RA, 
                        _Dec, 
                        Coord, 
                        dist, 
                        dist_err, 
                        z_dist, 
                        z_dist_err, 
                        z, 
                        B, 
                        B_err, 
                        B_abs, 
                        J, 
                        J_err, 
                        H, 
                        H_err, 
                        K, 
                        K_err, 
                        flag1, 
                        flag2, 
                        flag3,
                        %s # injecting map NSIDE into result tuple
                FROM GalaxyDistance2 
                WHERE B IS NOT NULL AND z_dist IS NOT NULL AND z_dist < 1206.0 
            '''
            galaxy_result = query_db([galaxy_select % map_nside])[0]
            print("Number of Galaxies: %s" % len(galaxy_result))
            t2 = time.time()
            print("\n********* start DEBUG ***********")
            print("Galaxy select execution time: %s" % (t2 - t1))
            print("********* end DEBUG ***********\n")

            t1 = time.time()

            with mp.Pool() as pool:
                galaxy_pixel_relations = pool.map(get_healpix_pixel_id, galaxy_result)

            galaxy_pixel_data = []
            for gp in galaxy_pixel_relations:
                _pixel_index = gp[1]
                _pixel_id = map_pixel_dict[_pixel_index][0]
                _galaxy_id = gp[0]
                galaxy_pixel_data.append((_pixel_id, _galaxy_id))

            t2 = time.time()
            print("\n********* start DEBUG ***********")
            print("Galaxy-pixel creation execution time: %s" % (t2 - t1))
            print("********* end DEBUG ***********\n")

            # Create CSV, upload, and clean up CSV
            galaxy_pixel_upload_csv = "%s/%s_gal_pix_upload.csv" % (formatted_healpix_dir, self.options.gw_id)
            try:
                t1 = time.time()
                print("Creating `%s`" % galaxy_pixel_upload_csv)
                with open(galaxy_pixel_upload_csv, 'w') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    for data in galaxy_pixel_data:
                        csvwriter.writerow(data)

                t2 = time.time()
                print("\n********* start DEBUG ***********")
                print("CSV creation execution time: %s" % (t2 - t1))
                print("********* end DEBUG ***********\n")
            except Error as e:
                print("Error in creating CSV:\n")
                print(e)
                print("\nExiting")
                return 1

            t1 = time.time()
            upload_sql = """LOAD DATA LOCAL INFILE '%s' 
                    INTO TABLE HealpixPixel_GalaxyDistance2 
                    FIELDS TERMINATED BY ',' 
                    LINES TERMINATED BY '\n' 
                    (HealpixPixel_id, GalaxyDistance2_id);"""

            success = bulk_upload(upload_sql % galaxy_pixel_upload_csv)
            if not success:
                print("\nUnsuccessful bulk upload. Exiting...")
                return 1

            t2 = time.time()
            print("\n********* start DEBUG ***********")
            print("CSV upload execution time: %s" % (t2 - t1))

            try:
                print("Removing `%s`..." % galaxy_pixel_upload_csv)
                os.remove(galaxy_pixel_upload_csv)

                # Clean up
                print("freeing `galaxy_pixel_data`...")
                del galaxy_pixel_data

                print("... Done")
            except Error as e:
                print("Error in file removal")
                print(e)
                print("\nExiting")
                return 1

        # clean up
        print("freeing `prob`...")
        print("freeing `distmu`...")
        print("freeing `distsigma`...")
        print("freeing `distnorm`...")
        print("freeing `header_gen`...")
        print("freeing `map_pixel_dict`...")
        del prob
        del distmu
        del distsigma
        del distnorm
        del header_gen
        del map_pixel_dict

        if not build_completeness_func:
            print("Skipping completeness func...")
        else:
            print("Building completeness func...")

            pixel_completeness_upload_csv = "%s/%s_pixel_completeness_upload.csv" % (
            formatted_healpix_dir, self.options.gw_id)

            completeness_select = '''
                SELECT 
                    sp1.id as N128_id, 0.5*(sd.D1+sd.D2) as Dist, sc.Completeness
                FROM SkyPixel sp1 
                JOIN SkyPixel sp2 on sp2.id = sp1.Parent_Pixel_id 
                JOIN SkyPixel sp3 on sp3.id = sp2.Parent_Pixel_id 
                JOIN SkyPixel sp4 on sp4.id = sp3.Parent_Pixel_id 
                JOIN SkyPixel sp5 on sp5.id = sp4.Parent_Pixel_id 
                JOIN SkyPixel sp6 on sp6.id = sp5.Parent_Pixel_id 
                JOIN SkyPixel sp7 on sp7.id = sp6.Parent_Pixel_id 
                JOIN SkyCompleteness sc on sc.SkyPixel_id in (sp1.id, sp2.id, sp3.id, sp4.id, sp5.id, sp6.id, sp7.id) 
                JOIN SkyDistance sd on sd.id = sc.SkyDistance_id 
                WHERE sp1.id in (%s)
                ORDER BY sp1.id, sd.D1 
            '''

            healpix_pixel_completeness_select = '''
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
                FROM HealpixPixel hp
                WHERE HealpixMap_id = %s and hp.N128_SkyPixel_id in (%s)
            '''

            # get clusters of n128 by id
            # get completeness where n128 id in (above cluster)
            # match healpix pixels to n128 ids, and compute completeness and push changes
            # repeat until done with clusters

            # Create list from dict so we can sort
            n128_pixel_id_list = [pixel_id for pixel_index, pixel_id in
                                  N128_dict.items()]  # pixel_index, pixel_id are both INT

            # clean up
            print("freeing `N128_dict`...")
            del N128_dict

            batch_size = 10000
            i = 0
            j = batch_size
            k = len(n128_pixel_id_list)

            process = psutil.Process(os.getpid())
            mb = process.memory_info().rss / 1e+6
            print("\nTotal mem usage: %0.3f [MB]\n" % mb)
            print("\nLength of N128 pixels: %s" % k)
            print("Batch size: %s" % batch_size)
            print("Starting loop...")

            number_of_iters = k // batch_size
            if k % batch_size > 0:
                number_of_iters += 1

            iter_num = 1
            while j < k:

                t1 = time.time()

                print("%s:%s" % (i, j))
                batch = n128_pixel_id_list[i:j]
                batch_id_string = ','.join(map(str, batch))

                completeness_result = query_db([completeness_select % batch_id_string])[0]
                print("Number of completeness records: %s" % len(completeness_result))
                healpix_pixel_completeness_result = \
                query_db([healpix_pixel_completeness_select % (healpix_map_id, batch_id_string)])[0]
                print("Number of healpix pixel records: %s" % len(healpix_pixel_completeness_result))

                i = j
                j += batch_size

                # DO WORK HERE
                completeness_values_dict = {}
                for c in completeness_result:
                    N128_id = int(c[0])

                    if N128_id not in completeness_values_dict:
                        completeness_values_dict[N128_id] = [[0.0], [1.0]]

                    comp = float(c[2]) if float(c[2]) <= 1.0 else 1.0
                    completeness_values_dict[N128_id][0].append(float(c[1]))
                    completeness_values_dict[N128_id][1].append(comp)

                # Add point at infinity
                for key, val in completeness_values_dict.items():
                    completeness_values_dict[key][0].append(np.inf)
                    completeness_values_dict[key][1].append(0.0)

                # clean up
                print("freeing `completeness_result`...")
                del completeness_result

                pixel_completeness_records = []
                for p in healpix_pixel_completeness_result:
                    pix_id = p[0]
                    pix_prob = float(p[3])
                    pix_mean_dist = float(p[7])
                    N128_id = int(p[10])

                    pix_completeness = np.interp(pix_mean_dist,
                                                 completeness_values_dict[N128_id][0],
                                                 completeness_values_dict[N128_id][1])

                    renorm2dprob = pix_prob * (1.0 - pix_completeness)
                    pixel_completeness_records.append((pix_id, pix_completeness, renorm2dprob, -1.0, healpix_map_id))

                # Append to data to CSV
                try:
                    t1 = time.time()

                    print("Appending `%s`" % pixel_completeness_upload_csv)
                    with open(pixel_completeness_upload_csv, 'a') as csvfile:
                        csvwriter = csv.writer(csvfile)
                        for data in pixel_completeness_records:
                            csvwriter.writerow(data)

                    t2 = time.time()
                    print("\n********* start DEBUG ***********")
                    print("Pixel Completeness CSV append execution time: %s" % (t2 - t1))
                    print("********* end DEBUG ***********\n")
                except Error as e:
                    print("Error in creating Pixel Completeness CSV:\n")
                    print(e)
                    print("\nExiting")
                    return 1

                # batch_insert(pixel_update, pixel_update_records)

                # clean up
                print("freeing `completeness_values_dict`...")
                print("freeing `pixel_completeness_records`...")
                print("freeing `healpix_pixel_completeness_result`...")
                del completeness_values_dict
                del pixel_completeness_records
                del healpix_pixel_completeness_result

                t2 = time.time()
                print("\n********* start DEBUG ***********")
                print("SELECT %s/%s complete - execution time: %s" % (iter_num, number_of_iters, (t2 - t1)))
                print("********* end DEBUG ***********\n")

                iter_num += 1

                process = psutil.Process(os.getpid())
                mb = process.memory_info().rss / 1e+6
                print("\nTotal mem usage: %0.3f [MB]\n" % mb)

            print("Out of loop...")

            # FINISH WORK HERE
            t1 = time.time()

            print("\n%s:%s" % (i, k))
            batch = n128_pixel_id_list[i:k]
            batch_id_string = ','.join(map(str, batch))

            completeness_result = query_db([completeness_select % batch_id_string])[0]
            healpix_pixel_completeness_result = \
            query_db([healpix_pixel_completeness_select % (healpix_map_id, batch_id_string)])[0]

            completeness_values_dict = {}
            for c in completeness_result:
                N128_id = int(c[0])

                if N128_id not in completeness_values_dict:
                    completeness_values_dict[N128_id] = [[0.0], [1.0]]

                comp = float(c[2]) if float(c[2]) <= 1.0 else 1.0
                completeness_values_dict[N128_id][0].append(float(c[1]))
                completeness_values_dict[N128_id][1].append(comp)

            # Add point at infinity
            for key, val in completeness_values_dict.items():
                completeness_values_dict[key][0].append(np.inf)
                completeness_values_dict[key][1].append(0.0)

            # clean up
            print("freeing `completeness_result`...")
            del completeness_result

            pixel_completeness_records = []
            for p in healpix_pixel_completeness_result:
                pix_id = p[0]
                pix_prob = float(p[3])
                pix_mean_dist = float(p[7])
                N128_id = int(p[10])

                pix_completeness = np.interp(pix_mean_dist,
                                             completeness_values_dict[N128_id][0],
                                             completeness_values_dict[N128_id][1])

                renorm2dprob = pix_prob * (1.0 - pix_completeness)
                pixel_completeness_records.append((pix_id, pix_completeness, renorm2dprob, -1.0, healpix_map_id))

            t2 = time.time()
            print("\n********* start DEBUG ***********")
            print("SELECT %s/%s complete - execution time: %s" % (iter_num, number_of_iters, (t2 - t1)))
            print("********* end DEBUG ***********\n")

            # Append to data to CSV
            try:
                t1 = time.time()
                print("Appending `%s`" % pixel_completeness_upload_csv)
                with open(pixel_completeness_upload_csv, 'a') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    for data in pixel_completeness_records:
                        csvwriter.writerow(data)

                t2 = time.time()
                print("\n********* start DEBUG ***********")
                print("Pixel Completeness CSV append execution time: %s" % (t2 - t1))
                print("********* end DEBUG ***********\n")
            except Error as e:
                print("Error in creating Pixel Completeness CSV:\n")
                print(e)
                print("\nExiting")
                return 1

            t1 = time.time()
            upload_sql = """LOAD DATA LOCAL INFILE '%s' 
                    INTO TABLE HealpixPixel_Completeness 
                    FIELDS TERMINATED BY ',' 
                    LINES TERMINATED BY '\n' 
                    (HealpixPixel_id, PixelCompleteness, Renorm2DProb, NetPixelProb, HealpixMap_id);"""

            success = bulk_upload(upload_sql % pixel_completeness_upload_csv)
            if not success:
                print("\nUnsuccessful bulk upload. Exiting...")
                return 1

            t2 = time.time()
            print("\n********* start DEBUG ***********")
            print("CSV upload execution time: %s" % (t2 - t1))

            try:
                print("Removing `%s`..." % pixel_completeness_upload_csv)
                os.remove(pixel_completeness_upload_csv)

                # clean up
                print("freeing `completeness_values_dict`...")
                print("freeing `pixel_completeness_records`...")
                print("freeing `healpix_pixel_completeness_result`...")
                del completeness_values_dict
                del pixel_completeness_records
                del healpix_pixel_completeness_result

                print("... Done")
            except Error as e:
                print("Error in file removal")
                print(e)
                print("\nExiting")
                return 1

        if not build_galaxy_weights:
            print("Skipping galaxy weights ...")
        else:
            print("Building galaxy weights ...")

            t1 = time.time()
            # Set & Retrieve NetProbToGalaxies
            healpix_map_NetProbToGalaxies_update = '''
                UPDATE HealpixMap hm
                SET NetProbToGalaxies = (SELECT 1-SUM(hpc.Renorm2DProb) 
                                         FROM HealpixPixel_Completeness hpc 
                                         WHERE HealpixMap_id = %s
                                        )
                WHERE hm.id = %s;
            '''

            query_db([healpix_map_NetProbToGalaxies_update % (healpix_map_id, healpix_map_id)], commit=True)

            select_NetProbToGalaxies = "SELECT NetProbToGalaxies FROM HealpixMap WHERE id = %s"
            select_NetProbToGalaxies_result = query_db([select_NetProbToGalaxies % healpix_map_id])[0]
            net_prob_to_galaxies = select_NetProbToGalaxies_result[0][0]
            print("Net probability to galaxies: %0.5f" % net_prob_to_galaxies)

            t2 = time.time()
            print("\n********* start DEBUG ***********")
            print("healpix_map_NetProbToGalaxies execution time: %s" % (t2 - t1))
            print("********* end DEBUG ***********\n")

            tt1 = time.time()
            # Get galaxy luminosity normalization
            lum_norm_select = '''
                SELECT SUM(POW(gd2.z_dist, 2)*POW(10.0, -0.4*gd2.B)) FROM GalaxyDistance2 gd2 
                WHERE gd2.id IN (SELECT DISTINCT _gd2.id FROM GalaxyDistance2 _gd2 
                                 JOIN HealpixPixel_GalaxyDistance2 hp_gd2 on hp_gd2.GalaxyDistance2_id = _gd2.id
                                 JOIN HealpixPixel hp on hp.id = hp_gd2.HealpixPixel_id WHERE hp.HealpixMap_id = %s)
            '''
            lum_norm = query_db([lum_norm_select % healpix_map_id])[0][0][0]

            # Compute luminosity weight and pre-compute what we can on z_prob...
            precompute_select = '''
                SELECT 
                    hp_gd2.id as HealpixPixel_GalaxyDistance2_id, 
                    hp.id, 
                    hp.Pixel_Index, 
                    hp.Prob, 
                    gd2.z_dist, 
                    gd2.z_dist_err, 
                    hp.Mean, 
                    hp.Stddev, 
                    POW(gd2.z_dist, 2)*POW(10.0, -0.4*(gd2.B))/%s as Bweight, 
                    ABS(gd2.z_dist - hp.Mean)/SQRT(POW(hp.Stddev, 2) + POW(gd2.z_dist_err, 2)) as SigmaTotal 
                FROM GalaxyDistance2 gd2 
                JOIN HealpixPixel_GalaxyDistance2 hp_gd2 on hp_gd2.GalaxyDistance2_id = gd2.id 
                JOIN HealpixPixel hp on hp.id = hp_gd2.HealpixPixel_id 
                WHERE hp.HealpixMap_id = %s and gd2.id IN (SELECT DISTINCT _gd2.id FROM GalaxyDistance2 _gd2 
                                JOIN HealpixPixel_GalaxyDistance2 hp__gd2 on hp__gd2.GalaxyDistance2_id = _gd2.id
                                JOIN HealpixPixel _hp on _hp.id = hp__gd2.HealpixPixel_id WHERE _hp.HealpixMap_id = %s) 
            '''

            precompute_result = query_db([precompute_select % (lum_norm, healpix_map_id, healpix_map_id)])[0]
            t2 = time.time()
            print("\n********* start DEBUG ***********")
            print("Precompute Select execution time: %s" % (t2 - t1))
            print("********* end DEBUG ***********\n")

            galaxy_attributes = []
            four_d_prob_norm = 0.0
            for r in precompute_result:
                hp_gd2_id = int(r[0])
                two_d_prob = float(r[3])
                lum_prob = float(r[8])
                sigma_total = float(r[9])
                z_prob = 1.0 - erf(sigma_total)

                four_d_prob = z_prob * two_d_prob * lum_prob
                four_d_prob_norm += four_d_prob

                galaxy_attributes.append([hp_gd2_id, lum_prob, z_prob, two_d_prob, four_d_prob])

            # clean up
            print("freeing `precompute_result`...")
            del precompute_result

            galaxy_attribute_data = []
            for g in galaxy_attributes:
                norm_4d_weight = g[4] / four_d_prob_norm
                galaxy_attribute_data.append(
                    (g[0], g[1], g[2], g[3], norm_4d_weight, norm_4d_weight * net_prob_to_galaxies))

            # clean up
            print("freeing `galaxy_attributes`...")
            del galaxy_attributes

            # Create CSV, upload, and clean up CSV
            galaxy_attributes_upload_csv = "%s/%s_galaxy_attributes_upload.csv" % (
            formatted_healpix_dir, self.options.gw_id)
            try:
                t1 = time.time()
                print("Creating `%s`" % galaxy_attributes_upload_csv)
                with open(galaxy_attributes_upload_csv, 'w') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    for data in galaxy_attribute_data:
                        csvwriter.writerow(data)

                t2 = time.time()
                print("\n********* start DEBUG ***********")
                print("CSV creation execution time: %s" % (t2 - t1))
                print("********* end DEBUG ***********\n")
            except Error as e:
                print("Error in creating CSV:\n")
                print(e)
                print("\nExiting")
                return 1

            t1 = time.time()
            upload_sql = """LOAD DATA LOCAL INFILE '%s' 
                    INTO TABLE HealpixPixel_GalaxyDistance2_Weight 
                    FIELDS TERMINATED BY ',' 
                    LINES TERMINATED BY '\n' 
                    (HealpixPixel_GalaxyDistance2_id, LumWeight, zWeight, Prob2DWeight, Norm4DWeight, GalaxyProb);"""

            success = bulk_upload(upload_sql % galaxy_attributes_upload_csv)
            if not success:
                print("\nUnsuccessful bulk upload. Exiting...")
                return 1

            t2 = time.time()
            print("\n********* start DEBUG ***********")
            print("CSV upload execution time: %s" % (t2 - t1))

            try:
                print("Removing `%s`..." % galaxy_attributes_upload_csv)
                os.remove(galaxy_attributes_upload_csv)

                # Clean up
                print("freeing `galaxy_attribute_data`...")
                del galaxy_attribute_data

                print("... Done")
            except Error as e:
                print("Error in file removal")
                print(e)
                print("\nExiting")
                return 1

            tt2 = time.time()
            print("\n********* start DEBUG ***********")
            print("HealpixPixel_GalaxyDistance2 Galaxy Attribute execution time: %s" % (tt2 - tt1))
            print("********* end DEBUG ***********\n")

            # update the healpix pixel net prob column...
            print("Updating healpix pixel net prob...")
            t1 = time.time()
            healpix_pixel_net_prob_update = '''
                UPDATE HealpixPixel_Completeness hpc1 
                JOIN 
                ( 
                    SELECT 
                        hpc2.id, 
                        (hpc2.Renorm2DProb + IFNULL(SUM(hp_gd2_w.GalaxyProb),0.0)) as NetPixelProb 
                    FROM HealpixPixel_Completeness hpc2 
                    LEFT JOIN HealpixPixel_GalaxyDistance2 hp_gd2 on hp_gd2.HealpixPixel_id = hpc2.HealpixPixel_id 
                    LEFT JOIN HealpixPixel_GalaxyDistance2_Weight hp_gd2_w on hp_gd2_w.HealpixPixel_GalaxyDistance2_id = hp_gd2.id 
                    WHERE hpc2.HealpixMap_id = %s
                    GROUP BY hpc2.id 
                ) temp on hpc1.id = temp.id 
                SET hpc1.NetPixelProb = temp.NetPixelProb 
                WHERE hpc1.HealpixMap_id = %s;
            '''
            query_db([healpix_pixel_net_prob_update % (healpix_map_id, healpix_map_id)], commit=True)
            t2 = time.time()
            print("\n********* start DEBUG ***********")
            print("Update HealpixPixel NetProb execution time: %s" % (t2 - t1))
            print("********* end DEBUG ***********\n")


if __name__ == "__main__":
    useagestring = """python Generate_Tiles.py [options]

Example with healpix_dir defaulted to 'Events/<gwid>':
python LoadMap.py --gw_id <gwid> --healpix_file <filename>

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
    print("Teglon `LoadMap` execution time: %s" % duration)
    print("********* end DEBUG ***********\n")
