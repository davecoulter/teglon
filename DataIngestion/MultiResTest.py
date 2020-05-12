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

        return (parser)

    def main(self):

        infile = "gw190425_skymap.fits"
        skymap = Table.read(infile)

        uniq = skymap['UNIQ']
        prob_density = skymap['PROBDENSITY']*(np.pi/180)**2 # prob/sq degree
        distmu = skymap['DISTMU']
        distsigma = skymap['DISTSIGMA']
        distnorm = skymap['DISTNORM']

        level, ipix_nested = ah.uniq_to_level_ipix(uniq)
        nside = ah.level_to_nside(level)
        ipix_ring = hp.nest2ring(nside, ipix_nested)

        pixels = []
        probs_per_nside = {}
        for i, pixel_index in enumerate(ipix_ring):

            pixel_prob = prob_density[i]*hp.nside2pixarea(nside[i],degrees=True)
            if not nside[i] in probs_per_nside:
                probs_per_nside[nside[i]] = pixel_prob
            else:
                probs_per_nside[nside[i]] += pixel_prob

            pixels.append(Pixel_Element(pixel_index, nside[i], pixel_prob))

        total_prob = 0
        for k,v in probs_per_nside.items():
            print("NSIDE %s: %s" % (k, v))
            total_prob += v
        print("Total prob: %s" % total_prob)
        print("# Pixel elements: %s" % len(pixels))



        # healpix_map_select = "SELECT id, NSIDE FROM HealpixMap WHERE GWID = '%s' and Filename = '%s'"

        # # Initialize with 0.0 prob to galaxies
        # healpix_map_insert = "INSERT INTO HealpixMap (GWID, URL, Filename, NSIDE, t_0, SubmissionTime, NetProbToGalaxies, RescaledNSIDE) VALUES (%s,%s,%s,%s,%s,%s,%s,%s);"
        # healpix_map_data = [("S190930s", "https://gracedb.ligo.org/superevents/S190930s/view/", infile, -1, 1253885758.235347,
        #                      "2019-09-30 13:36:04", 0.0, -1)]
        #
        # print("\nInserting %s Healpix Map: %s ..." % ("S190930s", infile))
        # insert_records(healpix_map_insert, healpix_map_data)

        # healpix_map_id = query_db([healpix_map_select % ("S190930s", infile)])[0][0][0]
        # print("Healpix Map Id: %s" % healpix_map_id)
        # print("...Done")
        #
        # N128_select = "SELECT id, Pixel_Index FROM SkyPixel WHERE NSIDE = 128;"
        # N128_result = query_db([N128_select])[0]
        # print("Number of NSIDE 128 sky pixels: %s" % len(N128_result))
        #
        # N128_dict = {}
        # for n128 in N128_result:
        #     N128_dict[int(n128[1])] = int(n128[0])
        #
        # # Clean up N128_result
        # print("freeing `N128_result`...")
        # del N128_result
        #
        # max_double_value = 1.5E+308
        #
        # distmu[distmu > max_double_value] = max_double_value
        # distsigma[distsigma > max_double_value] = max_double_value
        # distnorm[distnorm > max_double_value] = max_double_value
        #
        # mean, stddev, norm = distance.parameters_to_moments(distmu, distsigma)
        #
        # mean[mean > max_double_value] = max_double_value
        # stddev[stddev > max_double_value] = max_double_value
        # norm[norm > max_double_value] = max_double_value




        # Plot!!
        fig = plt.figure(figsize=(12, 12), dpi=1000)
        ax = fig.add_subplot(111)
        m = Basemap(projection='moll', lon_0=180.0)

        pixel_probs = [p.prob for p in pixels]
        # min_prob = np.min(pixel_probs)
        min_prob = 1e-7
        max_prob = np.max(pixel_probs)
        norm = colors.LogNorm(min_prob, max_prob)
        for p in pixels:
            p.plot(m, ax, facecolor=plt.cm.inferno(norm(p.prob)), edgecolor='None', linewidth=0.1)
            # p.plot(m, ax, facecolor="None", edgecolor='black', linewidth=0.1)

        meridians = np.arange(0., 360., 60.)
        m.drawparallels(np.arange(-90., 91., 30.), fontsize=14, labels=[True, True, False, False], dashes=[2, 2],
                        linewidth=0.5)
        m.drawmeridians(meridians, labels=[False, False, False, False], dashes=[2, 2], linewidth=0.5)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.inferno)
        sm.set_array([])  # can be an empty list

        # tks = np.linspace(min_prob, max_prob, 6)
        tks = np.logspace(np.log10(min_prob), np.log10(max_prob), 6)
        tks_strings = []

        for t in tks:
            tks_strings.append('%0.2f' % (t * 100))

        cb = fig.colorbar(sm, ax=ax, ticks=tks, orientation='vertical', fraction=0.04875, pad=0.02,
                          alpha=0.80)  # 0.08951
        cb.ax.set_yticklabels(tks_strings, fontsize=16)
        cb.set_label("2D Pixel Probability", fontsize=20, labelpad=9.0)


        ax.invert_xaxis()

        plt.ylabel(r'$\mathrm{Declination}$', fontsize=16, labelpad=36)
        plt.xlabel(r'$\mathrm{Right\;Ascension}$', fontsize=16, labelpad=30)

        output_path = "multi_res_test.png"
        fig.savefig(output_path, bbox_inches='tight')
        plt.close('all')
        print("... Done.")






if __name__ == "__main__":
    useagestring = """python MultiResTest.py
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
    print("Teglon `MultiResTest` execution time: %s" % duration)
    print("********* end DEBUG ***********\n")


