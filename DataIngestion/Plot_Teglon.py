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

        parser.add_option('--gw_id', default="", type="str",
                          help='LIGO superevent name, e.g. `S190425z` ')

        parser.add_option('--healpix_dir', default='../Events/{GWID}', type="str",
                          help='Directory for where to look for the healpix file.')

        parser.add_option('--healpix_file', default="", type="str", help='Healpix filename.')

        parser.add_option('--tele', default="s", type="str",
                          help='''Telescope abbreviation for the telescope to extract files for. Default: `s`. 
                          Available: (s:Swope, t:Thacher)
                          ''')

        parser.add_option('--band', default="r", type="str",
                          help='Telescope abbreviation for the telescope to extract files for. Default: `r`. Available: (g, r, i)')

        parser.add_option('--extinct', default="0.5", type="float",
                          help='Extinction in mags in the specified band to be less than. Default: 0.5 mag. Must be > 0.0')

        parser.add_option('--tile_file', default="{FILENAME}", type="str", help='Filename of tiles to plot.')

        parser.add_option('--galaxies_file', default="{FILENAME}", type="str", help='Optional: Filename of galaxies to plot.')

        parser.add_option('--get_tiles_from_db', action="store_true", default=False,
                          help='''Ignore tile file and get all observed tiles for map from database''')

        parser.add_option('--prob_type', default="2D", type="str",
                          help='''Probability type to consider. Default `2D`. Available: (4D, 2D)''')

        parser.add_option('--cum_prob_outer', default="0.9", type="float",
                          help='''Cumulative prob to cover in tiles. Default 0.9. Must be > 0.2 and < 0.95 
                          and > `cum_prob_inner`''')

        parser.add_option('--cum_prob_inner', default="0.5", type="float",
                          help='''Cumulative prob to cover in tiles. Default 0.5. Must be > 0.2 and < 0.95 
                          and < `cum_prob_inner`''')

        return (parser)

    def main(self):

        detector_mapping = {
            "s": "SWOPE",
            "t": "THACHER",
            "n": "NICKEL",
            "k": "KAIT"
        }

        band_mapping = {
            "g": "SDSS g",
            "r": "SDSS r",
            "i": "SDSS i",
            "Clear": "Clear"
        }

        # Valid prob types
        _4D = "4D"
        _2D = "2D"

        is_error = False

        # Parameter checks
        if self.options.gw_id == "":
            is_error = True
            print("GWID is required.")

        formatted_healpix_dir = self.options.healpix_dir
        if "{GWID}" in formatted_healpix_dir:
            formatted_healpix_dir = formatted_healpix_dir.replace("{GWID}", self.options.gw_id)

        hpx_path = "%s/%s" % (formatted_healpix_dir, self.options.healpix_file)
        tile_file_path = "%s/%s" % (formatted_healpix_dir, self.options.tile_file)
        galaxies_file_path = "%s/%s" % (formatted_healpix_dir, self.options.galaxies_file)

        if self.options.healpix_file == "":
            is_error = True
            print("You must specify which healpix file to process.")

        if self.options.band not in band_mapping:
            is_error = True
            print("Invalid band selection. Available bands: %s" % band_mapping.keys())

        if self.options.tele not in detector_mapping:
            is_error = True
            print("Invalid telescope selection. Available telescopes: %s" % detector_mapping.keys())

        if not self.options.get_tiles_from_db:
            if not os.path.exists(tile_file_path):
                is_error = True
                print("You must specify which tile file to plot.")

        plotGalaxies = False
        if os.path.exists(galaxies_file_path):
            plotGalaxies = True

        if self.options.extinct <= 0.0:
            is_error = True
            print("Extinction must be a valid float > 0.0")

        if not (self.options.prob_type == _4D or self.options.prob_type == _2D):
            is_error = True
            print("Prob type must either be `4D` or `2D`")

        if self.options.cum_prob_outer > 0.95 or \
                self.options.cum_prob_outer < 0.20 or \
                self.options.cum_prob_outer < self.options.cum_prob_inner:
            is_error = True
            print("Cum prob outer must be between 0.2 and 0.95 and > Cum prob inner")

        if self.options.cum_prob_inner > 0.95 or \
                self.options.cum_prob_inner < 0.20 or \
                self.options.cum_prob_inner > self.options.cum_prob_outer:
            is_error = True
            print("Cum prob inner must be between 0.2 and 0.95 and < Cum prob outer")

        if is_error:
            print("Exiting...")
            return 1

        # Get Map ID
        healpix_map_select = "SELECT id, RescaledNSIDE FROM HealpixMap WHERE GWID = '%s' and Filename = '%s'"
        healpix_map_result = query_db([healpix_map_select % (self.options.gw_id, self.options.healpix_file)])[0][0]
        healpix_map_id = int(healpix_map_result[0])
        healpix_map_nside = int(healpix_map_result[1])

        band_name = band_mapping[self.options.band]
        band_select = "SELECT id, Name, F99_Coefficient FROM Band WHERE `Name`='%s'"
        band_result = query_db([band_select % band_name])[0][0]
        band_id = band_result[0]
        band_F99 = float(band_result[2])

        telescope_name = detector_mapping[self.options.tele]
        detector_select_by_name = "SELECT id, Name, Deg_width, Deg_height, Deg_radius, Area, MinDec, MaxDec FROM Detector WHERE Name='%s'"
        detector_result = query_db([detector_select_by_name % telescope_name])[0][0]
        detector_id = int(detector_result[0])
        detector = Detector(detector_result[1], float(detector_result[2]), float(detector_result[3]),
                            detector_id=detector_id)

        galaxies_to_plot = []
        gal_ra = []
        gal_dec = []
        if plotGalaxies:
            with open('%s' % galaxies_file_path, 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
                # Skip Header
                next(csvreader)

                for row in csvreader:
                    gal_ra.append(row[1])
                    gal_dec.append(row[2])

                galaxies_to_plot = coord.SkyCoord(gal_ra, gal_dec, unit=(u.hour, u.deg))

        tiles_to_plot = []
        if not self.options.get_tiles_from_db:
            # Load tile_file
            with open('%s' % tile_file_path,'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
                # Skip Header
                next(csvreader)

                for row in csvreader:
                    name = row[0]
                    c = coord.SkyCoord(row[1], row[2], unit=(u.hour, u.deg))
                    t = Tile(c.ra.degree, c.dec.degree, detector.deg_width, detector.deg_height, healpix_map_nside)
                    t.field_name = name
                    t.net_prob = float(row[6])
                    tiles_to_plot.append(t)
        else:
            select_tiles_per_map = '''
                SELECT 
                    ot.FieldName, 
                    ot.RA, 
                    ot._Dec, 
                    ot.MJD, 
                    ot.Exp_Time, 
                    ot.Mag_Lim, 
                    d.`Name` as DetectorName, 
                    d.Deg_width, 
                    d.Deg_height
                FROM ObservedTile ot
                JOIN Detector d on d.id = ot.Detector_id
                WHERE ot.HealpixMap_id = %s;
            '''

            map_tiles = query_db([select_tiles_per_map % healpix_map_id])[0]
            print("Map tiles: %s" % len(map_tiles))
            for mt in map_tiles:
                c = coord.SkyCoord(mt[1], mt[2], unit=(u.deg, u.deg))
                t = Tile(c.ra.degree, c.dec.degree, float(mt[7]), float(mt[8]), healpix_map_nside)
                t.field_name = mt[6]
                tiles_to_plot.append(t)

        tile_colors = {
            'KAIT': "seagreen",
            'NICKEL': "mediumpurple",
            'SWOPE': "red",
            'THACHER': "darkorange"
        }

        select_2D_pix = '''
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
                running_prob.cum_prob <= %s
        '''

        select_4D_pix = '''
            SELECT 
                running_prob.id, 
                running_prob.HealpixMap_id, 
                running_prob.Pixel_Index, 
                running_prob.Prob, 
                running_prob.NetPixelProb, 
                running_prob.Distmu, 
                running_prob.Distsigma, 
                running_prob.Mean, 
                running_prob.Stddev, 
                running_prob.Norm, 
                running_prob.N128_SkyPixel_id, 
                running_prob.cum_net_pixel_prob 
            FROM 
                (SELECT 
                    hp_prob.id, 
                    hp_prob.HealpixMap_id, 
                    hp_prob.Pixel_Index,
                    hp_prob.Prob, 
                    hp_prob.NetPixelProb, 
                    hp_prob.Distmu, 
                    hp_prob.Distsigma, 
                    hp_prob.Mean, 
                    hp_prob.Stddev, 
                    hp_prob.Norm, 
                    hp_prob.N128_SkyPixel_id, 
                    SUM(hp_prob.NetPixelProb) OVER(ORDER BY hp_prob.NetPixelProb DESC) AS cum_net_pixel_prob 
                FROM 
                    (SELECT 
                        hp.id, 
                        hp.HealpixMap_id, 
                        hp.Pixel_Index,
                        hp.Prob, 
                        hpc.NetPixelProb, 
                        hp.Distmu, 
                        hp.Distsigma, 
                        hp.Mean, 
                        hp.Stddev, 
                        hp.Norm, 
                        hp.N128_SkyPixel_id 
                    FROM HealpixPixel hp 
                    JOIN HealpixPixel_Completeness hpc on hpc.HealpixPixel_id = hp.id
                    WHERE hp.HealpixMap_id = %s 
                    ORDER BY
                        hpc.NetPixelProb DESC) hp_prob 
                    GROUP BY
                        hp_prob.id, 
                        hp_prob.HealpixMap_id, 
                        hp_prob.Pixel_Index,
                        hp_prob.Prob, 
                        hp_prob.NetPixelProb, 
                        hp_prob.Distmu, 
                        hp_prob.Distsigma, 
                        hp_prob.Mean, 
                        hp_prob.Stddev, 
                        hp_prob.Norm, 
                        hp_prob.N128_SkyPixel_id 
                    ) running_prob 
            WHERE 
                running_prob.cum_net_pixel_prob <= %s 
        '''

        select_mwe_pix = '''
            SELECT sp.Pixel_Index
            FROM SkyPixel sp
            WHERE sp.id IN
            (
                SELECT sp.Parent_Pixel_id
                FROM SkyPixel sp
                WHERE sp.id IN
                (
                    SELECT sp.Parent_Pixel_id
                    FROM SkyPixel sp
                    JOIN SkyPixel_EBV sp_ebv ON sp_ebv.N128_SkyPixel_id = sp.id
                    WHERE sp_ebv.EBV*%s > %s and NSIDE = 128
                ) and NSIDE = 64
            ) and NSIDE = 32
        '''

        print("Selecting map pixels...")
        pixels_to_select = ""
        if self.options.prob_type == _2D:
            pixels_to_select = select_2D_pix % (healpix_map_id, self.options.cum_prob_outer)
        else:
            pixels_to_select = select_4D_pix % (healpix_map_id, self.options.cum_prob_outer)

        map_pix_result = query_db([pixels_to_select])[0]

        mwe_pix_result = query_db([select_mwe_pix % (band_F99, self.options.extinct)])[0]
        print("...done")

        print("Building pixel elements...")
        map_pix = [Pixel_Element(int(mp[2]), healpix_map_nside, float(mp[3]), pixel_id=int(mp[0]))
                   for mp in map_pix_result]
        map_pix_sorted = sorted(map_pix, key=lambda p: p.prob, reverse=True)

        mwe_pix = [Pixel_Element(int(mp[0]), 32, 0.0) for mp in mwe_pix_result]
        print("...done")

        cutoff_inner = self.options.cum_prob_inner
        cutoff_outer = self.options.cum_prob_outer
        index_inner = 0
        index_outer = 0

        print("Find index for inner contour...")
        cum_prob = 0.0
        for i in range(len(map_pix_sorted)):
            cum_prob += map_pix_sorted[i].prob
            index_inner = i
            if cum_prob >= cutoff_inner:
                break
        print("... %s" % index_inner)

        print("Find index for outer contour...")
        cum_prob = 0.0
        for i in range(len(map_pix_sorted)):
            cum_prob += map_pix_sorted[i].prob
            index_outer = i
            if cum_prob >= cutoff_outer:
                break
        print("... %s" % index_outer)

        print("Build multipolygons...")
        net_inner_polygon = []
        for p in map_pix_sorted[0:index_inner]:
            net_inner_polygon += p.query_polygon
        joined_inner_poly = unary_union(net_inner_polygon)

        # Fix any seams
        eps = 0.00001
        merged_inner_poly = []
        smoothed_inner_poly = joined_inner_poly.buffer(eps, 1, join_style=JOIN_STYLE.mitre).buffer(-eps, 1, join_style=JOIN_STYLE.mitre)

        try:
            test_iter = iter(smoothed_inner_poly)
            merged_inner_poly = smoothed_inner_poly
        except TypeError as te:
            merged_inner_poly.append(smoothed_inner_poly)

        print("Number of sub-polygons in `merged_inner_poly`: %s" % len(merged_inner_poly))
        sql_inner_poly = SQL_Polygon(merged_inner_poly, detector)

        net_outer_polygon = []
        for p in map_pix_sorted[0:index_outer]:
            net_outer_polygon += p.query_polygon
        joined_outer_poly = unary_union(net_outer_polygon)

        # Fix any seams
        merged_outer_poly = []
        smoothed_outer_poly = joined_outer_poly.buffer(eps, 1, join_style=JOIN_STYLE.mitre).buffer(-eps, 1, join_style=JOIN_STYLE.mitre)

        try:
            test_iter = iter(smoothed_outer_poly)
            merged_outer_poly = smoothed_outer_poly
        except TypeError as te:
            merged_outer_poly.append(smoothed_outer_poly)

        print("Number of sub-polygons in `merged_outer_poly`: %s" % len(merged_outer_poly))
        sql_outer_poly = SQL_Polygon(merged_outer_poly, detector)
        print("... done.")

        # Plot!!
        fig = plt.figure(figsize=(10, 10), dpi=800)
        ax = fig.add_subplot(111)
        m = Basemap(projection='moll', lon_0=180.0)

        sql_inner_poly.plot(m, ax, edgecolor='black', linewidth=0.5, facecolor='None')
        sql_outer_poly.plot(m, ax, edgecolor='gray', linewidth=0.5, facecolor='None')

        for p in mwe_pix:
            p.plot(m, ax, edgecolor='None', linewidth=0.5, facecolor='cornflowerblue', alpha=0.5)

        print("Plotting (%s) Tiles..." % len(tiles_to_plot))
        if not self.options.get_tiles_from_db:
            for i, t in enumerate(tiles_to_plot):
                t.plot(m, ax, edgecolor='r', facecolor='None', linewidth=0.25, alpha=1.0,
                       zorder=9900)
        else:
            for i, t in enumerate(tiles_to_plot):
                t.plot(m, ax, edgecolor=tile_colors[t.field_name], facecolor='None', linewidth=0.25, alpha=1.0,
                       zorder=9900)

        print("Plotting (%s) Galaxies..." % len(galaxies_to_plot))
        if plotGalaxies:
            for g in galaxies_to_plot:
                x, y = m(g.ra.degree, g.dec.degree)
                m.plot(x,y, 'ko', markersize=0.2, linewidth=0.25, alpha=0.3, zorder=9900)

        # # # -------------- Use this for mollweide projections --------------
        meridians = np.arange(0., 360., 60.)
        m.drawparallels(np.arange(-90., 91., 30.), fontsize=14, labels=[True, True, False, False], dashes=[2, 2],
                        linewidth=0.5) # , xoffset=2500000
        m.drawmeridians(meridians, labels=[False, False, False, False], dashes=[2, 2], linewidth=0.5)
        # # # ----------------------------------------------------------------

        ax.invert_xaxis()

        plt.ylabel(r'$\mathrm{Declination}$', fontsize=16, labelpad=36)
        plt.xlabel(r'$\mathrm{Right\;Ascension}$', fontsize=16, labelpad=30)

        output_path = "%s/%s" % (formatted_healpix_dir, self.options.tile_file.replace(".csv", ".png"))
        fig.savefig(output_path, bbox_inches='tight')  # ,dpi=840
        plt.close('all')
        print("... Done.")


if __name__ == "__main__":
    useagestring = """python Plot_Teglon.py [options]

python Plot_Teglon.py --gw_id <gwid> --healpix_file <filename> --extinct 0.50 --prob_type 2D --cum_prob_outer 0.90 
--cum_prob_inner 0.50 --band r --tile_file <FILENAME> --tele s --galaxies_file <FILENAME> 
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
    print("Teglon `Plot_Teglon` execution time: %s" % duration)
    print("********* end DEBUG ***********\n")


