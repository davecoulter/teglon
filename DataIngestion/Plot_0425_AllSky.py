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
import matplotlib.patheffects as path_effects


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

from spherical_geometry.polygon import SphericalPolygon
import ephem
from datetime import datetime, timezone, timedelta

from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy import units as unit
from scipy import interpolate
from astropy.time import Time
from astropy.coordinates import get_sun
import matplotlib.animation as animation


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

        lons = np.linspace(-180, 180, 1000)
        lats = np.linspace(-90, 90, 1000)
        LONS, LATS = np.meshgrid(lons, lats)
        earth_grid = EarthLocation(lat=LATS * unit.deg, lon=LONS * unit.deg)

        # discovery_time = Time('2019-04-25 08:18:26', scale='utc') # Trigger
        discovery_time = Time('2019-04-25 09:06:22.000', scale='utc') # First Obs

        sun_tuple = get_sun(discovery_time)
        sun_coord = SkyCoord(sun_tuple.ra.degree, sun_tuple.dec.degree, unit=(unit.deg, unit.deg))
        sun = sun_coord.transform_to(AltAz(obstime=discovery_time, location=earth_grid)).alt.degree

        f = interpolate.interp2d(lons, lats, sun, kind='cubic')
        sun_alt = f(lons, lats)


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
            'KAIT': "lawngreen",
            'NICKEL': "cyan",
            'SWOPE': "yellow",
            'THACHER': "blue"
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
        SELECT sp.Pixel_Index, sp_ebv.EBV
        FROM SkyPixel sp
        JOIN SkyPixel_EBV sp_ebv ON sp_ebv.N128_SkyPixel_id = sp.id
        WHERE sp_ebv.EBV*%s > %s and NSIDE = 128
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

        # mwe_pix = [Pixel_Element(int(mp[0]), 32, 0.0) for mp in mwe_pix_result]
        mwe_pix = [Pixel_Element(int(mp[0]), 128, float(mp[1])) for mp in mwe_pix_result]
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
        # m = Basemap(projection='ortho', lon_0=247.5, lat_0=0.0) # 20.0

        # HACK
        # Get all sky pixels and plot them
        select_all_pixels = '''
                SELECT
                    hp.id,
                    hp.HealpixMap_id,
                    hp.Pixel_Index,
                    hp.Prob,
                    hp.Distmu,
                    hp.Distsigma,
                    hp.Distnorm,
                    hp.Mean,
                    hp.Stddev,
                    hp.Norm,
                    hp.N128_SkyPixel_id,
                    hpc.PixelCompleteness,
                    hpc.Renorm2DProb,
                    hpc.NetPixelProb
                FROM HealpixPixel hp
                JOIN HealpixPixel_Completeness hpc ON hpc.HealpixPixel_id = hp.id
                WHERE hp.HealpixMap_id = 1;
                '''
        all_pixels_result = query_db([select_all_pixels])[0]
        all_pixels = [Pixel_Element(int(apr[2]), 256, float(apr[13])) for apr in all_pixels_result]

        # Scale colormap
        pixels_probs = [ap.prob for ap in all_pixels]
        min_prob = 1e-7
        max_prob = np.max(pixels_probs)
        print("min prob: %s" % min_prob)
        print("max prob: %s" % max_prob)

        log_norm = colors.LogNorm(min_prob, max_prob)

        for i, p in enumerate(all_pixels):

            clr = plt.cm.inferno(log_norm(min_prob))
            if p.prob > min_prob:
                clr = plt.cm.inferno(log_norm(p.prob))

            p.plot(m, ax, edgecolor='None', facecolor=clr, linewidth=0.25, alpha=0.9)
            if i % 10000 == 0:
                print("Plotted %s pixels" % i)

        sql_inner_poly.plot(m, ax, edgecolor='w', linewidth=0.35, facecolor='None',
                            path_effects=[path_effects.withStroke(linewidth=0.75, foreground='black')])
        sql_outer_poly.plot(m, ax, edgecolor='w', linewidth=0.35, facecolor='None',
                            path_effects=[path_effects.withStroke(linewidth=0.75, foreground='black')])

        mw_ebvs = [p.prob for p in mwe_pix]
        min_ebv = np.min(mw_ebvs)
        max_ebv = np.max(mw_ebvs)
        mwe_norm = colors.Normalize(min_ebv, 2.0)
        for p in mwe_pix:
            # p.plot(m, ax, edgecolor='None', linewidth=0.5, facecolor='firebrick', alpha=0.5)
            clr = plt.cm.Blues(mwe_norm(p.prob))
            p.plot(m, ax, edgecolor='None', linewidth=0.5, facecolor=clr, alpha=0.45)

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


        # Plot Sun
        twi18 = m.contourf(LONS, LATS, sun_alt,
                           latlon=True,
                           levels=[60.0, 90.0],
                           # colors=('aliceblue'),
                           colors=('cornsilk'),
                           alpha=0.15, zorder=9990)
        twi18 = m.contourf(LONS, LATS, sun_alt,
                           latlon=True,
                           levels=[40.0, 90.0],
                           # colors=('aliceblue'),
                           colors=('cornsilk'),
                           alpha=0.15, zorder=9990)
        twi18 = m.contourf(LONS, LATS, sun_alt,
                           latlon=True,
                           levels=[20.0, 90.0],
                           # colors=('aliceblue'),
                           colors=('cornsilk'),
                           alpha=0.15, zorder=9990)

        twi18 = m.contour(LONS, LATS, sun_alt,
                           latlon=True,
                           levels=[20.0, 40, 60, 90.0],
                           colors=('gold', 'gold', 'gold'),
                          # colors=('darkorange', 'darkorange', 'darkorange'),
                           alpha=1.0, zorder=9990,
                          linewidths=(0.5))

        fmt = {}
        strs = [r'$75\degree$', r'$60\degree$', r'$45\degree$']
        for l, s in zip(twi18.levels, strs):
            fmt[l] = s
        c_labels = ax.clabel(twi18, inline=True, fontsize=10, fmt=fmt,
                  levels=twi18.levels, colors=('gold', 'gold', 'gold'), use_clabeltext=True,
                  inline_spacing=60, zorder=9999)
        [txt.set_path_effects([path_effects.withStroke(linewidth=0.75,
            foreground='k')]) for txt in c_labels]


        x, y = m(sun_tuple.ra.degree, sun_tuple.dec.degree)
        m.plot(x, y, color='gold', marker='o', markersize=20.0, markeredgecolor='darkorange', linewidth=0.25, zorder=9999)


        # # # -------------- Use this for mollweide projections --------------
        meridians = np.arange(0., 360., 60.)
        # labels = [False, True, False, False] => right, left top, bottom

        dec_str = {
            60.0:r"+60$\degree$",
            30.0:r"+30$\degree$",
            0.0:r"0$\degree$",
            -30.0:r"-30$\degree$",
            -60.0:r"-60$\degree$"
        }
        m.drawparallels(np.arange(-90., 91., 30.), fmt=lambda dec: dec_str[dec], fontsize=14, labels=[True, False, False, False], dashes=[2, 2],
                        linewidth=0.5, color="silver", xoffset=2500000, alpha=0.8)
        m.drawmeridians(meridians, labels=[False, False, False, False], dashes=[2, 2], linewidth=0.5, color="silver", alpha=0.8)

        # HACK
        # min_prob = 1e-07
        # max_prob = 0.00121692958931
        # log_norm = colors.LogNorm(min_prob, max_prob)
        ra_labels = {
            300.0: "20h",
            240.0: "16h",
            180.0: "12h",
            120.0: "8h",
            60.0: "4h"
        }

        for mer in meridians[1:]:
            # plt.annotate("%0.0f" % mer, xy=m(mer, 0), xycoords='data', color="silver", fontsize=14, zorder=9999)
            plt.annotate(ra_labels[mer], xy=m(mer+8.0, -30), xycoords='data', color="silver", fontsize=14, zorder=9999,
                         path_effects=[path_effects.withStroke(linewidth=0.75, foreground='black')])
        # # # ----------------------------------------------------------------

        # discovery_time = Time('2019-04-25 09:06:22.000', scale='utc') # First Obs
        # plt.annotate("UT 2019-04-25 09:06:22", xy=m(mer + 8.0, -30), xycoords='data', color="silver", fontsize=14, zorder=9999)

        sm = plt.cm.ScalarMappable(norm=log_norm, cmap=plt.cm.inferno)
        sm.set_array([])  # can be an empty list
        # tks = np.linspace(min_prob, max_prob, 11)
        tks = np.logspace(np.log10(min_prob), np.log10(max_prob), 5)
        # tks_strings = []
        # for t in tks:
        #     tks_strings.append('%0.0E' % (t * 100))
        tks_strings = [r"$\mathrm{10^{-5}}$",
                       r"$\mathrm{10^{-4}}$",
                       "0.001",
                       "0.01",
                       "0.1"]

        cb = fig.colorbar(sm, ax=ax, ticks=tks, orientation='horizontal',
                          fraction=0.08951, pad=0.05, alpha=0.80,
                          aspect=40)

        cb.ax.set_xticklabels(tks_strings, fontsize=14)

        cb.set_label("% per Pixel", fontsize=14, labelpad=4.0)
        cb.outline.set_linewidth(1.0)

        ax.invert_xaxis()

        plt.ylabel(r'$\mathrm{Declination}$', fontsize=16, labelpad=5)
        plt.xlabel(r'$\mathrm{Right\;Ascension}$', fontsize=16, labelpad=5)

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


