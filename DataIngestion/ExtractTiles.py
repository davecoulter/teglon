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


# thacher_tile_q = tile_select % (THACHER_detector_result[0], band_result[3], map_result[0])
# THACHER_tile_result = query_db([thacher_tile_q])[0]


# with open('../Events/S190901ap/S190901ap_THACHER_bayestar_4D_90th.txt','w') as csvfile:

# 	csvwriter = csv.writer(csvfile)

# 	cols = []
# 	cols.append('# FieldName')
# 	cols.append('FieldRA')
# 	cols.append('FieldDec')
# 	cols.append('Telscope')
# 	cols.append('Filter')
# 	cols.append('ExpTime')
# 	cols.append('Priority')
# 	cols.append('Status')
# 	csvwriter.writerow(cols)

# 	for i, row in enumerate(THACHER_tile_result):

# 		c = coord.SkyCoord(row[2], row[3], unit=(u.deg, u.deg))
# 		coord_str = GetSexigesimalString(c) 

# 		cols = []

# 		cols.append(row[1])
# 		cols.append(coord_str[0])
# 		cols.append(coord_str[1])
# 		cols.append("THACHER")
# 		cols.append("r")
# 		cols.append("180")
# 		cols.append(row[4])
# 		cols.append('False')
# 		csvwriter.writerow(cols)

# 	print("Done w/ Thacher")


# galaxies_select = '''
# 	SELECT 
# 		t.Name_GWGC,
# 		t.Name_HyperLEDA,
# 		t.Name_2MASS,
# 		t.RA,
# 		t._Dec,
# 		t.z_dist,
# 		t.B,
# 		t.K,
# 		t.GalaxyProb,
# 		t.PixelCompleteness,
# 		t.Original2DPixelProb,
# 		t.ReweightedPixelProb,
# 		t.MeanDist,
# 		t.cum_prob
# 	FROM
# 		(SELECT 
# 			gd2.Name_GWGC,
# 			gd2.Name_HyperLEDA,
# 			gd2.Name_2MASS,
# 			gd2.RA,
# 			gd2._Dec,
# 			gd2.z_dist,
# 			gd2.B,
# 			gd2.K,
# 			hp_gd2_w.GalaxyProb,
# 			hpc.PixelCompleteness,
# 			hp.Prob as Original2DPixelProb,
# 			hpc.NetPixelProb as ReweightedPixelProb,
# 			hp.Mean as MeanDist,
# 			SUM(hp_gd2_w.GalaxyProb) OVER(ORDER BY SUM(hp_gd2_w.GalaxyProb) DESC) AS cum_prob
# 		FROM
# 			HealpixPixel_GalaxyDistance2_Weight hp_gd2_w
# 		JOIN HealpixPixel_GalaxyDistance2 hp_gd2 on hp_gd2.id = hp_gd2_w.HealpixPixel_GalaxyDistance2_id
# 		JOIN GalaxyDistance2 gd2 on gd2.id = hp_gd2.GalaxyDistance2_id
# 		JOIN HealpixPixel_Completeness hpc on hpc.HealpixPixel_id = hp_gd2.HealpixPixel_id
# 		JOIN HealpixPixel hp on hp.id = hpc.HealpixPixel_id
# 		-- WHERE gd2._Dec BETWEEN -40.0 and 70.0 and gd2.RA BETWEEN 180.0 and 315
# 		GROUP BY 
# 			gd2.Name_GWGC,
# 			gd2.Name_HyperLEDA,
# 			gd2.Name_2MASS,
# 			gd2.RA,
# 			gd2._Dec,
# 			gd2.z_dist,
# 			gd2.B,
# 			gd2.K,
# 			hp_gd2_w.GalaxyProb,
# 			hpc.PixelCompleteness,
# 			hp.Prob,
# 			hpc.NetPixelProb,
# 			hp.Mean
# 		ORDER BY hp_gd2_w.GalaxyProb DESC) t
# 		WHERE t.cum_prob < 0.9
# 		LIMIT 0, 10000

# '''

# galaxy_result = query_db([galaxies_select])[0]
# with open('../Events/S190901ap/S190901ap_GALAXIES_bayestar_90.txt','w') as csvfile:

# 	csvwriter = csv.writer(csvfile)

# 	cols = []
# 	cols.append('# FieldName')
# 	cols.append('FieldRA')
# 	cols.append('FieldDec')
# 	cols.append('Telscope')
# 	cols.append('Filter')
# 	cols.append('ExpTime')
# 	cols.append('Priority')
# 	cols.append('Status')
# 	csvwriter.writerow(cols)

# 	for i, row in enumerate(galaxy_result):

# 		c = coord.SkyCoord(row[3], row[4], unit=(u.deg, u.deg))
# 		coord_str = GetSexigesimalString(c) 

# 		cols = []

# 		Name_GWGC = row[0]
# 		Name_HyperLEDA = row[1]
# 		Name_2MASS = row[2]

# 		field_name = ""
# 		if Name_GWGC is not None:
# 			field_name = Name_GWGC
# 		elif Name_HyperLEDA is not None:
# 			field_name = "LEDA" + Name_HyperLEDA
# 		elif Name_2MASS is not None:
# 			field_name = Name_2MASS

# 		if field_name == "":
# 			raise("No field name!")

# 		cols.append(field_name)
# 		cols.append(coord_str[0])
# 		cols.append(coord_str[1])
# 		cols.append("NICKEL")
# 		cols.append("r")
# 		cols.append("120")
# 		cols.append(row[8])
# 		cols.append('False')
# 		csvwriter.writerow(cols)

# 	print("Done w/ Galaxies")


# with open('../Events/S190901ap/S190901ap_GALAXIES_bayestar_90.reg','w') as csvfile:

# 	csvfile.write("# Region file format: DS9 version 4.0 global\n\n")
# 	csvfile.write("global color=lightgreen\n")
# 	csvfile.write("ICRS\n")

# 	for i, row in enumerate(galaxy_result):

# 		Name_GWGC = row[0]
# 		Name_HyperLEDA = row[1]
# 		Name_2MASS = row[2]

# 		field_name = ""
# 		if Name_GWGC is not None:
# 			field_name = Name_GWGC
# 		elif Name_HyperLEDA is not None:
# 			field_name = "LEDA" + Name_HyperLEDA
# 		elif Name_2MASS is not None:
# 			field_name = Name_2MASS

# 		if field_name == "":
# 			raise("No field name!")

# 		csvfile.write('circle(%s,%s,120") # width=4 text="%s"\n' % (row[3], row[4], field_name))

# 	print("Done w/ Region File")


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

        parser.add_option('--tele', default="s", type="str",
                          help='Telescope abbreviation for the telescope to extract files for. Default: `s`. Available: (s:Swope, t:Thacher)')

        parser.add_option('--band', default="r", type="str",
                          help='Telescope abbreviation for the telescope to extract files for. Default: `r`. Available: (g, r, i)')

        parser.add_option('--exp_time', default="60.0", type="float",
                          help='Exposure time (seconds) to write out. Default: `60.0`. Must be > 0.0')

        parser.add_option('--extinct', default="0.5", type="float",
                          help='Extinction in mags in the specified band to be less than. Default: 0.5 mag. Must be > 0.0')

        parser.add_option('--prob_type', default="4D", type="str",
                          help='''Probability type to consider. Default `4D`. Available: (4D, 2D)''')

        parser.add_option('--cum_prob', default="0.9", type="float",
                          help='Cumulative prob to cover in tiles. Default 0.9. Must be > 0.2 and < 0.95')

        parser.add_option('--min_ra', default="-1.0", type="float",
                          help='''Optional, decimal format. If specified, `min_ra`, `max_ra`, `min_dec`, and `max_dec` 
                          are all required and supersede any detector limits. Tiles within this range will be selected 
                          <= to prob limit specified in `cum_prob`. `min_ra` must be >= 0.0 and < `max_ra`
                          ''')

        parser.add_option('--max_ra', default="-1.0", type="float",
                          help='''Optional, decimal format. If specified, `min_ra`, `max_ra`, `min_dec`, and `max_dec` 
                          are all required and supersede any detector limits. Tiles within this range will be selected 
                          <= to prob limit specified in `cum_prob`. `max_ra` must be <=360.0 and > `min_ra`
                          ''')

        parser.add_option('--min_dec', default="-1.0", type="float",
                          help='''Optional, decimal format. If specified, `min_ra`, `max_ra`, `min_dec`, and `max_dec` 
                          are all required and supersede any detector limits. Tiles within this range will be selected 
                          <= to prob limit specified in `cum_prob`. `min_dec` must be >= -90.0 and < `max_dec`
                          ''')

        parser.add_option('--max_dec', default="-1.0", type="float",
                          help='''Optional, decimal format. If specified, `min_ra`, `max_ra`, `min_dec`, and `max_dec` 
                          are all required and supersede any detector limits. Tiles within this range will be selected 
                          <= to prob limit specified in `cum_prob`. `max_dec` must be <= 90.0 and > `min_dec`
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

        detector_mapping = {
            "s": "SWOPE",
            "t": "THACHER"
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

        if self.options.tele not in detector_mapping:
            is_error = True
            print("Invalid telescope selection. Available telescopes: %s" % detector_mapping.keys())

        if self.options.band not in band_mapping:
            is_error = True
            print("Invalid band selection. Available bands: %s" % band_mapping.keys())

        if self.options.exp_time <= 0.0:
            is_error = True
            print("Exposure time must be > 0.0 seconds")

        if self.options.extinct <= 0.0:
            is_error = True
            print("Extinction must be a valid float > 0.0")

        if not (self.options.prob_type == _4D or self.options.prob_type == _2D):
            is_error = True
            print("Prob type must either be `4D` or `2D`")

        if self.options.cum_prob > 0.95 or self.options.cum_prob < 0.20:
            is_error = True
            print("Cumulative prob must be between 0.2 and 0.95")

        is_box_query = False
        if self.options.min_ra != -1 and \
                self.options.max_ra != -1 and \
                self.options.min_dec != -1 and \
                self.options.max_dec != -1:
            # if any of these are non-default values, then we will validate them all
            is_box_query = True

            if self.options.min_ra < 0.0 or self.options.min_ra >= self.options.max_ra:
                is_error = True
                print("Min RA must be >= 0.0 and be < Max RA")

            if self.options.max_ra > 360.0 or self.options.max_ra <= self.options.min_ra:
                is_error = True
                print("Max RA must be <= 360.0 and be > Min RA")

            if self.options.min_dec < -90.0 or self.options.min_dec >= self.options.max_dec:
                is_error = True
                print("Min Dec must be >= -90.0 and be < Max Dec")

            if self.options.max_dec > 90.0 or self.options.max_dec <= self.options.min_dec:
                is_error = True
                print("Max Dec must be <= 90.0 and be > Min Dec")

        if is_error:
            print("Exiting...")
            return 1

        hpx_path = "%s/%s" % (formatted_healpix_dir, self.options.healpix_file)
        telescope_name = detector_mapping[self.options.tele]
        band_name = band_mapping[self.options.band]

        # Get Map ID
        healpix_map_select = "SELECT id, RescaledNSIDE FROM HealpixMap WHERE GWID = '%s' and Filename = '%s'"
        healpix_map_id = int(query_db([healpix_map_select % (self.options.gw_id, self.options.healpix_file)])[0][0][0])

        band_select = "SELECT id, Name, F99_Coefficient FROM Band WHERE `Name`='%s'"
        band_result = query_db([band_select % band_name])[0][0]
        band_id = band_result[0]
        band_F99 = float(band_result[2])

        detector_select_by_name = "SELECT id, Name, Deg_width, Deg_height, Deg_radius, Area, MinDec, MaxDec FROM Detector WHERE Name='%s'"
        detector_result = query_db([detector_select_by_name % telescope_name])[0][0]
        detector_id = int(detector_result[0])

        # 4D NON-BOX QUERY
        # REPLACEMENT PARAMETERS: (detector_id, band_F99, self.options.extinct, healpix_map_id, self.options.cum_prob)
        tile_select_4D = '''
            SELECT 
                tt.id, 
                tt.FieldName,
                tt.RA,
                tt._Dec,
                tt.net_prob,
                tt.mean_pixel_dist,
                tt.A_lambda,
                tt.cum_prob
            FROM
                (SELECT 
                    t.id, 
                    t.FieldName,
                    t.RA,
                    t._Dec,
                    t.net_prob,
                    t.mean_pixel_dist,
                    t.A_lambda,
                    SUM(t.net_prob) OVER(ORDER BY SUM(t.net_prob) DESC) AS cum_prob
                 FROM (
                    select 
                        st.id, 
                        st.FieldName,
                        st.RA,
                        st._Dec,
                        SUM(hpc.NetPixelProb) as net_prob,
                        AVG(hp.Mean) as mean_pixel_dist,
                        sp_ebv.EBV*@F99 as A_lambda
                    from 
                        StaticTile st
                    join StaticTile_HealpixPixel st_hp on st_hp.StaticTile_id = st.id
                    JOIN HealpixPixel_Completeness hpc on hpc.HealpixPixel_id = st_hp.HealpixPixel_id
                    JOIN HealpixPixel hp on hp.id = hpc.HealpixPixel_id
                    JOIN SkyPixel_EBV sp_ebv on sp_ebv.N128_SkyPixel_id = st.N128_SkyPixel_id
                    JOIN Detector d on d.id = st.Detector_id 
                    where d.id = %s and 
                            st._Dec BETWEEN d.MinDec and d.MaxDec and 
                            sp_ebv.EBV*%s < %s and 
                            hp.HealpixMap_id = %s
                    group by
                        st.id,
                        st.FieldName, 
                        st.RA,
                        st._Dec,
                        A_lambda) t
                group by 
                    t.id, 
                    t.FieldName,
                    t.RA,
                    t._Dec,
                    t.net_prob,
                    t.mean_pixel_dist,
                    t.A_lambda
                order by t.net_prob desc) tt
                WHERE tt.cum_prob < %s;
		'''

        # 2D NON-BOX QUERY
        # REPLACEMENT PARAMETERS: (detector_id, band_F99, self.options.extinct, healpix_map_id, self.options.cum_prob)
        tile_select_2D = '''
        SELECT 
            tt.id, 
            tt.FieldName,
            tt.RA,
            tt._Dec,
            tt.net_2D_prob,
            tt.mean_pixel_dist,
            tt.A_lambda,
            tt.cum_2D_prob
        FROM
            (SELECT 
                t.id, 
                t.FieldName,
                t.RA,
                t._Dec,
                t.net_2D_prob,
                t.mean_pixel_dist,
                t.A_lambda,
                SUM(t.net_2D_prob) OVER(ORDER BY SUM(t.net_2D_prob) DESC) AS cum_2D_prob
             FROM (
                select 
                    st.id, 
                    st.FieldName,
                    st.RA,
                    st._Dec,
                    SUM(hp.Prob) as net_2D_prob,
                    AVG(hp.Mean) as mean_pixel_dist,
                    sp_ebv.EBV*@F99 as A_lambda
                from 
                    StaticTile st
                join StaticTile_HealpixPixel st_hp on st_hp.StaticTile_id = st.id
                JOIN HealpixPixel_Completeness hpc on hpc.HealpixPixel_id = st_hp.HealpixPixel_id
                JOIN HealpixPixel hp on hp.id = hpc.HealpixPixel_id
                JOIN SkyPixel_EBV sp_ebv on sp_ebv.N128_SkyPixel_id = st.N128_SkyPixel_id
                JOIN Detector d on d.id = st.Detector_id 
                where d.id = %s and 
                        st._Dec BETWEEN d.MinDec and d.MaxDec and 
                        sp_ebv.EBV*%s < %s and 
                        hp.HealpixMap_id = %s
                group by
                    st.id,
                    st.FieldName, 
                    st.RA,
                    st._Dec,
                    A_lambda) t
            group by 
                t.id, 
                t.FieldName,
                t.RA,
                t._Dec,
                t.net_2D_prob,
                t.mean_pixel_dist,
                t.A_lambda
            order by t.net_2D_prob desc) tt
            WHERE tt.cum_2D_prob < %s;
    '''

        # 4D BOX QUERY
        # REPLACEMENT PARAMETERS: (detector_id, self.options.min_dec, self.options.max_dec, self.options.min_ra,
        # self.options.max_ra, band_F99, self.options.extinct, healpix_map_id, self.options.cum_prob)
        box_tile_select_4D = '''
            SELECT 
                tt.id, 
                tt.FieldName,
                tt.RA,
                tt._Dec,
                tt.net_prob,
                tt.mean_pixel_dist,
                tt.A_lambda,
                tt.cum_prob
            FROM
                (SELECT 
                    t.id, 
                    t.FieldName,
                    t.RA,
                    t._Dec,
                    t.net_prob,
                    t.mean_pixel_dist,
                    t.A_lambda,
                    SUM(t.net_prob) OVER(ORDER BY SUM(t.net_prob) DESC) AS cum_prob
                 FROM (
                    select 
                        st.id, 
                        st.FieldName,
                        st.RA,
                        st._Dec,
                        SUM(hpc.NetPixelProb) as net_prob,
                        AVG(hp.Mean) as mean_pixel_dist,
                        sp_ebv.EBV*@F99 as A_lambda
                    from 
                        StaticTile st
                    join StaticTile_HealpixPixel st_hp on st_hp.StaticTile_id = st.id
                    JOIN HealpixPixel_Completeness hpc on hpc.HealpixPixel_id = st_hp.HealpixPixel_id
                    JOIN HealpixPixel hp on hp.id = hpc.HealpixPixel_id
                    JOIN SkyPixel_EBV sp_ebv on sp_ebv.N128_SkyPixel_id = st.N128_SkyPixel_id
                    JOIN Detector d on d.id = st.Detector_id 
                    where d.id = %s and 
                            st._Dec BETWEEN %s and %s and 
                            st.RA BETWEEN %s and %s and
                            sp_ebv.EBV*%s < %s and 
                            hp.HealpixMap_id = %s
                    group by
                        st.id,
                        st.FieldName, 
                        st.RA,
                        st._Dec,
                        A_lambda) t
                group by 
                    t.id, 
                    t.FieldName,
                    t.RA,
                    t._Dec,
                    t.net_prob,
                    t.mean_pixel_dist,
                    t.A_lambda
                order by t.net_prob desc) tt
                WHERE tt.cum_prob < %s;
            '''

        # 2D BOX QUERY
        # REPLACEMENT PARAMETERS: (detector_id, self.options.min_dec, self.options.max_dec, self.options.min_ra,
        # self.options.max_ra, band_F99, self.options.extinct, healpix_map_id, self.options.cum_prob)
        box_tile_select_2D = '''
                SELECT 
                    tt.id, 
                    tt.FieldName,
                    tt.RA,
                    tt._Dec,
                    tt.net_2D_prob,
                    tt.mean_pixel_dist,
                    tt.A_lambda,
                    tt.cum_2D_prob
                FROM
                    (SELECT 
                        t.id, 
                        t.FieldName,
                        t.RA,
                        t._Dec,
                        t.net_2D_prob,
                        t.mean_pixel_dist,
                        t.A_lambda,
                        SUM(t.net_2D_prob) OVER(ORDER BY SUM(t.net_2D_prob) DESC) AS cum_2D_prob
                     FROM (
                        select 
                            st.id, 
                            st.FieldName,
                            st.RA,
                            st._Dec,
                            SUM(hp.Prob) as net_2D_prob,
                            AVG(hp.Mean) as mean_pixel_dist,
                            sp_ebv.EBV*@F99 as A_lambda
                        from 
                            StaticTile st
                        join StaticTile_HealpixPixel st_hp on st_hp.StaticTile_id = st.id
                        JOIN HealpixPixel_Completeness hpc on hpc.HealpixPixel_id = st_hp.HealpixPixel_id
                        JOIN HealpixPixel hp on hp.id = hpc.HealpixPixel_id
                        JOIN SkyPixel_EBV sp_ebv on sp_ebv.N128_SkyPixel_id = st.N128_SkyPixel_id
                        JOIN Detector d on d.id = st.Detector_id 
                        where d.id = %s and 
                            st._Dec BETWEEN %s and %s and 
                            st.RA BETWEEN %s and %s and
                            sp_ebv.EBV*%s < %s and 
                            hp.HealpixMap_id = %s
                        group by
                            st.id,
                            st.FieldName, 
                            st.RA,
                            st._Dec,
                            A_lambda) t
                    group by 
                        t.id, 
                        t.FieldName,
                        t.RA,
                        t._Dec,
                        t.net_2D_prob,
                        t.mean_pixel_dist,
                        t.A_lambda
                    order by t.net_2D_prob desc) tt
                    WHERE tt.cum_2D_prob < %s;
            '''


        select_to_execute = ""
        if is_box_query:
            if self.options.prob_type == _4D:
                select_to_execute = box_tile_select_4D % (detector_id, self.options.min_dec, self.options.max_dec,
                                                          self.options.min_ra, self.options.max_ra, band_F99,
                                                          self.options.extinct, healpix_map_id, self.options.cum_prob)
            else:
                select_to_execute = box_tile_select_2D % (detector_id, self.options.min_dec, self.options.max_dec,
                                                          self.options.min_ra, self.options.max_ra, band_F99,
                                                          self.options.extinct, healpix_map_id, self.options.cum_prob)
        else:
            if self.options.prob_type == _4D:
                select_to_execute = tile_select_4D % (detector_id, band_F99, self.options.extinct, healpix_map_id,
                                                          self.options.cum_prob)
            else:
                select_to_execute = tile_select_2D % (detector_id, band_F99, self.options.extinct, healpix_map_id,
                                                          self.options.cum_prob)

        tile_result = query_db([select_to_execute])[0]

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

        box_file_formatter = "%s/%s_%s_%s_%s_box.csv"
        non_box_file_formatter = "%s/%s_%s_%s_%s.csv"
        formatted_output_path = ""

        if is_box_query:
            formatted_output_path = box_file_formatter % (formatted_healpix_dir, telescope_name, self.options.prob_type,
                                                     self.options.cum_prob, self.options.healpix_file.replace(",","_"))
        else:
            formatted_output_path = non_box_file_formatter % (formatted_healpix_dir, telescope_name, self.options.prob_type,
                                                     self.options.cum_prob, self.options.healpix_file.replace(",", "_"))

        with open(formatted_output_path, 'w') as csvfile:
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

            for i, row in enumerate(tile_result):
                c = coord.SkyCoord(row[2], row[3], unit=(u.deg, u.deg))
                coord_str = GetSexigesimalString(c)

                cols = []

                cols.append(row[1])
                cols.append(coord_str[0])
                cols.append(coord_str[1])
                cols.append(telescope_name)
                cols.append(self.options.band)
                cols.append(str(self.options.exp_time))
                cols.append(row[4])
                cols.append('False')
                csvwriter.writerow(cols)

            print("Done writing out %s" % telescope_name)


        t2 = time.time()
        print("\n********* start DEBUG ***********")
        print("Extract Tiles execution time: %s" % (t2 - t1))
        print("********* end DEBUG ***********\n")


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
    print("Teglon `ExtractTiles` execution time: %s" % duration)
    print("********* end DEBUG ***********\n")
