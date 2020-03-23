import matplotlib

# region imports
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
import subprocess as s

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
from scipy.interpolate import interp1d, interp2d

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
from scipy.stats import pearsonr
# endregion

# region config

# Set up dustmaps config
config["data_dir"] = "./"

# Generate all pixel indices
# cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

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
# endregion

# region db CRUD
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

# endregion



start = time.time()

index_files = False
load_index = False
generate_uniquePspsOBids_input = False
get_photo_z = True

path_format = "{}/{}"
# ps1_strm_dir = "../PS1_DR2_QueryData/PS1_STRM"
ps1_strm_dir = "/data2/ckilpatrick/photoz"
ps1_strm_base_file = "hlsp_ps1-strm_ps1_imaging_3pi-{}_grizy_v1.0_cat.csv"

output_file = path_format.format(ps1_strm_dir, "PS1_STRM_Index.txt")
uniquePspsOBid_index_output_file = path_format.format(ps1_strm_dir, "PS1_STRM_Indices_For_GW190814_Galaxies.txt")
uniquePspsOBid_data_output_file = path_format.format(ps1_strm_dir, "PS1_STRM_Data_For_GW190814_Galaxies.txt")

if index_files:
    small_initial_val = 9999999999999999999999999999
    large_initial_val = -9999999999999999999999999999

    # initialize output

    with open(output_file, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(("File_ID", "Min_uniquePspsOBid", "Max_uniquePspsOBid"))

    for file in os.listdir(ps1_strm_dir):
        if file.endswith(".csv"):

            file_id = (file.split("_")[4]).replace("3pi-", "")
            smallest_uniquePspsOBid = small_initial_val
            largest_uniquePspsOBid = large_initial_val

            file_path = path_format.format(ps1_strm_dir, file)

            print("Processing: %s..." % file_path)

            with open(file_path, 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
                next(csvreader)  # skip header

                for row in csvreader:
                    uniquePspsOBid = float(row[1])
                    if uniquePspsOBid > largest_uniquePspsOBid:
                        largest_uniquePspsOBid = uniquePspsOBid

                    if uniquePspsOBid < smallest_uniquePspsOBid:
                        smallest_uniquePspsOBid = uniquePspsOBid

            if smallest_uniquePspsOBid == small_initial_val or largest_uniquePspsOBid == large_initial_val:
                raise Exception("Couldn't find valid uniquePspsOBid for file: {}! Exiting...".format(file))

            # append the results
            with open(output_file, 'a') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow((file_id, "%s" % int(smallest_uniquePspsOBid), "%s" % int(largest_uniquePspsOBid)))

if load_index:

    PS1_STRM_FileIndex_data = []

    with open(output_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        next(csvreader)  # skip header

        for row in csvreader:
            PS1_STRM_FileIndex_data.append((int(row[0]), int(row[1]), int(row[2])))

    PS1_STRM_FileIndex_data_sorted = sorted(PS1_STRM_FileIndex_data, key=lambda x: x[0])

    insert_PS1_file_index = '''
        INSERT INTO PS1_STRM_FileIndex (File_id, uniquePspsOBid_min, uniquePspsOBid_max) VALUES (%s, %s, %s) 
    '''
    print("Inserting %s rows..." % len(PS1_STRM_FileIndex_data_sorted))
    batch_insert(insert_PS1_file_index, PS1_STRM_FileIndex_data_sorted, batch_size=5000)

if generate_uniquePspsOBids_input:
    select_uniquePspsOBids = '''
        SELECT uniquePspsOBid FROM PS1_DR2_S190814bv_Galaxies 
    '''
    ids = query_db([select_uniquePspsOBids])[0]
    with open(uniquePspsOBid_index_output_file, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(("uniquePspsOBid", ))
        for id in ids:
            csvwriter.writerow(id)

if get_photo_z:

    def get_records(file, id_list):
        with open(file, "r") as csvfile:
            datareader = csv.reader(csvfile)
            next(datareader)

            stop = len(id_list)
            count = 0
            for row in datareader:
                if int(row[1]) in id_list:
                    yield row
                    count += 1
                elif count == stop:
                    return

    select_file_index = '''
        SELECT Get_PS1_STRM_FileIndex(%s)
    '''

    # Load uniquePspsOBids from file...
    ids = []
    with open(uniquePspsOBid_index_output_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        next(csvreader)  # skip header

        for row in csvreader:
            ids.append((int(row[0])))

    file_maps = {}
    for id in ids:
        file_id = str(query_db([select_file_index % id])[0][0][0]).zfill(4)
        if file_id not in file_maps:
            file_maps[file_id] = []
        file_maps[file_id].append(id)

    print("Files to search: %s" % file_maps.keys())
    print("Number of files to search: %s" % len(file_maps))

    with open(uniquePspsOBid_data_output_file, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(("objID", "uniquePspsOBid", "raMean", "decMean", "l", "b", "class", "prob_Galaxy",
                            "prob_Star", "prob_QSO", "extrapolation_Class", "cellDistance_Class", "cellID_Class",
                            "z_phot", "z_photErr", "z_phot0", "extrapolation_Photoz", "cellDistance_Photoz",
                            "cellID_Photoz"))

    start_search = time.time()
    for file_id, obj_ids in file_maps.items():
        file_path = path_format.format(ps1_strm_dir, ps1_strm_base_file.format(file_id))
        records = get_records(file_path, obj_ids)
        for r in records:
            with open(uniquePspsOBid_data_output_file, 'a') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow(r)

    end_search = time.time()
    search_duration = (end_search - start_search)
    print("\n********* start DEBUG ***********")
    print("Execution time: %s" % search_duration)
    print("********* end DEBUG ***********\n")


end = time.time()
duration = (end - start)
print("\n********* start DEBUG ***********")
print("Execution time: %s" % duration)
print("********* end DEBUG ***********\n")