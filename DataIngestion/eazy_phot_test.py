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
from AAOmega_Objects import *

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
import random

from scipy.stats import norm, gaussian_kde, mode
from scipy.integrate import trapz
from functools import reduce
# endregion

# region config

# Set up dustmaps config
config["data_dir"] = "./"

# Generate all pixel indices
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

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



load_PS1_data = False
write_eazy_input = False
do_eazy = False
test_photozs = False

test_curve_fit = False
read_region_poly = True
h0_calc_test_single = False
h0_calc_test_sample = False
h0_calc_test_multiple_sample = False




if load_PS1_data:

    N128_dict = {}
    print("\tLoading NSIDE 128 pixels...")
    with open('N128_dict.pkl', 'rb') as handle:
        N128_dict = pickle.load(handle)
        del handle

    insert_PS1 = '''
        INSERT INTO PS1_Galaxy 
            (objID, 
            uniquePspsOBid, 
            raStack, 
            decStack, 
            raMean, 
            decMean, 
            gaia_ra, 
            gaia_dec, 
            Coord, 
            synth_B1,
            synth_B2, 
            ng, 
            gMeanPSFMag, 
            gMeanPSFMagErr, 
            gMeanKronMag, 
            gMeanKronMagErr,
            gMeanApMag, 
            gMeanApMagErr,
            nr, 
            rMeanPSFMag, 
            rMeanPSFMagErr, 
            rMeanKronMag, 
            rMeanKronMagErr, 
            rMeanApMag,
            rMeanApMagErr, 
            ni, 
            iMeanPSFMag, 
            iMeanPSFMagErr, 
            iMeanKronMag, 
            iMeanKronMagErr, 
            iMeanApMag, 
            iMeanApMagErr, 
            nz, 
            zMeanPSFMag, 
            zMeanPSFMagErr, 
            zMeanKronMag, 
            zMeanKronMagErr, 
            zMeanApMag, 
            zMeanApMagErr, 
            ny, 
            yMeanPSFMag, 
            yMeanPSFMagErr, 
            yMeanKronMag, 
            yMeanKronMagErr, 
            yMeanApMag, 
            yMeanApMagErr, 
            gQfPerfect, 
            rQfPerfect, 
            iQfPerfect, 
            zQfPerfect, 
            yQfPerfect, 
            qualityFlag, 
            objInfoFlag, 
            gpetRadius, 
            rpetRadius, 
            ipetRadius, 
            zpetRadius, 
            ypetRadius, 
            gpetR50, 
            rpetR50, 
            ipetR50, 
            zpetR50, 
            ypetR50, 
            primaryDetection, 
            bestDetection, 
            gKronFlux, 
            gKronFluxErr, 
            rKronFlux, 
            rKronFluxErr, 
            iKronFlux, 
            iKronFluxErr, 
            zKronFlux, 
            zKronFluxErr, 
            yKronFlux, 
            yKronFluxErr, 
            N128_SkyPixel_id 
            ) 
            VALUES 
            (%s, %s, %s, %s, %s, %s, %s, %s, ST_PointFromText(%s, 4326), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
    '''

    central_ra = 12.66735453
    central_dec = -24.87144076
    height = 6.9925457
    width = 5.5816329

    northern_limit = central_dec + height/2.0
    southern_limit = central_dec - height/2.0
    eastern_limit = central_ra + width/2.0
    western_limit = central_ra - width/2.0

    PS1_data = OrderedDict()
    with open("../PS1_DR2_QueryData/GW190814_davecoulter.csv", 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        next(csvreader)  # skip header

        for row in csvreader:

            ra_decimal = float(row[6])
            dec_decimal = float(row[7])

            if ra_decimal == -999 or dec_decimal == -999 or \
                    ra_decimal < western_limit or \
                    ra_decimal > eastern_limit or \
                    dec_decimal < southern_limit or \
                    dec_decimal > northern_limit:
                # out of our Northern 95 box
                pass

            cleaned_fields = []
            # ra_decimal = -999
            # dec_decimal = -999

            # PS1_data.append(row)
            for i, field in enumerate(row):

                if i == 8: # `ng` after `dec`
                    # ra_decimal = float(cleaned_fields[6])
                    # dec_decimal = float(cleaned_fields[7])

                    point = "POINT(%s %s)" % (dec_decimal, ra_decimal - 180.0)
                    cleaned_fields.append(point) # (dec, ra - 180)
                    cleaned_fields.append("SYNTH_B1") # add a place holder to update later
                    cleaned_fields.append("SYNTH_B2")  # add a place holder to update later

                if field == "null":
                    cleaned_fields.append("-999")
                else:
                    cleaned_fields.append(field)

            # replace the SYNTH_B1/2 placeholder
            g = float(cleaned_fields[14])  # gKronMag
            r = float(cleaned_fields[21])  # rKronMag

            # Using Transformation of Pan-STARRS1 gri to Stetson BVRI magnitudes.
            # https://arxiv.org/pdf/1706.06147.pdf
            # Eq. 1; Table 2
            synth_B1 = g + 0.561 * (g - r) + 0.194
            cleaned_fields[9] = synth_B1

            # Using Jester et al 2005 transformation from SDSS ugriz -> UBVRcIc
            # Combined "All stars with Rc-Ic < 1.15" V-R and V eqns...
            # http://classic.sdss.org/dr5/algorithms/sdssUBVRITransform.html
            synth_B2 = g + 0.39 * (g - r) + 0.21
            cleaned_fields[10] = synth_B2

            # Finally, get N128 pixel id from index
            if dec_decimal == -999 or ra_decimal == -999:
                raise Exception("Could not get RA/Dec from row!")

            n128_index = hp.ang2pix(nside=128,
                                    theta=0.5 * np.pi - np.radians(dec_decimal),
                                    phi=np.radians(ra_decimal))
            n128_id = N128_dict[n128_index]
            cleaned_fields.append(n128_id)

            # NOTE: bringing back stack information creates multiple versions of the same object for every
            #   difference in stack properties (e.g. `rpetR50`, `rpetRadius`, etc). Currently, we don't care
            #   about that information -- only the unique objects themselves.

            # SCRUB duplicates out by checking if uniquePspsOBid is already in the dictionary...
            uniquePspsOBid = int(cleaned_fields[1])
            if uniquePspsOBid not in PS1_data:
                PS1_data[uniquePspsOBid] = cleaned_fields

    # Unpack dictionary to list of tuples...
    PS1_data_list = [value for (key, value) in PS1_data.items()]
    print("Inserting %s rows..." % len(PS1_data))
    batch_insert(insert_PS1, PS1_data_list, batch_size=1000)

load_crossmatch = False
crossmatch_dict = OrderedDict()
if load_crossmatch:

    # DISTINCT g.id,
    # g.RA,
    # g._Dec,
    # p.raMean,
    # p.decMean,
    # AngSep(g.RA, g._Dec, p.raMean, p.decMean)
    # g.z_dist,
    # g.B,
    # p.synth_B1,
    # p.synth_B2,
    # p.gMeanKronMag + 0.39 * (p.gMeanKronMag - p.rMeanKronMag) + 0.21 as synth_B2,
    #
    # p.gMeanPSFMag,
    # p.gMeanPSFMagErr,
    # p.rMeanPSFMag,
    # p.rMeanPSFMagErr,
    # p.iMeanPSFMag,
    # p.iMeanPSFMagErr,
    # p.zMeanPSFMag,
    # p.zMeanPSFMagErr,
    # p.yMeanPSFMag,
    # p.yMeanPSFMagErr,
    #
    # p.gMeanApMag,
    # p.gMeanApMagErr,
    # p.rMeanApMag,
    # p.rMeanApMagErr,
    # p.iMeanApMag,
    # p.iMeanApMagErr,
    # p.zMeanApMag,
    # p.zMeanApMagErr,
    # p.yMeanApMag,
    # p.yMeanApMagErr,
    #
    # p.gMeanKronMag,
    # p.gMeanKronMagErr,
    # p.rMeanKronMag,
    # p.rMeanKronMagErr,
    # p.iMeanKronmag,
    # p.iMeanKronMagErr,
    # p.zMeanKronMag,
    # p.zMeanKronMagErr,
    # p.yMeanKronMag,
    # p.yMeanKronMagErr,
    # p.gKronFlux,
    # p.gKronFluxErr,
    # p.rKronFlux,
    # p.rKronFluxErr,
    # p.iKronFlux,
    # p.iKronFluxErr,
    # p.zKronFlux,
    # p.zKronFluxErr,
    # p.yKronFlux,
    # p.yKronFluxErr,
    #
    # g.PGC,
    # g.Name_GWGC,
    # g.Name_HyperLEDA,
    # g.Name_2MASS,
    # g.Name_SDSS_DR12,
    #
    # (CASE
    #  WHEN g.flag2 IN (1, 3)
    # THEN
    # g.z  # assumed to be spec z's
    # ELSE - 1.0000
    # END) as spec_z

    with open("../PS1_DR2_QueryData/GLADE_crossmatch.csv", 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        next(csvreader)  # skip header

        for row in csvreader:

            id = int(row[0])
            sep = float(row[5])
            B = None
            try:
                B = float(row[7])
            except:
                continue

            synth_B1 = float(row[8])
            synth_B2 = float(row[9])
            diff_1 = B - synth_B1
            diff_2 = B - synth_B2
            z_dist = float(row[6])
            z = float(row[56])

            #### KRON ####
            # gKronMag = float(row[31])
            # gKronMagErr = float(row[32])
            #
            # rKronMag = float(row[33])
            # rKronMagErr = float(row[34])
            #
            # iKronMag = float(row[35])
            # iKronMagErr = float(row[36])
            #
            # zKronMag = float(row[37])
            # zKronMagErr = float(row[38])
            #
            # yKronMag = float(row[39])
            # yKronMagErr = float(row[40])
            #
            # gKronFlux = float(row[41])
            # gKronFluxErr = float(row[42])
            #
            # rKronFlux = float(row[43])
            # rKronFluxErr = float(row[44])
            #
            # iKronFlux = float(row[45])
            # iKronFluxErr = float(row[46])
            #
            # zKronFlux = float(row[47])
            # zKronFluxErr = float(row[48])
            #
            # yKronFlux = float(row[49])
            # yKronFluxErr = float(row[50])

            #### APERTURE ####
            gApMag = float(row[21])
            gApMagErr = float(row[22])

            rApMag = float(row[23])
            rApMagErr = float(row[24])

            iApMag = float(row[25])
            iApMagErr = float(row[26])

            zApMag = float(row[27])
            zApMagErr = float(row[28])

            yApMag = float(row[29])
            yApMagErr = float(row[30])

            gKronFlux = float(row[41])
            gKronFluxErr = float(row[42])

            rKronFlux = float(row[43])
            rKronFluxErr = float(row[44])

            iKronFlux = float(row[45])
            iKronFluxErr = float(row[46])

            zKronFlux = float(row[47])
            zKronFluxErr = float(row[48])

            yKronFlux = float(row[49])
            yKronFluxErr = float(row[50])


            # # #### PSF ####
            # gMeanPSFMag = float(row[11])
            # gMeanPSFMagErr = float(row[12])
            #
            # rMeanPSFMag = float(row[13])
            # rMeanPSFMagErr = float(row[14])
            #
            # iMeanPSFMag = float(row[15])
            # iMeanPSFMagErr = float(row[16])
            #
            # zMeanPSFMag = float(row[17])
            # zMeanPSFMagErr = float(row[18])
            #
            # yMeanPSFMag = float(row[19])
            # yMeanPSFMagErr = float(row[20])
            #
            # # HACK: Only here to not break downstream parsing...
            # gKronFlux = float(row[41])
            # gKronFluxErr = float(row[42])
            #
            # rKronFlux = float(row[43])
            # rKronFluxErr = float(row[44])
            #
            # iKronFlux = float(row[45])
            # iKronFluxErr = float(row[46])
            #
            # zKronFlux = float(row[47])
            # zKronFluxErr = float(row[48])
            #
            # yKronFlux = float(row[49])
            # yKronFluxErr = float(row[50])


            # only log unique matches
            if id not in crossmatch_dict:

                ### KRON ###
                # data = (sep, B, synth_B1, synth_B2, diff_1, diff_2, z_dist, gKronMag, gKronMagErr,
                #         rKronMag, rKronMagErr, iKronMag, iKronMagErr, zKronMag, zKronMagErr,
                #         yKronMag, yKronMagErr, gKronFlux, gKronFluxErr, rKronFlux, rKronFluxErr, iKronFlux,
                #         iKronFluxErr, zKronFlux, zKronFluxErr, yKronFlux, yKronFluxErr, z)

                ### PSF ###
                # data = (sep, B, synth_B1, synth_B2, diff_1, diff_2, z_dist, gMeanPSFMag, gMeanPSFMagErr,
                #                         rMeanPSFMag, rMeanPSFMagErr, iMeanPSFMag, iMeanPSFMagErr, zMeanPSFMag, zMeanPSFMagErr, yMeanPSFMag,
                #                         yMeanPSFMagErr, gKronFlux, gKronFluxErr, rKronFlux, rKronFluxErr, iKronFlux,
                #                         iKronFluxErr, zKronFlux, zKronFluxErr, yKronFlux, yKronFluxErr, z)

                ### APPERTURE ###
                data = (sep, B, synth_B1, synth_B2, diff_1, diff_2, z_dist, gApMag, gApMagErr,
                        rApMag, rApMagErr, iApMag, iApMagErr, zApMag, zApMagErr,
                        yApMag, yApMagErr, gKronFlux, gKronFluxErr, rKronFlux, rKronFluxErr, iKronFlux,
                        iKronFluxErr, zKronFlux, zKronFluxErr, yKronFlux, yKronFluxErr, z)

                crossmatch_dict[id] = data

    print("Number of matches: %s" % len(crossmatch_dict))

if write_eazy_input:
    # output into eazy format:
    with open("../PS1_DR2_QueryData/GLADE_crossmatch_eazy.cat", 'w', csv.QUOTE_NONE) as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ')
        csvwriter.writerow(("#", "id",
                            "m_PS1.g", "e_PS1.g",
                            "m_PS1.r", "e_PS1.r",
                            "m_PS1.i", "e_PS1.i",
                            "m_PS1.z", "e_PS1.z",
                            "m_PS1.y", "e_PS1.y",
                            "m_synth_R", "e_synth_R",
                            "z_spec"))

        csvwriter.writerow(("#", "id",
                            "F334", "E334",
                            "F335", "E335",
                            "F336", "E336",
                            "F337", "E337",
                            "F338", "E338",
                            "F338", "E338",
                            "F143", "E143",
                            "z_spec"))

        bad_val = -999
        for id, data in crossmatch_dict.items():

            ## Mags
            g_mag = data[7]
            g_mag_err = data[8]
            r_mag = data[9]
            r_mag_err = data[10]
            i_mag = data[11]
            i_mag_err = data[12]
            z_mag = data[13]
            z_mag_err = data[14]
            y_mag = data[15]
            y_mag_err = data[16]
            z = data[27]

            if g_mag > bad_val and r_mag > bad_val and i_mag > bad_val:
                # Using Jester et al 2005 transformation from SDSS ugriz -> UBVRcIc
                # Combined "All stars with Rc-Ic < 1.15" V-R and V eqns...
                # http://classic.sdss.org/dr5/algorithms/sdssUBVRITransform.html
                R_mag = 0.41 * g_mag - 0.5 * r_mag + 1.09 * i_mag - 0.23
                R_mag_err = np.sqrt((0.41*g_mag)**2 + (0.5*r_mag)**2 + (1.09*i_mag)**2)

                # Using Transformation of Pan-STARRS1 gri to Stetson BVRI magnitudes.
                # https://arxiv.org/pdf/1706.06147.pdf
                # Eq. 4; Table 2
                # R_mag = -0.116 * g_mag + 1.116 * r_mag - 0.142
                # R_mag_err = np.sqrt((0.116 * g_mag) ** 2 + (1.116 * r_mag) ** 2)

                csvwriter.writerow((id, "%0.5E" % g_mag, "%0.5E" % g_mag_err, "%0.5E" % r_mag, "%0.5E" % r_mag_err,
                                    "%0.5E" % i_mag, "%0.5E" % i_mag_err, "%0.5E" % z_mag, "%0.5E" % z_mag_err,
                                    "%0.5E" % y_mag, "%0.5E" % y_mag_err, "%0.5f" % R_mag, "%0.5f" % R_mag_err,
                                    "%0.5f" % z))

    print("Copying over cat file...")
    s.run(["cp", "-v", "../PS1_DR2_QueryData/GLADE_crossmatch_eazy.cat",
           "../../../eazy-photoz/inputs/GLADE_crossmatch_eazy.cat"])


    # fig = plt.figure(figsize=(12, 12), dpi=800)
    # ax1 = fig.add_subplot(221)
    #
    #
    # # ax.hist([d[5] for d in crossmatch_data], color='r', histtype='step', label='g + 0.39 * (g - r) + 0.21')
    # # ax.hist([d[4] for d in crossmatch_data], color='b', histtype='step', label='g + 0.561 * (g - r) + 0.194')
    #
    # sep = [d[0]*3600 for d in crossmatch_data]
    # B = [d[1] for d in crossmatch_data]
    # synth_B1 = [d[2] for d in crossmatch_data]
    # synth_B2 = [d[3] for d in crossmatch_data]
    # residual_B1 = [d[4] for d in crossmatch_data]
    # residual_B2 = [d[5] for d in crossmatch_data]
    #
    # z_dist = [d[6] for d in crossmatch_data]
    # gKron = [d[7] for d in crossmatch_data]
    # rKron = [d[8] for d in crossmatch_data]
    #
    #
    # p1 = pearsonr(B, residual_B2)
    # ax1.plot(B, residual_B2,'.', label="r=%0.4f; p=%0.4E" % (p1[0], p1[1]))
    # ax1.set_xlabel('GLADE B [mag]')
    # ax1.set_ylabel('B - Synthetic B [mag]')
    # ax1.legend()
    #
    # # p2 = pearsonr(z_dist, residual_B2)
    # # ax2 = fig.add_subplot(222)
    # # ax2.plot(z_dist, residual_B2, '.', label="r=%0.4f; p=%0.4E" % (p2[0], p2[1]))
    # # ax2.set_xlabel('Dist [Mpc]')
    # # ax2.set_ylabel('B - Synthetic B [mag]')
    # # ax2.legend()
    #
    # g_r = np.asarray(gKron) - np.asarray(rKron)
    # p2 = pearsonr(g_r, residual_B2)
    # ax2 = fig.add_subplot(222)
    # ax2.plot(g_r, residual_B2, '.', label="r=%0.4f; p=%0.4E" % (p2[0], p2[1]))
    # ax2.set_xlabel('PS1 Kron g - r [mag]')
    # ax2.set_ylabel('B - Synthetic B [mag]')
    # ax2.legend()
    #
    # p3 = pearsonr(gKron, residual_B2)
    # ax3 = fig.add_subplot(223)
    # ax3.plot(gKron, residual_B2, '.', label="r=%0.4f; p=%0.4E" % (p3[0], p3[1]))
    # ax3.set_xlabel('PS1 Kron g')
    # ax3.set_ylabel('B - Synthetic B [mag]')
    # ax3.legend()
    #
    # p4 = pearsonr(rKron, residual_B2)
    # ax4 = fig.add_subplot(224)
    # ax4.plot(rKron  , residual_B2, '.', label="r=%0.4f; p=%0.4E" % (p4[0], p4[1]))
    # ax4.set_xlabel('PS1 Kron r')
    # ax4.set_ylabel('B - Synthetic B [mag]')
    # ax4.legend()
    #
    #
    # fig.savefig("test_crossmatch.png", bbox_inches='tight')
    # print("Done...")
    # plt.close('all')

if do_eazy:
    # s.run(["ls", "-la", "../../../eazy-photoz/inputs/OUTPUT"])
    # default
    # s.run(["../src/eazy", "-p", "zphot.param.default"], cwd="../../../eazy-photoz/inputs")
    # ps1
    s.run(["../src/eazy", "-p", "zphot.param"], cwd="../../../eazy-photoz/inputs")

galaxy_photoz_residual = OrderedDict()
if test_photozs:

    print(len(crossmatch_dict))

    with open("../../../eazy-photoz/inputs/OUTPUT/photz.zout", 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', skipinitialspace=True)
        next(csvreader)  # skip header1
        next(csvreader)  # skip header2

        for row in csvreader:
            id = int(row[0])
            photoz = float(row[11])
            spec_z = crossmatch_dict[id][-1]
            gMeanApMag = crossmatch_dict[id][7]
            rMeanApMag = crossmatch_dict[id][9]
            iMeanApMag = crossmatch_dict[id][11]
            zMeanApMag = crossmatch_dict[id][13]
            yMeanApMag = crossmatch_dict[id][15]

            if spec_z > 0: # GLADE spec z
                galaxy_photoz_residual[id] = (spec_z, photoz, gMeanApMag, rMeanApMag, iMeanApMag, zMeanApMag,
                                              yMeanApMag)

    fig = plt.figure(figsize=(20, 16), dpi=800)
    ax1 = fig.add_subplot(341)

    y = lambda x: x - 0.015
    model_x = np.linspace(0, 0.25, 100)
    model_y = y(model_x)

    ax1.plot(model_x, model_y, color='k', linestyle='--', label=r'$z_{\mathrm{spec}} - 0.015$')

    residuals = np.asarray([value[0]-value[1] for key, value in galaxy_photoz_residual.items()])
    spec_zs = np.asarray([value[0] for key, value in galaxy_photoz_residual.items()])
    gApMags = np.asarray([value[2] for key, value in galaxy_photoz_residual.items()])
    rApMags = np.asarray([value[3] for key, value in galaxy_photoz_residual.items()])
    iApMags = np.asarray([value[4] for key, value in galaxy_photoz_residual.items()])
    zApMags = np.asarray([value[5] for key, value in galaxy_photoz_residual.items()])
    yApMags = np.asarray([value[6] for key, value in galaxy_photoz_residual.items()])

    ax1.plot(spec_zs, residuals, '.', alpha=0.35)
    ax1.vlines([0.0082, 0.0660], -0.3, 0.3, colors='r', linestyles='--')
    ax1.set_xlabel(r'$z_{\mathrm{spec}}$')
    ax1.set_ylabel(r'$z_{\mathrm{spec}} - z_{\mathrm{phot}}$')
    ax1.set_ylim([-0.2, 0.2])
    ax1.text(0.009, -0.05, 'z_min = 0.0082', rotation=90, color='r')
    ax1.text(0.07, -0.05, 'z_max = 0.0660', rotation=90, color='r')
    ax1.grid(color='gray', linestyle=':')
    ax1.legend(loc="lower right")


    model_spec_y = y(spec_zs)
    res_of_res = model_spec_y - residuals

    ax2 = fig.add_subplot(342)
    ax2.plot(gApMags - rApMags, res_of_res, '.', alpha=0.35)
    ax2.set_xlabel('(g - r) Aperture mag')
    ax2.set_ylabel(r'$(z_{\mathrm{spec}}-0.015) - (z_{\mathrm{spec}} - z_{\mathrm{phot}})$')
    ax2.grid(color='gray', linestyle=':')

    ax3 = fig.add_subplot(343)
    ax3.plot(gApMags - iApMags, res_of_res, '.', alpha=0.35)
    ax3.set_xlabel('(g - i) Aperture mag')
    ax3.set_ylabel(r'$(z_{\mathrm{spec}}-0.015) - (z_{\mathrm{spec}} - z_{\mathrm{phot}})$')
    ax3.grid(color='gray', linestyle=':')

    ax4 = fig.add_subplot(344)
    ax4.plot(gApMags - zApMags, res_of_res, '.', alpha=0.35)
    ax4.set_xlabel('(g - z) Aperture mag')
    ax4.set_ylabel(r'$(z_{\mathrm{spec}}-0.015) - (z_{\mathrm{spec}} - z_{\mathrm{phot}})$')
    ax4.grid(color='gray', linestyle=':')

    ax5 = fig.add_subplot(345)
    ax5.plot(gApMags - yApMags, res_of_res, '.', alpha=0.35)
    ax5.set_xlabel('(g - y) Aperture mag')
    ax5.set_ylabel(r'$(z_{\mathrm{spec}}-0.015) - (z_{\mathrm{spec}} - z_{\mathrm{phot}})$')
    ax5.grid(color='gray', linestyle=':')


    # R_mags = 0.41 * gApMags - 0.5 * rApMags + 1.09 * iApMags - 0.23
    ax6 = fig.add_subplot(346)
    ax6.plot(rApMags - iApMags, res_of_res, '.', alpha=0.35)
    ax6.set_xlabel('(r - i) Aperture mag')
    ax6.set_ylabel(r'$(z_{\mathrm{spec}}-0.015) - (z_{\mathrm{spec}} - z_{\mathrm{phot}})$')
    ax6.grid(color='gray', linestyle=':')

    ax7 = fig.add_subplot(347)
    ax7.plot(rApMags - zApMags, res_of_res, '.', alpha=0.35)
    ax7.set_xlabel('(r - z) Aperture mag')
    ax7.set_ylabel(r'$(z_{\mathrm{spec}}-0.015) - (z_{\mathrm{spec}} - z_{\mathrm{phot}})$')
    ax7.grid(color='gray', linestyle=':')

    ax8 = fig.add_subplot(348)
    ax8.plot(rApMags - zApMags, res_of_res, '.', alpha=0.35)
    ax8.set_xlabel('(r - y) Aperture mag')
    ax8.set_ylabel(r'$(z_{\mathrm{spec}}-0.015) - (z_{\mathrm{spec}} - z_{\mathrm{phot}})$')
    ax8.grid(color='gray', linestyle=':')

    ax9 = fig.add_subplot(349)
    ax9.plot(iApMags - zApMags, res_of_res, '.', alpha=0.35)
    ax9.set_xlabel('(i - z) Aperture mag')
    ax9.set_ylabel(r'$(z_{\mathrm{spec}}-0.015) - (z_{\mathrm{spec}} - z_{\mathrm{phot}})$')
    ax9.grid(color='gray', linestyle=':')

    ax10 = fig.add_subplot(3,4,10)
    ax10.plot(iApMags - yApMags, res_of_res, '.', alpha=0.35)
    ax10.set_xlabel('(i - y) Aperture mag')
    ax10.set_ylabel(r'$(z_{\mathrm{spec}}-0.015) - (z_{\mathrm{spec}} - z_{\mathrm{phot}})$')
    ax10.grid(color='gray', linestyle=':')

    ax11 = fig.add_subplot(3, 4, 11)
    ax11.plot(zApMags - yApMags, res_of_res, '.', alpha=0.35)
    ax11.set_xlabel('(z - y) Aperture mag')
    ax11.set_ylabel(r'$(z_{\mathrm{spec}}-0.015) - (z_{\mathrm{spec}} - z_{\mathrm{phot}})$')
    ax11.grid(color='gray', linestyle=':')



    # get residual of z residuals wrt to the line...




    fig.savefig("test_photoz.png", bbox_inches='tight')
    print("Done...")
    plt.close('all')

path_format = "{}/{}"
ps1_strm_dir = "../PS1_DR2_QueryData/PS1_STRM"
if test_curve_fit:

    from scipy.optimize import curve_fit

    def func1(x, a, b, c, d):
    # def func1(x, a, b):
        # return a * np.exp(-b * x) + c
        # return a * np.sqrt(b * (x + c)) + d
        return a * (x-d)**2 + b * (x-d) + c
        # return a * np.log10(b * (x - c)) + d
        # return a * np.log2(b * x) + c
        # return a * x**3 + b*x**2 + c*x
        # return a + b * (x - d)/(1 + c * (x - d))

        # return 1.0 / (1 + np.exp(-b * (x - a)))

    def func2(x, a, b, c):
        # return a * np.exp(-b * x) + c
        # return a * np.log(b * x) + c
        return a * x ** 2 + b * x + c
        # return a * x**3 + b*x**2 + c*x




    _14_0_20_0 = "14.0_20.0"

    _20_0_20_5 = "20.0_20.5"
    _20_5_21_0 = "20.5_21.0"
    _21_0_21_5 = "21.0_21.5"
    _21_5_22_0 = "21.5_22.0"
    Q3 = "Q3"
    Q4 = "Q4"

    osDES_keylist = [
        (_20_0_20_5, Q3, "red", "-", "20.0--20.5"),
        (_20_0_20_5, Q4, "red", "--", ""),

        (_20_5_21_0, Q3, "blue", "-", "20.5--21.0"),
        (_20_5_21_0, Q4, "blue", "--", ""),

        (_21_0_21_5, Q3, "darkgreen", "-", "21.0--21.5"),
        (_21_0_21_5, Q4, "darkgreen", "--", ""),

        (_21_5_22_0, Q3, "khaki", "-", "21.5--22.0"),
        (_21_5_22_0, Q4, "khaki", "--", "")
    ]

    ozDES_data = OrderedDict()
    ozDES_models = OrderedDict()

    # Load all datasets
    for key in osDES_keylist:
        file_path = path_format.format(ps1_strm_dir, "{}_{}.csv".format(key[0], key[1]))

        if key[0] not in ozDES_data:
            ozDES_data[key[0]] = OrderedDict()

        ozDES_data[key[0]][key[1]] = [[], []]

        with open(file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)

            for row in csvreader:
                exp_time = float(row[0])
                success = float(row[1]) / 100.

                ozDES_data[key[0]][key[1]][0].append(exp_time)
                ozDES_data[key[0]][key[1]][1].append(success)

            # exp_times = [0.0] + ozDES_data[key[0]][key[1]][0]
            # successes = [0.0] + ozDES_data[key[0]][key[1]][1]
            exp_times = ozDES_data[key[0]][key[1]][0]
            successes = ozDES_data[key[0]][key[1]][1]


            # Curve fit test
            popt, pcov = None, None
            sigmas1 = [0.05, 0.1, 0.20, 0.20]
            sigmas2 = [0.05, 0.1, 0.20]
            if len(exp_times) == 3:
                popt, pcov = curve_fit(func2, exp_times[0:3], successes[0:3], sigma=sigmas2)
            else:
                popt, pcov = curve_fit(func1, exp_times[0:4], successes[0:4], sigma=sigmas1)

            # Get slope of first two points...
            # m = (successes[1] - successes[0]) / (exp_times[1] - exp_times[0])
            # y_0 = successes[0] - m * exp_times[0]
            # model = interp1d([0] + exp_times, [y_0] + successes)

            if key[0] not in ozDES_models:
                ozDES_models[key[0]] = OrderedDict()

            # ozDES_models[key[0]][key[1]] = model
            ozDES_models[key[0]][key[1]] = popt


    fig = plt.figure(figsize=(8, 6), dpi=800)
    ax1 = fig.add_subplot(111)

    model_time = np.linspace(40, 300, 500)
    for key in osDES_keylist:

        if key[1] == Q4:
            continue

        # model = ozDES_models[key[0]][key[1]]
        model_popt = ozDES_models[key[0]][key[1]]

        exp_time = ozDES_data[key[0]][key[1]][0]
        success = np.asarray(ozDES_data[key[0]][key[1]][1]) * 100.0

        # find time to >= 80%
        # model_success = model(model_time)
        model_success = None
        if len(success) == 3:
            model_success = func2(model_time, *model_popt)
        else:
            model_success = func1(model_time, *model_popt)

        # thresh = 0.75
        # thresh_txt = thresh * 100
        # time_for_bin_to_thresh_success = next(model_time[i] for i, s in enumerate(model_success) if s >= thresh)

        addendum = ""
        # if key[0] == _20_0_20_5:
        #     pass
        # elif key[0] == _20_5_21_0:
        #     addendum += "; 80 min => %0.0f%%" % float(model(80.0) * 100.0)
        # elif key[0] == _21_0_21_5:
        #     addendum += "; 120 min => %0.0f%%" % float(model(120.0) * 100.0)
        # elif key[0] == _21_5_22_0:
        #     addendum += "; 160 min => %0.0f%%" % float(model(160.0) * 100.0)

        # Plot data
        if key[4] != "":
            # lbl = key[4] + "; %0.0f min => %0.0f%%" % (time_for_bin_to_thresh_success, thresh_txt) + addendum
            # ax1.plot(exp_time, success, color=key[2], linestyle=key[3], label=lbl, alpha=1.0)
            lbl=""
            ax1.plot(exp_time, success, color=key[2], marker=".", label=lbl, alpha=1.0)
        else:
            # ax1.plot(exp_time, success, color=key[2], linestyle=key[3], alpha=1.0)
            ax1.plot(exp_time, success, color=key[2], marker=".", alpha=1.0)

        # plot model -- only good until 250 seconds
        ax1.plot(model_time, model_success * 100, color=key[2], linestyle='--', alpha=0.3)

    ax1.vlines(40.0, 30, 103, colors='k', linestyles='--', label="Min ExpTime: 40 min")

    ax1.set_xlim([0, 1600])
    ax1.set_ylim([30, 103])
    ax1.set_xlabel("Exposure time (minutes)")
    ax1.set_ylabel("Redshift Completeness (%)")
    ax1.legend(loc="lower right", mode='expand', labelspacing=2.0, bbox_to_anchor=(0.35, 0., 0.65, 0.5))
    fig.savefig(path_format.format(ps1_strm_dir, "OzDES_Fig5.png"), bbox_inches='tight')
    plt.close('all')

if read_region_poly:

    path_format = "{}/{}"
    ps1_strm_dir = "../PS1_DR2_QueryData/PS1_STRM"

    # region load ozDES
    ozDES_data = OrderedDict()
    ozDES_models = OrderedDict()
    _20_0_20_5 = "20.0_20.5"
    _20_5_21_0 = "20.5_21.0"
    _21_0_21_5 = "21.0_21.5"
    _21_5_22_0 = "21.5_22.0"
    Q3 = "Q3"
    Q4 = "Q4"

    osDES_keylist = [
        (_20_0_20_5, Q3, "red", "-", "20.0--20.5"),
        (_20_0_20_5, Q4, "red", "--", ""),

        (_20_5_21_0, Q3, "blue", "-", "20.5--21.0"),
        (_20_5_21_0, Q4, "blue", "--", ""),

        (_21_0_21_5, Q3, "darkgreen", "-", "21.0--21.5"),
        (_21_0_21_5, Q4, "darkgreen", "--", ""),

        (_21_5_22_0, Q3, "khaki", "-", "21.5--22.0"),
        (_21_5_22_0, Q4, "khaki", "--", "")
    ]

    # Load all datasets
    for key in osDES_keylist:
        file_path = path_format.format(ps1_strm_dir, "{}_{}.csv".format(key[0], key[1]))

        if key[0] not in ozDES_data:
            ozDES_data[key[0]] = OrderedDict()

        ozDES_data[key[0]][key[1]] = [[], []]

        with open(file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)

            for row in csvreader:
                exp_time = float(row[0])
                success = float(row[1]) / 100.

                ozDES_data[key[0]][key[1]][0].append(exp_time)
                ozDES_data[key[0]][key[1]][1].append(success)

            exp_times = ozDES_data[key[0]][key[1]][0]
            successes = ozDES_data[key[0]][key[1]][1]

            # Assume it stays linear to min exp time. Get slope of first two points...
            m = (successes[1] - successes[0]) / (exp_times[1] - exp_times[0])
            y_0 = successes[0] - m * exp_times[0]
            model = interp1d([0] + exp_times, [y_0] + successes)

            if key[0] not in ozDES_models:
                ozDES_models[key[0]] = OrderedDict()

            ozDES_models[key[0]][key[1]] = model

    def get_osDES_model(kron_r_mag):
        model = ozDES_models[_20_0_20_5][Q3]
        required_exps = 1

        if r >= 20.5 and r < 21.0:
            model = ozDES_models[_20_5_21_0][Q3]
            required_exps = 2
        elif r >= 21.0 and r < 21.5:
            model = ozDES_models[_21_0_21_5][Q3]
            required_exps = 3
        elif r >= 21.5:
            model = ozDES_models[_21_5_22_0][Q3]
            required_exps = 4

        return required_exps, model
    # endregion


    map_nside = 1024

    # region Spin up AAOmega Pix and Galaxies
    northern_95th_AAOmega_galaxy_dict_by_galaxy_id = {}
    northern_95th_AAOmega_galaxy_dict_by_pixel_index = {}
    # Get galaxies in north
    good_candidate_ps1_galaxy_select = '''
            SELECT
                ps1.id as Galaxy_id, 
                ps1.gaia_ra,
                ps1.gaia_dec,
                ps1.rMeanKronMag,
                ps.z_phot0,
                ps.prob_Galaxy,
                hp.id as Pixel_id,
                hp.Pixel_Index,
                ps.z_photErr,
                ps1.synth_B2
            FROM 
                PS1_Galaxy ps1
            JOIN HealpixPixel_PS1_Galaxy hp_ps1 on hp_ps1.PS1_Galaxy_id = ps1.id
            JOIN PS1_STRM ps on ps.uniquePspsOBid = ps1.uniquePspsOBid
            JOIN HealpixPixel hp on hp.id = hp_ps1.HealpixPixel_id
            WHERE 
                ps1.GoodCandidate = 1 
            '''
    northern_95th_galaxy_result = query_db([good_candidate_ps1_galaxy_select])[0]
    print("Returned %s galaxies" % len(northern_95th_galaxy_result))
    for g in northern_95th_galaxy_result:
        galaxy_id = int(g[0])
        ra = float(g[1])
        dec = float(g[2])
        r = float(g[3])
        z = float(g[4])
        prob_Galaxy = float(g[5])
        pix_id = int(g[6])
        pix_index = int(g[7])

        z_err = float(g[8])
        synth_B = float(g[9])

        required_exps, osDES_model = get_osDES_model(r)
        aaomega_galaxy = AAOmega_Galaxy(galaxy_id, ra, dec, prob_Galaxy, z, r, pix_index, required_exps, osDES_model,
                                        z_photErr=z_err, synth_B=synth_B)

        if pix_index not in northern_95th_AAOmega_galaxy_dict_by_pixel_index:
            northern_95th_AAOmega_galaxy_dict_by_pixel_index[pix_index] = {}

        northern_95th_AAOmega_galaxy_dict_by_pixel_index[pix_index][galaxy_id] = aaomega_galaxy
        northern_95th_AAOmega_galaxy_dict_by_galaxy_id[galaxy_id] = aaomega_galaxy

    northern_AAOmega_pixel_dict = {}
    northern_95th_pixel_select = '''
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
        FROM HealpixPixel 
        WHERE id IN (%s) 
    '''
    northern_95th_pixel_ids = []
    northern_95th_pixel_ids_file = path_format.format(ps1_strm_dir, "northern_95th_pixel_ids.txt")
    with open(northern_95th_pixel_ids_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        next(csvreader)  # skip header
        for row in csvreader:
            id = row[0]
            northern_95th_pixel_ids.append(id)
    northern_95th_pixel_result = query_db([northern_95th_pixel_select % ",".join(northern_95th_pixel_ids)])[0]
    print("Total NSIDE=1024 pixels in Northern 95th: %s" % len(northern_95th_pixel_result))
    for m in northern_95th_pixel_result:
        pix_id = int(m[0])
        index = int(m[2])
        prob = float(m[3])
        dist = float(m[7])
        stddev = float(m[8])

        contained_galaxies_dict = {}
        if index in northern_95th_AAOmega_galaxy_dict_by_pixel_index:
            contained_galaxies_dict = northern_95th_AAOmega_galaxy_dict_by_pixel_index[index]

            gal_count = len(contained_galaxies_dict)
            for gal_id, gal in contained_galaxies_dict.items():
                gal.prob_fraction = prob/gal_count

        p = AAOmega_Pixel(index, map_nside, prob, contained_galaxies_dict, pixel_id=pix_id, mean_dist=dist, stddev_dist=stddev)
        northern_AAOmega_pixel_dict[index] = p











    _50_AAOmega_pixel_dict = {}
    _50th_pixel_select = '''
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
             FROM HealpixPixel 
             WHERE id IN (%s) 
         '''
    _50th_pixel_ids = []
    _50th_pixel_ids_file = path_format.format(ps1_strm_dir, "50th_pixel_ids.txt")
    with open(_50th_pixel_ids_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        next(csvreader)  # skip header
        for row in csvreader:
            id = row[0]
            _50th_pixel_ids.append(id)

    _50th_pixel_result = query_db([_50th_pixel_select % ",".join(_50th_pixel_ids)])[0]
    print("Total NSIDE=1024 pixels in 50th: %s" % len(_50th_pixel_result))
    for m in _50th_pixel_result:
        pix_id = int(m[0])
        index = int(m[2])
        prob = float(m[3])
        dist = float(m[7])
        stddev = float(m[8])

        _50_AAOmega_pixel_dict[index] = (pix_id, index, prob, dist, stddev)











    southern_AAOmega_pixel_dict = {}
    southern_95th_pixel_select = '''
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
        FROM HealpixPixel
        WHERE id IN (%s)
    '''
    southern_95th_pixel_ids = []
    southern_95th_pixel_ids_file = path_format.format(ps1_strm_dir, "southern_95th_pixel_ids.txt")
    with open(southern_95th_pixel_ids_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        next(csvreader)  # skip header
        for row in csvreader:
            id = row[0]
            southern_95th_pixel_ids.append(id)
    southern_95th_pixel_result = query_db([southern_95th_pixel_select % ",".join(southern_95th_pixel_ids)])[0]
    print("Total NSIDE=1024 pixels in Southern 95th: %s" % len(southern_95th_pixel_result))
    for m in southern_95th_pixel_result:
        pix_id = int(m[0])
        index = int(m[2])
        prob = float(m[3])
        dist = float(m[7])
        stddev = float(m[8])

        p = AAOmega_Pixel(index, map_nside, prob, contained_galaxies={}, pixel_id=pix_id, mean_dist=dist,
                          stddev_dist=stddev)
        southern_AAOmega_pixel_dict[index] = p

    count_south = len(southern_AAOmega_pixel_dict)
    print("Count south: %s" % count_south)

    for s_index, s_pix in southern_AAOmega_pixel_dict.items():
        j = random.randint(0, len(northern_AAOmega_pixel_dict) - 1)
        n_pix_index = list(northern_AAOmega_pixel_dict.keys())[j]
        p = northern_AAOmega_pixel_dict[n_pix_index]

        fake_galaxies = {}
        for g_id, g in p.contained_galaxies.items():
            fake_g_id = g_id+100000
            a = AAOmega_Galaxy(fake_g_id, p.coord.ra.degree, p.coord.dec.degree, g.prob_galaxy, g.z, g.kron_r,
                               s_pix.index, g.required_exps, g.efficiency_func)

            fake_galaxies[fake_g_id] = a

        gal_count = len(fake_galaxies)
        for gal_id, gal in fake_galaxies.items():
            gal.prob_fraction = s_pix.prob / gal_count
        s_pix.contained_galaxies = fake_galaxies

    northern_AAOmega_pixel_dict.update(southern_AAOmega_pixel_dict)



    # endregion

    # get AAOmega Static grid to do galaxy demographics
    # select_aaomega_static_grid = '''
    #     SELECT
    #         st.RA, st._Dec, d.Deg_radius
    #     FROM StaticTile st
    #     JOIN Detector d on d.id = st.Detector_id
    #     WHERE d.id = 8
    # '''
    # static_grid_result = query_db([select_aaomega_static_grid])[0]
    # static_grid_tiles = []
    # for sg in static_grid_result:
    #     ra = float(sg[0])
    #     dec = float(sg[1])
    #     radius = float(sg[2])
    #
    #     static_grid_tiles.append(AAOmega_Tile(ra, dec, map_nside, radius,
    #                                       northern_AAOmega_pixel_dict, num_exposures=1, tile_num=1))







    # Load observation plan from region file and create AAOmega tiles
    # filename = 'test2.reg'
    filename = 'Obs10.reg'
    filepath = path_format.format(ps1_strm_dir, filename)
    localization_poly = [] # used to hold polygons from regions


    aaomega_detector = Detector("2dF", detector_width_deg=None, detector_height_deg=None, detector_radius_deg=1.05)
    aaomega_tiles = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

        for l in lines:
            if "circle(" in l:
                # EX: circle(0:52:11.133, -25:43:00.494, 3780.000") # color=red
                tup_str = l.split("{")[1].split("}")[0]
                tile_tup = tup_str.split(",")
                tile_num = int(tile_tup[0])
                exp_num = int(tile_tup[1])

                tokens = l.replace("circle(", "").replace("\") # color=red text={%s}\n" % tup_str, "").split(",")
                ra = tokens[0].strip()
                dec = tokens[1].strip()



                c = coord.SkyCoord(ra, dec, unit=(u.hour, u.deg))



                radius_deg = float(tokens[2])/3600. # input in arcseconds

                a = AAOmega_Tile(c.ra.degree, c.dec.degree, map_nside, radius_deg,
                                                  northern_AAOmega_pixel_dict, num_exposures=exp_num,
                                                  tile_num=tile_num)

                if len(a.contained_pixels_dict) > 0:

                    print("RA Hour: %s; Dec Deg: %s" % (c.ra.hms[0], c.dec.dms[0]))

                    aaomega_tiles.append(a)

            elif "polygon" in l:

                l_str = l
                if "color=cyan" in l:
                    l_str = l_str.replace(" # color=cyan\n", "")
                elif "color=red" in l:
                    l_str = l_str.replace(" # color=red\n", "")
                else:
                    continue

                poly_vertices = []

                # EX: polygon(1:28:03.984,-33:49:25.649, ... 1:28:14.531,-33:46:44.047)
                tokens = l_str.replace("polygon(", "").replace(")", "").split(",")

                i = 0
                while i < len(tokens):
                    ra = tokens[i]
                    dec = tokens[i + 1]
                    c = coord.SkyCoord(ra, dec, unit=(u.hour, u.deg))
                    poly_vertices.append([c.ra.degree, c.dec.degree])
                    i += 2

                localization_poly.append(SQL_Polygon([geometry.Polygon(poly_vertices)], aaomega_detector))


    # DEBUG
    # For Ryan, serialize the per tile galaxy information for the sample.
    # for i, t in enumerate(aaomega_tiles):
    #
    #     flattened_galaxies = []
    #
    #     for pix_index, pix in t.contained_pixels_dict.items():
    #         for gal_id, gal in pix.contained_galaxies.items():
    #             flattened_galaxies.append((gal.galaxy_id, gal.z, gal.z_photErr))
    #
    #     if len(flattened_galaxies) > 0:
    #         with open("../PS1_DR2_QueryData/AAOmegaTiles/tile_%s.txt" % i, 'w') as csvfile:
    #             csvwriter = csv.writer(csvfile, delimiter=',')
    #             csvwriter.writerow(("galaxy_id", "phot_z", "phot_z_err"))
    #
    #             for f in flattened_galaxies:
    #                 csvwriter.writerow(f)

    # raise Exception("Stop")



    total_exps = 0
    total_slews = 0
    sortedTiles = sorted(aaomega_tiles, key=lambda x: x.tile_num)
    all_gals, all_prob = 0, 0
    all_gal_ids = []
    print("Tile Num\tNum Exp\t\tGalaxies\tNet Prob")
    for t in sortedTiles:

        total_slews += 1
        total_exps += t.num_exposures

        gals, prob, gal_ids = t.calculate_efficiency()

        all_gal_ids += gal_ids
        all_gals += gals
        all_prob += prob
        print("%s\t\t\t%s\t\t\t%s\t\t\t%0.6f" % (t.tile_num, t.num_exposures, gals, prob))
        # print("%s\t\t\t%s\t\t\t%s\t\t\t%0.8f" % (t.tile_num, t.num_exposures, gals, prob))

    print("****************\n")
    print("Total Galaxies %s/%s" % (all_gals,
                                    np.sum([len(p.contained_galaxies)
                                            for pi, p in northern_AAOmega_pixel_dict.items()])))
    print("Total Prob: %0.2f" % all_prob)
    total_hours = (total_exps * 40 + total_slews * 15) / 60.
    total_nights = total_hours / 10.0
    print("Total Hours: %0.2f; Total 11-hour nights: %0.2f" % (total_hours, total_nights))


    print("\n")
    print("\n")
    print("\n")

    # #### Figuring out the percentage of the full sample that we think we can get by percentile membership
    # h = 0.7
    # phi = 1.6e-2 * h ** 3  # +/- 0.3 Mpc^-3
    # a = -1.07  # +/- 0.07
    # L_B_star = 1.2e+10 / h ** 2  # +/- 0.1
    #
    # _95_denominator_gals = [g for g_id, g in northern_95th_AAOmega_galaxy_dict_by_galaxy_id.items()]
    # _50_denominator_gals = []
    # for g in _95_denominator_gals:
    #     if g.pix_index in _50_AAOmega_pixel_dict:
    #         _50_denominator_gals.append(g)
    #
    #
    # _95_numerator_gals = []
    # _50_numerator_gals = []
    #
    # for gi in all_gal_ids:
    #     gal = northern_95th_AAOmega_galaxy_dict_by_galaxy_id[gi]
    #
    #     if gal.pix_index in _50_AAOmega_pixel_dict:
    #         _50_numerator_gals.append(gal)
    #
    #     if gal.pix_index in northern_95th_AAOmega_galaxy_dict_by_pixel_index:
    #         _95_numerator_gals.append(gal)
    #
    # _95th_denominator_luminosities = []
    # for g in _95_denominator_gals:
    #     synth_B2 = g.synth_B
    #     z_dist = cosmo.luminosity_distance(g.z).value
    #
    #     L_Sun__L_star = 10 ** (-0.4 * ((synth_B2 - (5 * np.log10(z_dist * 1e+6) - 5)) - 5.48)) / L_B_star
    #     _95th_denominator_luminosities.append(L_Sun__L_star)
    #
    # _50th_denominator_luminosities = []
    # for g in _50_denominator_gals:
    #     synth_B2 = g.synth_B
    #     z_dist = cosmo.luminosity_distance(g.z).value
    #
    #     L_Sun__L_star = 10 ** (-0.4 * ((synth_B2 - (5 * np.log10(z_dist * 1e+6) - 5)) - 5.48)) / L_B_star
    #     _50th_denominator_luminosities.append(L_Sun__L_star)
    #
    #
    # _95th_numerator_luminosities = []
    # for g in _95_numerator_gals:
    #     synth_B2 = g.synth_B
    #     z_dist = cosmo.luminosity_distance(g.z).value
    #
    #     L_Sun__L_star = 10 ** (-0.4 * ((synth_B2 - (5 * np.log10(z_dist * 1e+6) - 5)) - 5.48)) / L_B_star
    #     _95th_numerator_luminosities.append(L_Sun__L_star)
    #
    # _50th_numerator_luminosities = []
    # for g in _50_numerator_gals:
    #     synth_B2 = g.synth_B
    #     z_dist = cosmo.luminosity_distance(g.z).value
    #
    #     L_Sun__L_star = 10 ** (-0.4 * ((synth_B2 - (5 * np.log10(z_dist * 1e+6) - 5)) - 5.48)) / L_B_star
    #     _50th_numerator_luminosities.append(L_Sun__L_star)
    #
    # y_95_denominator, binEdges_95_denominator = np.histogram(np.log10(_95th_denominator_luminosities), bins=np.linspace(-6.0, 2.0, 45))
    # bincenters_95_denominator = 0.5 * (binEdges_95_denominator[1:] + binEdges_95_denominator[:-1])
    #
    # y_50_denominator, binEdges_50_denominator = np.histogram(np.log10(_50th_denominator_luminosities), bins=np.linspace(-6.0, 2.0, 45))
    # bincenters_50_denominator = 0.5 * (binEdges_50_denominator[1:] + binEdges_50_denominator[:-1])
    #
    # y_95_numerator, binEdges_95_numerator = np.histogram(np.log10(_95th_numerator_luminosities), bins=np.linspace(-6.0, 2.0, 45))
    # bincenters_95_numerator = 0.5 * (binEdges_95_numerator[1:] + binEdges_95_numerator[:-1])
    #
    # y_50_numerator, binEdges_50_numerator = np.histogram(np.log10(_50th_numerator_luminosities), bins=np.linspace(-6.0, 2.0, 45))
    # bincenters_50_numerator = 0.5 * (binEdges_50_numerator[1:] + binEdges_50_numerator[:-1])
    #
    # _95_comp_bins = []
    # _50_comp_bins = []
    # _95_comp_ratios = []
    # _50_comp_ratios = []
    #
    # log_LB_LStar_efficiency_thresh = -1.3439626432899594
    # for i, bs in enumerate(bincenters_95_denominator):
    #     _95_denominator_interval = y_95_denominator[i]
    #     _50_denominator_interval = y_50_denominator[i]
    #     _95_numerator_interval = y_95_numerator[i]
    #     _50_numerator_interval = y_50_numerator[i]
    #
    #     _95_comp_ratio = 0
    #     if _95_denominator_interval > 0 and _95_numerator_interval > 0:
    #         if bs < log_LB_LStar_efficiency_thresh:
    #             _95_comp_ratio = (0.7*_95_numerator_interval) / _95_denominator_interval
    #         else:
    #             _95_comp_ratio = _95_numerator_interval / _95_denominator_interval
    #
    #     _50_comp_ratio = 0
    #     if _50_denominator_interval > 0 and _50_numerator_interval > 0:
    #         if bs < log_LB_LStar_efficiency_thresh:
    #             _50_comp_ratio = (0.7*_50_numerator_interval) / _50_denominator_interval
    #         else:
    #             _50_comp_ratio = _50_numerator_interval / _50_denominator_interval
    #
    #
    #     if _95_denominator_interval > 0 and bs > -5:
    #         _95_comp_bins.append(bs)
    #         _95_comp_ratios.append(_95_comp_ratio)
    #     elif _95_denominator_interval == 0 and bs <= -5:
    #         _95_comp_bins.append(bs)
    #         _95_comp_ratios.append(_95_comp_ratio)
    #
    #     if _50_denominator_interval > 0 and bs > -4:
    #         _50_comp_bins.append(bs)
    #         _50_comp_ratios.append(_50_comp_ratio)
    #     elif _50_denominator_interval == 0 and bs <= -4:
    #         _50_comp_bins.append(bs)
    #         _50_comp_ratios.append(_50_comp_ratio)
    #
    # from scipy.signal import savgol_filter
    #
    # x1 = np.linspace(np.min(_50_comp_bins), np.max(_50_comp_bins), 100)
    # itp1 = interp1d(_50_comp_bins, _50_comp_ratios, kind='linear')
    # window_size1, poly_order1 = 5, 3
    # # window_size, poly_order = 3, 2
    # yy_sg1 = savgol_filter(itp1(x1), window_size1, poly_order1)
    #
    # x2 = np.linspace(np.min(_95_comp_bins), np.max(_95_comp_bins), 100)
    # itp2 = interp1d(_95_comp_bins, _95_comp_ratios, kind='linear')
    # # window_size2, poly_order2 = 7, 5
    # window_size2, poly_order2 = 7, 3
    # # window_size2, poly_order2 = 21, 3
    #
    # yy_sg2 = savgol_filter(itp2(x2), window_size2, poly_order2)
    #
    #
    # fig = plt.figure(figsize=(10, 10), dpi=800)
    # ax = fig.add_subplot(111)
    #
    # # ax.plot(_95_comp_bins, _95_comp_ratios, '+', color='red')
    # # ax.plot(_50_comp_bins, _50_comp_ratios, '+', color='red')
    # ax.plot(x1, yy_sg1, '-', color='orange')
    #
    # ax.plot(x2, yy_sg2, '-', color='green')
    #
    # ax.set_xlim([-6.0, 2.0])
    # ax.set_xticks([-6, -5, -4, -3, -2, -1, 0, 1])
    # ax.set_xlabel(r"Log($L_B/L^{*}_{B}$)", fontsize=24)
    # # ax.set_ylabel(r"Log($N$)", fontsize=24)
    # ax.tick_params(axis='both', which='major', labelsize=24, length=8.0, width=2)
    # fig.savefig("PS1_GW190814_Imaged_vs_Total.png", bbox_inches='tight')
    # plt.close('all')
    #
    #
    # raise Exception("Stop")




    # Plot regions as a test
    fig = plt.figure(figsize=(10, 10), dpi=1000)
    ax = fig.add_subplot(111)

    m = Basemap(projection='stere',
                lon_0=15.0,
                lat_0=-20.0,
                llcrnrlat=-35.5,
                urcrnrlat=-19.5,
                llcrnrlon=8.0,
                urcrnrlon=25.0)


    # gal_weighted_prob = []
    # for pix_index, p in northern_AAOmega_pixel_dict.items():
    #     net_prob = 0.0
    #     for gi, g in p.contained_galaxies.items():
    #         net_prob += g.prob_fraction
    #     gal_weighted_prob.append(net_prob)
    all_prob = []
    for pix_index, p in northern_AAOmega_pixel_dict.items():
        all_prob.append(p.prob)

    min_gal_weighted_prob = np.min(all_prob)
    max_gal_weighted_prob = np.max(all_prob)
    n = colors.Normalize(min_gal_weighted_prob, max_gal_weighted_prob)

    # max_galaxies_per_pix = -999
    # min_galaxies_per_pix = 999
    # max_galaxies_per_pix = 30
    # min_galaxies_per_pix = 0
    # for pix_index, p in northern_AAOmega_pixel_dict.items():
    #     count = len(p.contained_galaxies)
    #     if count < min_galaxies_per_pix:
    #         min_galaxies_per_pix = count
    #     if count > max_galaxies_per_pix:
    #         max_galaxies_per_pix = count
    # n = colors.Normalize(min_galaxies_per_pix, max_galaxies_per_pix)
    # print(min_galaxies_per_pix, max_galaxies_per_pix)

    for pix_index, p in northern_AAOmega_pixel_dict.items():
        # count = len(p.contained_galaxies)
        # clr = plt.cm.Greys(n(10))
        # if count == 0:
        #     clr = plt.cm.Greys(n(0))
        # elif count > 0 and count < 10:
        #     clr = plt.cm.Greys(n(5))
        # elif count >= 10 and count < 20:
        #     clr = plt.cm.Greys(n(15))
        # elif count >= 20 and count < 30:
        #     clr = plt.cm.Greys(n(25))
        # elif count >= 30:
        #     clr = plt.cm.Greys(n(30))



        clr = plt.cm.Greys(n(p.prob))
        p.plot(m, ax, edgecolor="None", facecolor=clr, linewidth="1.0", alpha=1.0)

    for l in localization_poly:
        l.plot(m, ax, edgecolor='black', linewidth=1.5, facecolor='None')

    for t in aaomega_tiles:
        # clr = "green"
        # if t.num_exposures == 2:
        #     clr = "blue"
        # elif t.num_exposures == 3:
        #     clr = "red"

        if t.num_exposures == 3:
            t.plot(m, ax, edgecolor="red", facecolor="None", linewidth="2.0")

    for t in aaomega_tiles:

        if t.num_exposures == 2:
            t.plot(m, ax, edgecolor="blue", facecolor="None", linewidth="2.0")





    # region Draw axes
    # draw parallels.
    sm_label_size = 18
    label_size = 28
    title_size = 36

    _90_x1 = 0.77
    _90_y1 = 0.558

    _90_x2 = 0.77
    _90_y2 = 0.40

    _90_text_y = 0.37
    _90_text_x = 0.32

    _50_x1 = 0.60
    _50_y1 = 0.51

    _50_x2 = 0.48
    _50_y2 = 0.40

    _50_text_y = 0.37
    _50_text_x = 0.64

    parallels = list(np.arange(-90., 90., 10.))
    dec_ticks = m.drawparallels(parallels, labels=[0, 1, 0, 0])
    for i, tick_obj in enumerate(dec_ticks):
        a = coord.Angle(tick_obj, unit=u.deg)

        for text_obj in dec_ticks[tick_obj][1]:
            direction = '+' if a.dms[0] > 0.0 else '-'
            text_obj.set_text(r'${0}{1:0g}^{{\degree}}$'.format(direction, np.abs(a.dms[0])))
            text_obj.set_size(sm_label_size)
            x = text_obj.get_position()[0]

            new_x = x * (1.0 + 0.08)
            text_obj.set_x(new_x)

    # draw meridians
    meridians = np.arange(0., 360., 7.5)
    ra_ticks = m.drawmeridians(meridians, labels=[0, 0, 0, 1])

    RA_label_dict = {
        7.5: r'$00^{\mathrm{h}}30^{\mathrm{m}}$',
        15.0: r'$01^{\mathrm{h}}00^{\mathrm{m}}$',
        22.5: r'$01^{\mathrm{h}}30^{\mathrm{m}}$',
    }

    for i, tick_obj in enumerate(ra_ticks):
        for text_obj in ra_ticks[tick_obj][1]:
            if tick_obj in RA_label_dict:
                text_obj.set_text(RA_label_dict[tick_obj])
                text_obj.set_size(sm_label_size)


    sm = plt.cm.ScalarMappable(norm=n, cmap=plt.cm.Greys)
    sm.set_array([])  # can be an empty list

    tks = np.linspace(min_gal_weighted_prob, max_gal_weighted_prob, 5)
    # tks = np.logspace(np.log10(min_prob), np.log10(max_prob), 11)
    tks_strings = []

    for t in tks:
        tks_strings.append('%0.2f' % (t * 100))

    cb = fig.colorbar(sm, ax=ax, ticks=tks, orientation='vertical', fraction=0.04875, pad=0.02,
                      alpha=0.80)  # 0.08951
    cb.ax.set_yticklabels(tks_strings, fontsize=16)
    cb.set_label("2D Pixel Probability", fontsize=label_size, labelpad=9.0)

    cb.ax.tick_params(width=2.0, length=6.0)

    cb.outline.set_linewidth(2.0)




    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)

    ax.invert_xaxis()

    plt.ylabel(r'$\mathrm{Declination}$', fontsize=label_size, labelpad=36)
    plt.xlabel(r'$\mathrm{Right\;Ascension}$', fontsize=label_size, labelpad=30)
    # endregion

    fig.savefig('GW190814_PS1_region_test.png', bbox_inches='tight')  # ,dpi=840
    plt.close('all')
    print("... Done.")

if h0_calc_test_single:

    start = time.time()

    northern_95th_pixel_ids = []
    northern_95th_pixel_ids_file = path_format.format(ps1_strm_dir, "northern_95th_pixel_ids.txt")
    with open(northern_95th_pixel_ids_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        next(csvreader)  # skip header
        for row in csvreader:
            id = row[0]
            northern_95th_pixel_ids.append(id)

    H0 = 70 # km/s/Mpc
    c = 2.998e+5 # km/s

    D0 = 267
    D0_err = 52
    D0_dist = norm(loc=D0, scale=D0_err)

    # z bounds given D0 and H0
    z_min = (H0 * (D0 - D0_err))/c
    z_max = (H0 * (D0 + D0_err))/c
    print("\nz bounds [%s, %s]\n" % (z_min, z_max))

    get_from_db = False
    match_galaxies = []
    if get_from_db:
        id_str = ",".join([str(i) for i in northern_95th_pixel_ids])
        select_galaxies_in_volume = '''
        SELECT 
            g.id as gd2_id, 
            g.RA, 
            g._Dec, 
            g.z,
            g.z_dist,
            g.B,
            p.ps1_galaxy_id,
            p.gaia_ra,
            p.gaia_dec,
            p.synth_B1,
            p.synth_B2,
            p.z_phot0,
            p.z_photErr,
            p.N128_SkyPixel_id as ps1_N128_SkyPixel_id,
            AngSep(g.RA, g._Dec, p.gaia_ra, p.gaia_dec) AS sep
        FROM 
            (SELECT 
                gd2.id,
                gd2.RA,
                gd2._Dec,
                gd2.z,
                gd2.z_dist,
                gd2.B
            FROM 
                GalaxyDistance2 gd2
            JOIN 
                HealpixPixel_GalaxyDistance2 hp_gd2 on hp_gd2.GalaxyDistance2_id = gd2.id
            JOIN 
                HealpixPixel hp on hp.id = hp_gd2.HealpixPixel_id
            WHERE 
                hp.HealpixMap_id = 2 AND
                gd2.flag2 IN (1,3) AND 
                hp.id IN (%s)) AS g
        JOIN 
            (SELECT 
                ps1.id as ps1_galaxy_id,
                ps1.gaia_ra,
                ps1.gaia_dec,
                ps1.synth_B1,
                ps1.synth_B2,
                ps.z_phot0,
                ps.z_photErr,
                ps1.N128_SkyPixel_id
            FROM 
                PS1_Galaxy ps1 
            JOIN 
                PS1_STRM ps on ps.uniquePspsOBid = ps1.uniquePspsOBid
            JOIN 
                HealpixPixel_PS1_Galaxy hp_ps1 on hp_ps1.PS1_Galaxy_id = ps1.id
            JOIN 
                HealpixPixel hp on hp.id = hp_ps1.HealpixPixel_id
            WHERE 
                hp.HealpixMap_id = 2 AND
                ps.z_phot0 <> -999 AND
                hp.id IN (%s)) AS p 
        ON AngSep(g.RA, g._Dec, p.gaia_ra, p.gaia_dec)*3600 <= 1.0
        WHERE g.z BETWEEN %s AND %s 
    '''

        galaxy_result = query_db([select_galaxies_in_volume % (id_str, id_str, z_min, z_max)])[0]

        with open(path_format.format(ps1_strm_dir, "GLADE_matched_with_PS1_Northern_95th.txt"), 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(("gd2_id", "glade_ra", "glade_dec", "z", "z_dist", "B", "ps1_galaxy_id", "gaia_ra",
                                "gaia_dec", "synth_B1", "synth_B2", "z_phot0", "z_photErr", "N128_SkyPixel_id", "sep"))
            for g in galaxy_result:
                gd2_id = int(g[0])
                glade_ra = float(g[1])
                glade_dec = float(g[2])
                z = float(g[3])
                z_dist = float(g[4])
                B = float(g[5])
                ps1_galaxy_id = int(g[6])
                gaia_ra = float(g[7])
                gaia_dec = float(g[8])
                synth_B1 = float(g[9])
                synth_B2 = float(g[10])
                z_phot0 = float(g[11])
                z_photErr = float(g[12])
                N128_SkyPixel_id = int(g[13])
                sep = float(g[14])

                match_galaxies.append((gd2_id, glade_ra, glade_dec, z, z_dist, B, ps1_galaxy_id, gaia_ra, gaia_dec,
                                       synth_B1, synth_B2, z_phot0, z_photErr, N128_SkyPixel_id, sep))
                csvwriter.writerow(g)
    else:
        with open(path_format.format(ps1_strm_dir, "GLADE_matched_with_PS1_Northern_95th.txt"), 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
            next(csvreader)  # skip header

            for g in csvreader:
                gd2_id = int(g[0])
                glade_ra = float(g[1])
                glade_dec = float(g[2])
                z = float(g[3])
                z_dist = float(g[4])
                B = float(g[5])
                ps1_galaxy_id = int(g[6])
                gaia_ra = float(g[7])
                gaia_dec = float(g[8])
                synth_B1 = float(g[9])
                synth_B2 = float(g[10])
                z_phot0 = float(g[11])
                z_photErr = float(g[12])
                N128_SkyPixel_id = int(g[13])
                sep = float(g[14])

                match_galaxies.append((gd2_id, glade_ra, glade_dec, z, z_dist, B, ps1_galaxy_id, gaia_ra, gaia_dec,
                                       synth_B1, synth_B2, z_phot0, z_photErr, N128_SkyPixel_id, sep))

    print("\nMatched sample contains %s galaxies..." % len(match_galaxies))

    # Model the distribution for the redshifts in the galaxy sample.
    # Sample this distribution for the "true z" of this simulated event
    # z_values = [g[11] for g in match_galaxies] # g[11] = PS1 phot z
    z_values = [g[3] for g in match_galaxies]  # g[3] = GLADE phot z
    # Gaussian Kernel Density Estimateion
    z_kde = gaussian_kde(z_values)
    model_z = np.linspace(0, 0.13, 101)
    model_z_pdf = z_kde.evaluate(model_z)

    true_z = z_kde.resample(1)[0][0]
    true_D = c * true_z / H0  # convert to True distance via H0...
    print("Sampled z: %s" % true_z)
    print("True D given H0=%s and z=%s: %s Mpc" % (H0, true_z, true_D))


    # Create a distance distribution for this true D
    measured_D = true_D + (1 if random.random() < 0.5 else -1) * true_D * random.uniform(0.0, 0.333)  # add some noise
    measured_D_err = measured_D * 0.3
    measured_D_dist = norm(loc=measured_D, scale=measured_D_err)
    print("Measured Distance, Err: %s +/- %s Mpc" % (measured_D, measured_D_err))



    # Get galaxy H0
    glade_z = match_galaxies[0][3]
    spec_z_err = 1.5e-4 # this is a guess for GLADE galaxies
    H0_i_spec = (c * glade_z)/measured_D
    H0_i_err_spec = np.sqrt(spec_z_err**2*(c/measured_D)**2 + measured_D_err**2*(-(c*glade_z)/measured_D**2)**2)
    H_spec = norm(loc=H0_i_spec, scale=H0_i_err_spec)

    print("Spec-z h0: %0.4f" % H0_i_spec)
    print("Spec-z err: %0.4f" % H0_i_err_spec)


    ps1_z = match_galaxies[0][11]
    # phot_z_err = match_galaxies[0][12]
    phot_z_err = 0.03
    H0_i_phot = (c * ps1_z) / measured_D
    H0_i_err_phot = np.sqrt(phot_z_err**2*(c/measured_D)**2 + measured_D_err**2*(-(c*ps1_z)/measured_D**2)**2)
    H_phot = norm(loc=H0_i_phot, scale=H0_i_err_phot)

    print("Phot-z h0: %0.4f" % H0_i_phot)
    print("Phot-z err: %0.4f" % H0_i_err_phot)


    fig = plt.figure(figsize=(18, 18), dpi=800)
    ax1 = fig.add_subplot(221)

    n1, bins1, patches1 = ax1.hist(z_values, histtype='step', bins=np.linspace(0, 0.13, 20), color="red",
                                   density=True, label="GLADE spec z\nNorthern 95th\nH0=70, z=[%0.4f, %0.4f]" %
                                                       (z_min, z_max))
    ax1.axvline(true_z, 0, 1, color='b', linestyle=":", label="true_z/D=%0.4f/%0.0f" % (true_z, true_D))
    ax1.plot(model_z, model_z_pdf, 'k--', label="KDE")
    ax1.set_xlabel("Photo z")
    ax1.set_ylabel("PDF")
    ax1.legend(loc="upper right", labelspacing=1.5)


    ax2 = fig.add_subplot(222)
    D0_dist_input = np.linspace(D0 - 3*D0_err, D0 + 3*D0_err, 100)
    D_dist_input = np.linspace(measured_D - 3 * measured_D_err, measured_D + 3 * measured_D_err, 100)
    ax2.plot(D0_dist_input, D0_dist.pdf(D0_dist_input), color='r', label="0814\nD=%0.0f+/-%0.0f" % (D0, D0_err))

    ax2.axvline(measured_D, 0, 1, color='b', linestyle=":", label="0814 sampled D=%0.0f Mpc" % measured_D)
    ax2.plot(D_dist_input, measured_D_dist.pdf(D_dist_input), color='k', linestyle='--',
             label="Resampled\nD=%0.0f+/-%0.0f" % (measured_D, measured_D_err))

    ax2.set_xlabel(r"$\mathrm{D_L}$ [Mpc]")
    ax2.set_ylabel("PDF")
    ax2.set_ylim(ymin=0)
    ax2.legend(loc="upper right", labelspacing=1.5)


    ax3 = fig.add_subplot(223)
    H0_input = np.linspace(H0_i_spec - 5*H0_i_err_phot, H0_i_spec + 5*H0_i_err_phot, 100)
    ax3.plot(H0_input, H_spec.pdf(H0_input), color='r', label="Spec H0\nGal spec z=%0.4f+/-%0.4f" %
                                                              (glade_z, spec_z_err))
    fractional_spec_err = (H0_i_err_spec/H0_i_spec)*100
    fractional_phot_err = (H0_i_err_phot/H0_i_phot)*100

    ax3.axvline(H0_i_spec, 0, 1, color='r', linestyle=":", label="H0_spec=%0.2f +/- %0.0f%%" %
                                                                 (H0_i_spec, fractional_spec_err))
    ax3.plot(H0_input, H_phot.pdf(H0_input), color='k', linestyle='--', label="Phot H0\nGal phot z=%0.4f+/-%0.4f" %
                                                                              (ps1_z, phot_z_err))
    ax3.axvline(H0_i_phot, 0, 1, color='k', linestyle=":", label="H0_phot=%0.2f +/- %0.0f%%" %
                                                                 (H0_i_phot, fractional_phot_err))
    ax3.set_xlabel(r"$\mathrm{H_0}$ [km s$^-1$ Mpc$^-1$]")
    ax3.set_ylabel("PDF")
    ax3.set_ylim(ymin=0)
    ax3.set_xlim([20, 140])
    ax3.legend(loc="upper right", labelspacing=1.5)


    fig.savefig("H0_single_test.png", bbox_inches='tight')
    plt.close('all')

    end = time.time()
    duration = (end - start)
    print("\n********* start DEBUG ***********")
    print("Execution time: %s" % duration)
    print("********* end DEBUG ***********\n")

if h0_calc_test_sample:

    start = time.time()

    # Global variables
    H0 = 70  # km/s/Mpc
    c = 2.998e+5  # km/s
    D0 = 267.
    D0_err = 52.
    # D0_frac_err = D0_err/D0
    D0_dist = norm(loc=D0, scale=D0_err)
    print("Original Distance, Err: %s +/- %s Mpc" % (D0, D0_err))

    # z bounds given D0 and H0
    z_min = (H0 * (D0 - D0_err)) / c
    z_max = (H0 * (D0 + D0_err)) / c
    print("\nz bounds [%s, %s]\n" % (z_min, z_max))


    # Get localization pixels
    northern_95th_pixel_ids = []
    northern_95th_pixel_ids_file = path_format.format(ps1_strm_dir, "northern_95th_pixel_ids.txt")
    with open(northern_95th_pixel_ids_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        next(csvreader)  # skip header
        for row in csvreader:
            id = row[0]
            northern_95th_pixel_ids.append(id)


    # Get galaxies in localization
    get_from_db = False
    match_galaxies = []
    if get_from_db:
        id_str = ",".join([str(i) for i in northern_95th_pixel_ids])
    #     select_galaxies_in_volume = '''
    #     SELECT
    #         g.id as gd2_id,
    #         g.RA,
    #         g._Dec,
    #         g.z,
    #         g.z_dist,
    #         g.B,
    #         p.ps1_galaxy_id,
    #         p.gaia_ra,
    #         p.gaia_dec,
    #         p.synth_B1,
    #         p.synth_B2,
    #         p.z_phot0,
    #         p.z_photErr,
    #         p.N128_SkyPixel_id as ps1_N128_SkyPixel_id,
    #         AngSep(g.RA, g._Dec, p.gaia_ra, p.gaia_dec) AS sep
    #     FROM
    #         (SELECT
    #             gd2.id,
    #             gd2.RA,
    #             gd2._Dec,
    #             gd2.z,
    #             gd2.z_dist,
    #             gd2.B
    #         FROM
    #             GalaxyDistance2 gd2
    #         JOIN
    #             HealpixPixel_GalaxyDistance2 hp_gd2 on hp_gd2.GalaxyDistance2_id = gd2.id
    #         JOIN
    #             HealpixPixel hp on hp.id = hp_gd2.HealpixPixel_id
    #         WHERE
    #             hp.HealpixMap_id = 2 AND
    #             gd2.flag2 IN (1,3) AND
    #             hp.id IN (%s)) AS g
    #     JOIN
    #         (SELECT
    #             ps1.id as ps1_galaxy_id,
    #             ps1.gaia_ra,
    #             ps1.gaia_dec,
    #             ps1.synth_B1,
    #             ps1.synth_B2,
    #             ps.z_phot0,
    #             ps.z_photErr,
    #             ps1.N128_SkyPixel_id
    #         FROM
    #             PS1_Galaxy ps1
    #         JOIN
    #             PS1_STRM ps on ps.uniquePspsOBid = ps1.uniquePspsOBid
    #         JOIN
    #             HealpixPixel_PS1_Galaxy hp_ps1 on hp_ps1.PS1_Galaxy_id = ps1.id
    #         JOIN
    #             HealpixPixel hp on hp.id = hp_ps1.HealpixPixel_id
    #         WHERE
    #             hp.HealpixMap_id = 2 AND
    #             ps.z_phot0 <> -999 AND
    #             hp.id IN (%s)) AS p
    #     ON AngSep(g.RA, g._Dec, p.gaia_ra, p.gaia_dec)*3600 <= 1.0
    #     WHERE g.z BETWEEN %s AND %s
    # '''

        select_galaxies_in_volume = '''
            SELECT 
                gd2.id, 
                gd2.RA, 
                gd2._Dec, 
                gd2.z, 
                gd2.z_dist, 
                gd2.B 
            FROM 
                GalaxyDistance2 gd2 
            JOIN 
                HealpixPixel_GalaxyDistance2 hp_gd2 on hp_gd2.GalaxyDistance2_id = gd2.id 
            JOIN 
                HealpixPixel hp on hp.id = hp_gd2.HealpixPixel_id 
            WHERE 
                hp.HealpixMap_id = 2 AND 
                hp.id IN (%s) AND 
                gd2.z BETWEEN %s AND %s  
            '''

        # galaxy_result = query_db([select_galaxies_in_volume % (id_str, id_str, z_min, z_max)])[0]
        galaxy_result = query_db([select_galaxies_in_volume % (id_str, z_min, z_max)])[0]

        # with open(path_format.format(ps1_strm_dir, "GLADE_matched_with_PS1_Northern_95th.txt"), 'w') as csvfile:
        #     csvwriter = csv.writer(csvfile, delimiter=',')
        #     csvwriter.writerow(("gd2_id", "glade_ra", "glade_dec", "z", "z_dist", "B", "ps1_galaxy_id", "gaia_ra",
        #                         "gaia_dec", "synth_B1", "synth_B2", "z_phot0", "z_photErr", "N128_SkyPixel_id", "sep"))
        #     for g in galaxy_result:
        #         gd2_id = int(g[0])
        #         glade_ra = float(g[1])
        #         glade_dec = float(g[2])
        #         z = float(g[3])
        #         z_dist = float(g[4])
        #         B = float(g[5])
        #         ps1_galaxy_id = int(g[6])
        #         gaia_ra = float(g[7])
        #         gaia_dec = float(g[8])
        #         synth_B1 = float(g[9])
        #         synth_B2 = float(g[10])
        #         z_phot0 = float(g[11])
        #         z_photErr = float(g[12])
        #         N128_SkyPixel_id = int(g[13])
        #         sep = float(g[14])
        #
        #         match_galaxies.append((gd2_id, glade_ra, glade_dec, z, z_dist, B, ps1_galaxy_id, gaia_ra, gaia_dec,
        #                                synth_B1, synth_B2, z_phot0, z_photErr, N128_SkyPixel_id, sep))
        #         csvwriter.writerow(g)

        with open(path_format.format(ps1_strm_dir, "GLADE_PS1_Northern_95th.txt"), 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(("gd2_id", "glade_ra", "glade_dec", "z", "z_dist", "B"))
            for g in galaxy_result:
                gd2_id = int(g[0])
                glade_ra = float(g[1])
                glade_dec = float(g[2])
                z = float(g[3])
                z_dist = float(g[4])
                B = float(g[5])

                match_galaxies.append((gd2_id, glade_ra, glade_dec, z, z_dist, B))
                csvwriter.writerow(g)

    else:
        # with open(path_format.format(ps1_strm_dir, "GLADE_matched_with_PS1_Northern_95th.txt"), 'r') as csvfile:
        #     csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        #     next(csvreader)  # skip header
        #
        #     for g in csvreader:
        #         gd2_id = int(g[0])
        #         glade_ra = float(g[1])
        #         glade_dec = float(g[2])
        #         z = float(g[3])
        #         z_dist = float(g[4])
        #         B = float(g[5])
        #         ps1_galaxy_id = int(g[6])
        #         gaia_ra = float(g[7])
        #         gaia_dec = float(g[8])
        #         synth_B1 = float(g[9])
        #         synth_B2 = float(g[10])
        #         z_phot0 = float(g[11])
        #         z_photErr = float(g[12])
        #         N128_SkyPixel_id = int(g[13])
        #         sep = float(g[14])
        #
        #         match_galaxies.append((gd2_id, glade_ra, glade_dec, z, z_dist, B, ps1_galaxy_id, gaia_ra, gaia_dec,
        #                                synth_B1, synth_B2, z_phot0, z_photErr, N128_SkyPixel_id, sep))
        with open(path_format.format(ps1_strm_dir, "GLADE_PS1_Northern_95th.txt"), 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
            next(csvreader)  # skip header

            for g in csvreader:
                gd2_id = int(g[0])
                glade_ra = float(g[1])
                glade_dec = float(g[2])
                z = float(g[3])
                z_dist = float(g[4])
                B = float(g[5])

                match_galaxies.append((gd2_id, glade_ra, glade_dec, z, z_dist, B))
    print("\nMatched sample contains %s galaxies..." % len(match_galaxies))

    # Model the distribution for the redshifts in the galaxy sample.
    # Sample this distribution for the "true z" of this simulated event
    # z_values = [g[11] for g in match_galaxies] # g[11] = PS1 phot z
    z_values = [g[3] for g in match_galaxies]  # g[3] = GLADE phot z
    z_values_phot = []


    # spec_z_err = 1.5e-4  # this is a guess for GLADE galaxies
    spec_z_err = 4.2e-4  # OzDES error on Q=3 redshifts
    avg_ps1_fake_z_err = 0.0322 # avg from PS1 STRM
    for g in match_galaxies:
        glade_z = g[3]
        spec_dist = norm(loc=glade_z, scale=avg_ps1_fake_z_err) # use spec mean, but the avg phot err to sample

        fake_z = spec_dist.ppf(random.uniform(0, 1))  # Sampled Distance
        z_values_phot.append(fake_z)

    z_kde = gaussian_kde(z_values)
    model_z = np.linspace(0, 0.15, 151)
    model_z_pdf = z_kde.evaluate(model_z)
    true_z = z_kde.resample(1)[0][0]
    # true_zs = z_kde.resample(1000)[0]
    # true_z = true_zs[0]
    true_D = c * true_z/H0 # convert to True distance via H0...
    print("Sampled z: %s" % true_z)
    print("True D given H0=%s and z=%s: %s Mpc" % (H0, true_z, true_D))

    # Create a distance distribution for this true D
    measured_D = true_D + (1 if random.random() < 0.5 else -1) * true_D * random.uniform(0.0, 0.25)  # add some noise
    measured_D_err = measured_D * 0.25
    measured_D_dist = norm(loc=measured_D, scale=measured_D_err)
    print("Measured Distance, Err: %s +/- %s Mpc" % (measured_D, measured_D_err))



    # Process ensemble
    H0_spec = []
    H0_spec_err = []
    H0_spec_dist = []

    H0_phot = []
    H0_phot_err = []
    H0_phot_dist = []
    for i, g in enumerate(match_galaxies):
        glade_z = g[3]
        H0_i_spec = (c * glade_z)/measured_D
        H0_i_err_spec = np.sqrt(spec_z_err**2*(c/measured_D)**2 + measured_D_err**2*(-(c*glade_z)/measured_D**2)**2)
        H_spec = norm(loc=H0_i_spec, scale=H0_i_err_spec)
        H0_spec.append(H0_i_spec)
        H0_spec_err.append(H0_i_err_spec)
        H0_spec_dist.append(H_spec)

        ps1_z = z_values_phot[i]
        phot_z_err = avg_ps1_fake_z_err
        H0_i_phot = (c * ps1_z) / measured_D
        H0_i_err_phot = np.sqrt(phot_z_err**2*(c/measured_D)**2 + measured_D_err**2*(-(c*ps1_z)/measured_D**2)**2)
        H_phot = norm(loc=H0_i_phot, scale=H0_i_err_phot)

        H0_phot.append(H0_i_phot)
        H0_phot_err.append(H0_i_err_phot)
        H0_phot_dist.append(H_phot)


    fig = plt.figure(figsize=(18, 18), dpi=800)
    ax1 = fig.add_subplot(221)

    n1, bins1, patches1 = ax1.hist(z_values, histtype='step', bins=np.linspace(0, 0.13, 20), color="red",
                                   density=True, label="GLADE all z\nNorthern 95th\nH0=70\nz=[%0.4f, %0.4f]" %
                                                       (z_min, z_max))


    ax1.plot(model_z, model_z_pdf, 'k--', label="KDE")

    ax1.vlines(true_z, 0, z_kde.pdf(true_z), color='b', linestyle=":", label="'true' z=%0.4f" % (true_z))

    ax1.set_xlabel("GLADE spec z")
    ax1.set_ylabel("PDF")
    ax1.set_xlim([0.04, 0.08])
    ax1.legend(loc="upper left", labelspacing=1.5)


    ax2 = fig.add_subplot(222)
    D0_dist_input = np.linspace(D0 - 3*D0_err, D0 + 3*D0_err, 100)
    D_dist_input = np.linspace(measured_D - 3 *measured_D_err, measured_D + 3 * measured_D_err, 100)
    ax2.plot(D0_dist_input, D0_dist.pdf(D0_dist_input), color='r', label="0814\nD=%0.0f+/-%0.0f" % (D0, D0_err))
    ax2.vlines(true_D, 0, D0_dist.pdf(true_D), colors='b', linestyles=":", label="'true' D=%0.0f Mpc" % true_D)
    ax2.plot(D_dist_input, measured_D_dist.pdf(D_dist_input), color='k', linestyle='--', label="Measured\nD=%0.0f+/-%0.0f" %
                                                                                      (measured_D, measured_D_err))
    ax2.set_xlabel(r"$\mathrm{D_L}$ [Mpc]")
    ax2.set_ylabel("PDF")
    ax2.set_ylim(ymin=0)
    ax2.legend(loc="upper right", labelspacing=1.5)


    ax3 = fig.add_subplot(223)
    H0_input = np.linspace(0.0, 150.0, 300)
    delta_input = np.abs(H0_input[1] - H0_input[0])

    # running_output_spec = np.zeros(len(H0_input))
    # for sd in H0_spec_dist:
    #     running_output_spec = np.add(running_output_spec, sd.pdf(H0_input))
    # spec_pdf = running_output_spec / trapz(running_output_spec, H0_input)
    spec_pdf = reduce((lambda x, y: np.add(x, y)), map(lambda x: x.pdf(H0_input), H0_spec_dist))
    spec_pdf = spec_pdf/trapz(spec_pdf, H0_input) # normalize
    ax3.plot(H0_input, spec_pdf, color='r')





    # def find_nearest_index(array, value):
    #     array = np.asarray(array)
    #     idx = (np.abs(array - value)).argmin()
    #     return idx

    threshold_16 = 0.16
    threshold_50 = 0.50
    threshold_84 = 0.84

    spec_running_prob = 0.0
    spec_index_of_16 = -1
    spec_index_of_50 = -1
    spec_index_of_84 = -1
    spec_found_16 = False
    spec_found_50 = False

    for i, p in enumerate(spec_pdf):

        spec_running_prob += p * delta_input
        if spec_running_prob >= threshold_16 and not spec_found_16:
            spec_found_16 = True
            spec_index_of_16 = i

        if spec_running_prob >= threshold_50 and not spec_found_50:
            spec_found_50 = True
            spec_index_of_50 = i

        if spec_running_prob >= threshold_84:
            spec_index_of_84 = i
            break

    spec_median = H0_input[spec_index_of_50]
    spec_lower_bound = H0_input[spec_index_of_16]
    spec_upper_bound = H0_input[spec_index_of_84]
    print(spec_lower_bound, spec_median, spec_upper_bound)
    spec_frac_err = 100*(spec_upper_bound - spec_lower_bound)/(2 * spec_median)

    ax3.vlines(spec_lower_bound, 0.0, spec_pdf[spec_index_of_16], colors='r', linestyles=':')
    ax3.vlines(spec_median, 0.0, spec_pdf[spec_index_of_50], colors='r', linestyles='-',
               label=r"Ensemble H0$\mathrm{_{spec}}$=%0.2f$^{+%0.0f}_{-%0.0f}$ (%0.0f)%%" %
                     (spec_median, (spec_median - spec_lower_bound), (spec_upper_bound - spec_median), spec_frac_err))
    ax3.vlines(spec_upper_bound, 0.0, spec_pdf[spec_index_of_84], colors='r', linestyles=':')



    # running_output_phot = np.zeros(len(H0_input))
    # for sd in H0_phot_dist:
    #     running_output_phot = np.add(running_output_phot, sd.pdf(H0_input))
    # phot_pdf = running_output_phot / trapz(running_output_phot, H0_input)
    phot_pdf = reduce((lambda x, y: np.add(x, y)), map(lambda x: x.pdf(H0_input), H0_phot_dist))
    phot_pdf = phot_pdf/trapz(phot_pdf, H0_input)

    ax3.plot(H0_input, phot_pdf, color='k', linestyle='--')

    # get the max value of the dist
    phot_running_prob = 0.0

    phot_index_of_16 = -1
    phot_index_of_50 = -1
    phot_index_of_84 = -1
    phot_found_16 = False
    phot_found_50 = False

    for i, p in enumerate(phot_pdf):

        phot_running_prob += p * delta_input

        if phot_running_prob >= threshold_16 and not phot_found_16:
            phot_found_16 = True
            phot_index_of_16 = i

        if phot_running_prob >= threshold_50 and not phot_found_50:
            phot_found_50 = True
            phot_index_of_50 = i

        if phot_running_prob >= threshold_84:
            phot_index_of_84 = i
            break

    phot_median = H0_input[phot_index_of_50]
    phot_lower_bound = H0_input[phot_index_of_16]
    phot_upper_bound = H0_input[phot_index_of_84]
    print(phot_lower_bound, phot_median, phot_upper_bound)
    phot_frac_err = 100 * (phot_upper_bound - phot_lower_bound) / (2 * phot_median)

    ax3.vlines(phot_lower_bound, 0.0, phot_pdf[phot_index_of_16], colors='k', linestyles=':')
    ax3.vlines(phot_median, 0.0, phot_pdf[phot_index_of_50], colors='k', linestyles='--',
               label=r"Ensemble H0$\mathrm{_{phot}}$=%0.2f$^{+%0.0f}_{-%0.0f}$ (%0.0f)%%" %
                     (phot_median, (phot_median - phot_lower_bound), (phot_upper_bound - phot_median), phot_frac_err))
    ax3.vlines(phot_upper_bound, 0.0, phot_pdf[phot_index_of_84], colors='k', linestyles=':')


    ax3.set_xlabel(r"$\mathrm{H_0}$ [km s$^-1$ Mpc$^-1$]")
    ax3.set_ylabel("PDF")
    ax3.set_ylim(ymin=0)

    ax3.set_xlim([10, 150])

    ax3.legend(loc="upper right", labelspacing=1.5)

    # ax4 = fig.add_subplot(224)
    # test_sum = reduce((lambda x, y: np.add(x, y)), map(lambda x: x.pdf(H0_input), H0_spec_dist))
    # ax4.plot(H0_input, test_sum)
    # ax4.set_ylim(ymin=0)
    # ax4.set_xlim([10, 150])


    fig.savefig("H0_sample_test.png", bbox_inches='tight')
    plt.close('all')

    end = time.time()
    duration = (end - start)
    print("\n********* start DEBUG ***********")
    print("Execution time: %s" % duration)
    print("********* end DEBUG ***********\n")

if h0_calc_test_multiple_sample:

    start = time.time()

    # Global variables
    H0 = 70  # km/s/Mpc
    c = 2.998e+5  # km/s
    D0 = 267.
    D0_err = 52.
    D0_dist = norm(loc=D0, scale=D0_err)
    print("Original Distance, Err: %s +/- %s Mpc" % (D0, D0_err))

    # z bounds given D0 and H0
    z_min = (H0 * (D0 - D0_err)) / c
    z_max = (H0 * (D0 + D0_err)) / c
    print("\nz bounds [%s, %s]\n" % (z_min, z_max))

    def get_confidence_intervals(x_arr, y_arr):
        delta_input = x_arr[1] - x_arr[0]
        threshold_16 = 0.16
        threshold_50 = 0.50
        threshold_84 = 0.84
        running_prob = 0.0
        index_of_16 = -1
        index_of_50 = -1
        index_of_84 = -1
        found_16 = False
        found_50 = False

        for i, p in enumerate(y_arr):

            running_prob += p * delta_input
            if running_prob >= threshold_16 and not found_16:
                found_16 = True
                index_of_16 = i

            if running_prob >= threshold_50 and not found_50:
                found_50 = True
                index_of_50 = i

            if running_prob >= threshold_84:
                index_of_84 = i
                break

        if index_of_16 == -1 or index_of_50 == -1 or index_of_84 == -1:
            print("\n\n************************************")
            print(x_arr)
            print(y_arr)
            print(i)
            print(index_of_16)
            print(index_of_50)
            print(index_of_84)
            raise Exception("Could not find indices!")

        median_xval = x_arr[index_of_50]
        xval_lower_bound = x_arr[index_of_16]
        xval_upper_bound = x_arr[index_of_84]
        frac_measurement = 100 * (xval_upper_bound - xval_lower_bound) / (2 * median_xval)

        return xval_lower_bound, index_of_16, median_xval, index_of_50, xval_upper_bound, index_of_84, frac_measurement

    # Start iteration here
    percent_spec_measurement = 999
    percent_phot_measurement = 999
    threshold_measurement = 3

    distance_dists = []
    spec_dists = []
    phot_dists = []
    joint_spec_min_H0 = joint_spec_median_H0 = joint_spec_max_H0 = 999
    index_spec_min_H0 = index_spec_median_H0 = index_spec_max_H0 = 999
    joint_spec_pdf = None
    joint_phot_pdf = None
    H0_input = np.linspace(0.0, 150.0, 300)
    fig = plt.figure(figsize=(18, 18), dpi=800)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)

    select_complete_pixels = '''
        SELECT sp.id as SkyPixel_id, sp.Pixel_Index, sp.Pixel_Index, sp.RA, sp._Dec, sc.SmoothedCompleteness 
        FROM SkyCompleteness sc  
        JOIN SkyDistance sd on sd.id = sc.SkyDistance_id 
        JOIN SkyPixel sp on sp.id = sc.SkyPixel_id 
        WHERE sd.id BETWEEN 47 AND 64 AND 
            sc.SmoothedCompleteness >= 0.9 
    '''

    skypixel_result = query_db([select_complete_pixels])[0]
    skypixel_ids = []
    for sp in skypixel_result:
        if sp[0] not in skypixel_ids:
            skypixel_ids.append(sp[0])

    print("Number of skypixels: %s" % len(skypixel_ids))
    print("Starting iterations...")
    while percent_spec_measurement > threshold_measurement and len(spec_dists) < len(skypixel_ids):

        iteration_index = len(spec_dists)
        iter_str = str(iteration_index + 1)

        # Get galaxies in highly complete pixels
        print("\nQuery pixel id %s" % skypixel_ids[iteration_index])
        select_galaxies = '''
            SELECT 
                gd2.z
            FROM GalaxyDistance2 gd2
            JOIN SkyPixel_GalaxyDistance2 sp_gd2 ON sp_gd2.GalaxyDistance2_id = gd2.id
            WHERE sp_gd2.SkyPixel_id = %s AND gd2.z BETWEEN %s AND %s  
        '''
        galaxy_result = query_db([select_galaxies % (skypixel_ids[iteration_index], z_min, z_max)])[0]
        z_values = [float(g[0]) for g in galaxy_result]
        print("Iteration %s sample contains %s galaxies..." % (iter_str, len(z_values)))

        # spec_z_err = 1.5e-4  # this is a guess for GLADE galaxies
        spec_z_err = 4.2e-4  # OzDES error on Q=3 redshifts

        z_kde = gaussian_kde(z_values)
        model_z = np.linspace(0, 0.15, 151)
        model_z_pdf = z_kde.evaluate(model_z)
        true_z = z_kde.resample(1)[0][0]
        true_D = c * true_z / H0  # convert to True distance via H0...
        print("Sampled z: %s" % true_z)
        print("True D given H0=%s and z=%s: %s Mpc" % (H0, true_z, true_D))

        print("Iter %s: percent measurement: %0.0f" % (iter_str, percent_spec_measurement))

        # Create a distance distribution for this true D
        measured_D = true_D + (1 if random.random() < 0.5 else -1) * true_D * random.uniform(0.0, 0.25)  # add some noise
        measured_D_err = measured_D * 0.25
        measured_D_dist = norm(loc=measured_D, scale=measured_D_err)
        print("Measured Distance, Err: %s +/- %s Mpc" % (measured_D, measured_D_err))

        # Process ensemble
        H0_spec = []
        H0_spec_err = []
        H0_spec_dist = []

        for i, glade_z in enumerate(z_values):
            H0_i_spec = (c * glade_z)/measured_D
            H0_i_err_spec = np.sqrt(spec_z_err**2*(c/measured_D)**2 + measured_D_err**2*(-(c*glade_z)/measured_D**2)**2)
            H_spec = norm(loc=H0_i_spec, scale=H0_i_err_spec)
            H0_spec.append(H0_i_spec)
            H0_spec_err.append(H0_i_err_spec)
            H0_spec_dist.append(H_spec)

        spec_pdf = reduce((lambda x, y: np.add(x, y)), map(lambda z: z.pdf(H0_input), H0_spec_dist))
        # spec_pdf = spec_pdf / trapz(spec_pdf, H0_input)  # normalize
        spec_pdf_norm = spec_pdf / trapz(spec_pdf, H0_input)  # normalize

        joint_spec_min_H0, index_spec_min_H0, joint_spec_median_H0, index_spec_median_H0, joint_spec_max_H0, \
            index_spec_max_H0, percent_spec_measurement = get_confidence_intervals(H0_input, spec_pdf)

        spec_dists.append(spec_pdf)

        joint_spec_pdf = reduce((lambda x, y: np.multiply(x, y)), spec_dists)
        joint_spec_pdf = joint_spec_pdf / trapz(joint_spec_pdf, H0_input)  # normalize

        joint_spec_min_H0, index_spec_min_H0, joint_spec_median_H0, index_spec_median_H0, joint_spec_max_H0, \
        index_spec_max_H0, percent_spec_measurement = get_confidence_intervals(H0_input, joint_spec_pdf)

        # if len(spec_dists) > 1:
        #     joint_spec_pdf = reduce((lambda x, y: np.multiply(x, y)), spec_dists)
        #     joint_spec_pdf = joint_spec_pdf / trapz(joint_spec_pdf, H0_input)  # normalize
        #
        #     joint_spec_min_H0, index_spec_min_H0, joint_spec_median_H0, index_spec_median_H0, joint_spec_max_H0, \
        #     index_spec_max_H0, percent_spec_measurement = get_confidence_intervals(H0_input, joint_spec_pdf)
        # else:
        #     joint_spec_pdf = spec_pdf

        # , label=r"iter=%s; dist=%0.0f$\pm$%0.0f Mpc" % (str(len(spec_dists)+1), measured_D, measured_D_err)
        # ax1.plot(H0_input, spec_pdf, color="gray", alpha=0.2)
        ax1.plot(H0_input, spec_pdf_norm, color="gray", alpha=0.2)

        # distance_dists.append(measured_D_dist)
        # spec_dists.append(spec_pdf)

    while percent_phot_measurement > threshold_measurement and len(phot_dists) < len(skypixel_ids):

        iteration_index = len(phot_dists)
        iter_str = str(iteration_index + 1)

        # Get galaxies in highly complete pixels
        print("\nQuery pixel id %s" % skypixel_ids[iteration_index])
        select_galaxies = '''
            SELECT 
                gd2.z
            FROM GalaxyDistance2 gd2
            JOIN SkyPixel_GalaxyDistance2 sp_gd2 ON sp_gd2.GalaxyDistance2_id = gd2.id
            WHERE sp_gd2.SkyPixel_id = %s AND gd2.z BETWEEN %s AND %s  
        '''
        galaxy_result = query_db([select_galaxies % (skypixel_ids[iteration_index], z_min, z_max)])[0]
        z_values = [float(g[0]) for g in galaxy_result]
        print("Iteration %s sample contains %s galaxies..." % (iter_str, len(z_values)))

        # spec_z_err = 1.5e-4  # this is a guess for GLADE galaxies
        spec_z_err = 4.2e-4  # OzDES error on Q=3 redshifts
        avg_ps1_fake_z_err = 0.0322  # avg from PS1 STRM
        z_values_phot = []
        for glade_z in z_values:
            fake_z_dist = norm(loc=glade_z, scale=avg_ps1_fake_z_err)  # use spec mean, but the avg phot err to sample
            fake_z = fake_z_dist.ppf(random.uniform(0, 1))  # Sampled Distance
            z_values_phot.append(fake_z)

        z_kde = gaussian_kde(z_values)
        model_z = np.linspace(0, 0.15, 151)
        model_z_pdf = z_kde.evaluate(model_z)
        true_z = z_kde.resample(1)[0][0]
        true_D = c * true_z / H0  # convert to True distance via H0...
        print("Sampled z: %s" % true_z)
        print("True D given H0=%s and z=%s: %s Mpc" % (H0, true_z, true_D))
        print("Iter %s: percent measurement: %0.0f" % (iter_str, percent_spec_measurement))

        # Create a distance distribution for this true D
        measured_D = true_D + (1 if random.random() < 0.5 else -1) * true_D * random.uniform(0.0, 0.25)  # add some noise
        measured_D_err = measured_D * 0.25
        measured_D_dist = norm(loc=measured_D, scale=measured_D_err)
        print("Measured Distance, Err: %s +/- %s Mpc" % (measured_D, measured_D_err))

        # Process ensemble
        H0_phot = []
        H0_phot_err = []
        H0_phot_dist = []
        for i, glade_z in enumerate(z_values):
            ps1_z = z_values_phot[i]
            phot_z_err = avg_ps1_fake_z_err
            H0_i_phot = (c * ps1_z) / measured_D
            H0_i_err_phot = np.sqrt(phot_z_err**2*(c/measured_D)**2 + measured_D_err**2*(-(c*ps1_z)/measured_D**2)**2)
            H_phot = norm(loc=H0_i_phot, scale=H0_i_err_phot)

            H0_phot.append(H0_i_phot)
            H0_phot_err.append(H0_i_err_phot)
            H0_phot_dist.append(H_phot)

        phot_pdf = reduce((lambda x, y: np.add(x, y)), map(lambda z: z.pdf(H0_input), H0_phot_dist))
        # phot_pdf = phot_pdf / trapz(phot_pdf, H0_input)  # normalize
        phot_pdf_norm = phot_pdf / trapz(phot_pdf, H0_input)  # normalize

        joint_phot_min_H0, index_phot_min_H0, joint_phot_median_H0, index_phot_median_H0, joint_phot_max_H0, \
            index_phot_max_H0, percent_phot_measurement = get_confidence_intervals(H0_input, phot_pdf)
        phot_dists.append(phot_pdf)

        joint_phot_pdf = reduce((lambda x, y: np.multiply(x, y)), phot_dists)
        joint_phot_pdf = joint_phot_pdf / trapz(joint_phot_pdf, H0_input)  # normalize
        # joint_phot_pdf = joint_phot_pdf / trapz(joint_phot_pdf, H0_input)  # normalize

        joint_phot_min_H0, index_phot_min_H0, joint_phot_median_H0, index_phot_median_H0, joint_phot_max_H0, \
        index_phot_max_H0, percent_phot_measurement = get_confidence_intervals(H0_input, joint_phot_pdf)

        # if len(phot_dists) > 1:
            # joint_phot_pdf = reduce((lambda x, y: np.multiply(x, y)), phot_dists)
            # joint_phot_pdf = joint_phot_pdf / trapz(joint_phot_pdf, H0_input)  # normalize
            #
            # joint_phot_min_H0, index_phot_min_H0, joint_phot_median_H0, index_phot_median_H0, joint_phot_max_H0, \
            # index_phot_max_H0, percent_phot_measurement = get_confidence_intervals(H0_input, joint_phot_pdf)
        # else:
        #     joint_phot_pdf = phot_pdf

        # , label=r"iter=%s; dist=%0.0f$\pm$%0.0f Mpc" % (str(len(spec_dists)+1), measured_D, measured_D_err)
        # ax2.plot(H0_input, phot_pdf, color="gray", alpha=0.2)
        ax2.plot(H0_input, phot_pdf_norm, color="gray", alpha=0.2)

        # distance_dists.append(measured_D_dist)
        # phot_dists.append(phot_pdf)

    ax1.plot(H0_input, joint_spec_pdf, color='r', linewidth=2)
    ax1.vlines(joint_spec_min_H0, 0.0, joint_spec_pdf[index_spec_min_H0], colors='r', linestyles=':')
    ax1.vlines(joint_spec_median_H0, 0.0, joint_spec_pdf[index_spec_median_H0], colors='r', linestyles='-',
               label=r"iter=%s H0$\mathrm{_{spec}}$=%0.2f$^{+%0.2f}_{-%0.2f}$ (%0.2f)%%" %
                     (str(len(spec_dists)), joint_spec_median_H0, (joint_spec_median_H0 - joint_spec_min_H0),
                      (joint_spec_max_H0 - joint_spec_median_H0), percent_spec_measurement))
    ax1.vlines(joint_spec_max_H0, 0.0, joint_spec_pdf[index_spec_max_H0], colors='r', linestyles=':')

    ax1.set_xlabel(r"$\mathrm{H_0}$ [km s$^-1$ Mpc$^-1$]")
    ax1.set_ylabel("PDF")
    ax1.set_ylim(ymin=0)
    ax1.set_xlim([10, 150])
    ax1.legend(loc="upper right", labelspacing=1.5)


    ax2.plot(H0_input, joint_phot_pdf, color='k', linewidth=2)
    ax2.vlines(joint_phot_min_H0, 0.0, joint_phot_pdf[index_phot_min_H0], colors='k', linestyles=':')
    ax2.vlines(joint_phot_median_H0, 0.0, joint_phot_pdf[index_phot_median_H0], colors='k', linestyles='-',
               label=r"iter=%s H0$\mathrm{_{phot}}$=%0.2f$^{+%0.2f}_{-%0.2f}$ (%0.2f)%%" %
                     (str(len(phot_dists)), joint_phot_median_H0, (joint_phot_median_H0 - joint_phot_min_H0),
                      (joint_phot_max_H0 - joint_phot_median_H0), percent_phot_measurement))
    ax2.vlines(joint_phot_max_H0, 0.0, joint_phot_pdf[index_phot_max_H0], colors='k', linestyles=':')

    ax2.set_xlabel(r"$\mathrm{H_0}$ [km s$^-1$ Mpc$^-1$]")
    ax2.set_ylabel("PDF")
    ax2.set_ylim(ymin=0)
    ax2.set_xlim([10, 150])
    ax2.legend(loc="upper right", labelspacing=1.5)

    fig.savefig("h0_calc_test_multiple_sample.png", bbox_inches='tight')
    plt.close('all')

    # fig = plt.figure(figsize=(18, 18), dpi=800)
    # ax1 = fig.add_subplot(221)
    #
    # n1, bins1, patches1 = ax1.hist(z_values, histtype='step', bins=np.linspace(0, 0.13, 20), color="red",
    #                                density=True, label="GLADE all z\nNorthern 95th\nH0=70\nz=[%0.4f, %0.4f]" %
    #                                                    (z_min, z_max))
    #
    # ax1.plot(model_z, model_z_pdf, 'k--', label="KDE")
    # ax1.vlines(true_z, 0, z_kde.pdf(true_z), color='b', linestyle=":", label="'true' z=%0.4f" % (true_z))
    # ax1.set_xlabel("GLADE spec z")
    # ax1.set_ylabel("PDF")
    # ax1.set_xlim([0.04, 0.08])
    # ax1.legend(loc="upper left", labelspacing=1.5)
    #
    #
    # # ax2 = fig.add_subplot(222)
    # # D0_dist_input = np.linspace(D0 - 3*D0_err, D0 + 3*D0_err, 100)
    # # D_dist_input = np.linspace(measured_D - 3 *measured_D_err, measured_D + 3 * measured_D_err, 100)
    # # ax2.plot(D0_dist_input, D0_dist.pdf(D0_dist_input), color='r', label="0814\nD=%0.0f+/-%0.0f" % (D0, D0_err))
    # # ax2.vlines(true_D, 0, D0_dist.pdf(true_D), colors='b', linestyles=":", label="'true' D=%0.0f Mpc" % true_D)
    # # ax2.plot(D_dist_input, measured_D_dist.pdf(D_dist_input), color='k', linestyle='--', label="Measured\nD=%0.0f+/-%0.0f" %
    # #                                                                                   (measured_D, measured_D_err))
    # # ax2.set_xlabel(r"$\mathrm{D_L}$ [Mpc]")
    # # ax2.set_ylabel("PDF")
    # # ax2.set_ylim(ymin=0)
    # # ax2.legend(loc="upper right", labelspacing=1.5)
    #
    #
    # ax3 = fig.add_subplot(223)
    #
    # delta_input = np.abs(H0_input[1] - H0_input[0])
    #
    # running_output_spec = np.zeros(len(H0_input))
    # for sd in H0_spec_dist:
    #     running_output_spec = np.add(running_output_spec, sd.pdf(H0_input))
    # spec_pdf = running_output_spec / trapz(running_output_spec, H0_input)
    # ax3.plot(H0_input, spec_pdf, color='r')
    #
    #
    # # def find_nearest_index(array, value):
    # #     array = np.asarray(array)
    # #     idx = (np.abs(array - value)).argmin()
    # #     return idx
    #
    # threshold_16 = 0.16
    # threshold_50 = 0.50
    # threshold_84 = 0.84
    #
    # spec_running_prob = 0.0
    # spec_index_of_16 = -1
    # spec_index_of_50 = -1
    # spec_index_of_84 = -1
    # spec_found_16 = False
    # spec_found_50 = False
    #
    # for i, p in enumerate(spec_pdf):
    #
    #     spec_running_prob += p * delta_input
    #     if spec_running_prob >= threshold_16 and not spec_found_16:
    #         spec_found_16 = True
    #         spec_index_of_16 = i
    #
    #     if spec_running_prob >= threshold_50 and not spec_found_50:
    #         spec_found_50 = True
    #         spec_index_of_50 = i
    #
    #     if spec_running_prob >= threshold_84:
    #         spec_index_of_84 = i
    #         break
    #
    # spec_median = H0_input[spec_index_of_50]
    # spec_lower_bound = H0_input[spec_index_of_16]
    # spec_upper_bound = H0_input[spec_index_of_84]
    # print(spec_lower_bound, spec_median, spec_upper_bound)
    # spec_frac_err = 100*(spec_upper_bound - spec_lower_bound)/(2 * spec_median)
    #
    # ax3.vlines(spec_lower_bound, 0.0, spec_pdf[spec_index_of_16], colors='r', linestyles=':')
    # ax3.vlines(spec_median, 0.0, spec_pdf[spec_index_of_50], colors='r', linestyles='-',
    #            label=r"Ensemble H0$\mathrm{_{spec}}$=%0.2f$^{+%0.0f}_{-%0.0f}$ (%0.0f)%%" %
    #                  (spec_median, (spec_median - spec_lower_bound), (spec_upper_bound - spec_median), spec_frac_err))
    # ax3.vlines(spec_upper_bound, 0.0, spec_pdf[spec_index_of_84], colors='r', linestyles=':')
    #
    #
    #
    # running_output_phot = np.zeros(len(H0_input))
    # for sd in H0_phot_dist:
    #     running_output_phot = np.add(running_output_phot, sd.pdf(H0_input))
    # phot_pdf = running_output_phot / trapz(running_output_phot, H0_input)
    #
    # ax3.plot(H0_input, phot_pdf, color='k', linestyle='--')
    #
    # # get the max value of the dist
    # phot_running_prob = 0.0
    #
    # phot_index_of_16 = -1
    # phot_index_of_50 = -1
    # phot_index_of_84 = -1
    # phot_found_16 = False
    # phot_found_50 = False
    #
    # for i, p in enumerate(phot_pdf):
    #
    #     phot_running_prob += p * delta_input
    #
    #     if phot_running_prob >= threshold_16 and not phot_found_16:
    #         phot_found_16 = True
    #         phot_index_of_16 = i
    #
    #     if phot_running_prob >= threshold_50 and not phot_found_50:
    #         phot_found_50 = True
    #         phot_index_of_50 = i
    #
    #     if phot_running_prob >= threshold_84:
    #         phot_index_of_84 = i
    #         break
    #
    # phot_median = H0_input[phot_index_of_50]
    # phot_lower_bound = H0_input[phot_index_of_16]
    # phot_upper_bound = H0_input[phot_index_of_84]
    # print(phot_lower_bound, phot_median, phot_upper_bound)
    # phot_frac_err = 100 * (phot_upper_bound - phot_lower_bound) / (2 * phot_median)
    #
    # ax3.vlines(phot_lower_bound, 0.0, phot_pdf[phot_index_of_16], colors='k', linestyles=':')
    # ax3.vlines(phot_median, 0.0, phot_pdf[phot_index_of_50], colors='k', linestyles='--',
    #            label=r"Ensemble H0$\mathrm{_{phot}}$=%0.2f$^{+%0.0f}_{-%0.0f}$ (%0.0f)%%" %
    #                  (phot_median, (phot_median - phot_lower_bound), (phot_upper_bound - phot_median), phot_frac_err))
    # ax3.vlines(phot_upper_bound, 0.0, phot_pdf[phot_index_of_84], colors='k', linestyles=':')
    #
    #
    # ax3.set_xlabel(r"$\mathrm{H_0}$ [km s$^-1$ Mpc$^-1$]")
    # ax3.set_ylabel("PDF")
    # ax3.set_ylim(ymin=0)
    #
    # ax3.set_xlim([10, 150])
    #
    # ax3.legend(loc="upper right", labelspacing=1.5)
    #
    #
    # fig.savefig("H0_sample_test.png", bbox_inches='tight')
    # plt.close('all')

    end = time.time()
    duration = (end - start)
    print("\n********* start DEBUG ***********")
    print("Execution time: %s" % duration)
    print("********* end DEBUG ***********\n")








