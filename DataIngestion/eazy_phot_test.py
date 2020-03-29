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
from scipy.stats import pearsonr
from scipy.interpolate import interp1d, interp2d

import pprint
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


    # endregion

    class AAOmega_Galaxy():
        def __init__(self, galaxy_id, ra, dec, prob_galaxy, z, kron_r, pix_index, prob_fraction = 0.0):
            self.galaxy_id = galaxy_id
            self.ra = ra
            self.dec = dec
            self.prob_fraction = prob_fraction
            self.prob_galaxy = prob_galaxy
            self.z = z
            self.kron_r = kron_r
            self.pix_index = pix_index
            self.efficiency_func = ozDES_models[_20_0_20_5][Q3]
            self.num_exps = 0
            self.available = True
            self.required_exps = 1

            if self.kron_r >= 20.5 and self.kron_r < 21.0:
                self.required_exps = 2
                self.efficiency_func = ozDES_models[_20_5_21_0][Q3]
            elif self.kron_r >= 21.0 and self.kron_r < 21.5:
                self.required_exps = 3
                self.efficiency_func = ozDES_models[_21_0_21_5][Q3]
            elif self.kron_r >= 21.5:
                self.required_exps = 4
                self.efficiency_func = ozDES_models[_21_5_22_0][Q3]

        def compute_weight(self, num_exps):
            MIN_EXP = 40  # minutes
            total_exp_time = num_exps * MIN_EXP
            efficiency = self.efficiency_func(total_exp_time)
            metric = efficiency * self.prob_galaxy * self.prob_fraction
            return metric

        def increment_exps(self, num_exps):
            self.num_exps += num_exps
            self.available = self.num_exps < self.required_exps

    class AAOmega_Pixel(Pixel_Element):
        def __init__(self, index, nside, prob, pixel_id=None, mean_dist=None, stddev_dist=None):
            Pixel_Element.__init__(self, index, nside, prob, pixel_id, mean_dist, stddev_dist)
            self.galaxy_ids = []

        def get_available_galaxies_by_multiplicity(self, N):
            for g_id in self.galaxy_ids:
                gal = galaxy_dict[g_id]
                if gal.available and gal.required_exps == N:
                    yield gal

    map_nside = 1024
    localization_poly = []
    northern_pixels = []

    # Key dictionary off pixel index
    pixel_dict = {}

    # region Load pixel_ids from file...
    pixel_select = '''
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
    northern_pixel_ids = []
    northern_95th_pixel_ids = path_format.format(ps1_strm_dir, "northern_95th_pixel_ids.txt")
    with open(northern_95th_pixel_ids, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        next(csvreader)  # skip header

        for row in csvreader:
            id = row[0]
            northern_pixel_ids.append(id)
    pixel_result_north = query_db([pixel_select % ",".join(northern_pixel_ids)])[0]
    print("Total NSIDE=1024 pixels in Northern 95th: %s" % len(pixel_result_north))

    for m in pixel_result_north:
        pix_id = int(m[0])
        index = int(m[2])
        prob = float(m[3])
        dist = float(m[7])
        stddev = float(m[8])
        p = AAOmega_Pixel(index, map_nside, prob, pixel_id=pix_id, mean_dist=dist, stddev_dist=stddev)
        pixel_dict[index] = p

    # Key dictionary off pixel index
    galaxy_dict = {}
    # Get galaxies in north
    ps1_galaxy_select = '''
        SELECT
            ps1.id as Galaxy_id, 
            ps1.gaia_ra,
            ps1.gaia_dec,
            ps1.rMeanKronMag,
            ps.z_phot0,
            ps.prob_Galaxy,
            hp.id as Pixel_id,
            hp.Pixel_Index
        FROM 
            PS1_Galaxy ps1
        JOIN HealpixPixel_PS1_Galaxy hp_ps1 on hp_ps1.PS1_Galaxy_id = ps1.id
        JOIN PS1_STRM ps on ps.uniquePspsOBid = ps1.uniquePspsOBid
        JOIN HealpixPixel hp on hp.id = hp_ps1.HealpixPixel_id
        WHERE 
            ps1.GoodCandidate = 1 AND
            hp.id IN (%s) 
        '''
    galaxy_result_north = query_db([ps1_galaxy_select % ",".join(northern_pixel_ids)])[0]
    print("Returned %s galaxies" % len(galaxy_result_north))

    for g in galaxy_result_north:
        galaxy_id = int(g[0])
        ra = float(g[1])
        dec = float(g[2])
        r = float(g[3])
        z = float(g[4])
        prob_Galaxy = float(g[5])
        pix_id = int(g[6])
        pix_index = int(g[7])

        # associate this galaxy with the hosting pixel
        if galaxy_id not in pixel_dict[pix_index].galaxy_ids:
            pixel_dict[pix_index].galaxy_ids.append(galaxy_id)

        aaomega_galaxy = AAOmega_Galaxy(galaxy_id, ra, dec, prob_Galaxy, z, r, pix_index)
        if galaxy_id not in galaxy_dict:
            galaxy_dict[galaxy_id] = aaomega_galaxy

        # if pix_index not in galaxy_dict:
        #     galaxy_dict[pix_index] = []
        # galaxy_dict[pix_index].append(aaomega_galaxy)

    for gal_id, galaxy in galaxy_dict.items():
        pixel = pixel_dict[galaxy.pix_index]
        prob_fraction = pixel.prob/len(pixel.galaxy_ids)
        galaxy.prob_fraction = prob_fraction

    class AAOmega_Tile(Tile):
        def __init__(self, central_ra_deg, central_dec_deg, nside, radius, num_exposures, tile_num):
            Tile.__init__(self, central_ra_deg, central_dec_deg, width=None, height=None, nside=nside, radius=radius)
            self.num_exposures = num_exposures
            self.tile_num = tile_num

        def compute_best_target(self):

            total_prob = 0.0
            total_num_galxies = 0
            total_fibers = 370 * self.num_exposures

            contained_pixels = []
            for pi in self.enclosed_pixel_indices:
                # some pixels will be outside of our localization...
                if pi in pixel_dict:
                    contained_pixels.append(pixel_dict[pi])

            # contained_pixels = [pixel_dict[pi] for pi in self.enclosed_pixel_indices]

            ## TEST
            # test_n1_pixels = [9001088, 9066655, 9078943, 9066656, 9058477, 9029757,
            #                           9029758]


            # total_fibers = 370 * self.num_exposures

            # N3 TESTS
            # total_fibers = 22  # no edge case
            # total_fibers = 23  # missing second exposure
            # total_fibers = 13  # missing second/third exposure
            # total_fibers = 14  # missing third exposure

            # N2 TESTS
            # total_fibers = 14  # no edge case
            # total_fibers = 15  # missing second exposure

            # total_fibers = 5 * self.num_exposures
            # 15/10



            if self.num_exposures == 1:
                # get N=1 list
                n1_galaxies = []
                n1_weights = []
                for p in contained_pixels:
                    gals = list(p.get_available_galaxies_by_multiplicity(1))
                    n1_galaxies += gals
                    for g in gals:
                        n1_weights.append(g.compute_weight(1))

                ordered_indices_n1 = (-np.asarray(n1_weights)).argsort()
                top_galaxies_n1 = list(np.asarray(n1_galaxies)[ordered_indices_n1])
                top_weights_n1 = list(np.asarray(n1_weights)[ordered_indices_n1])

                # Get final list.
                ordered_galaxies = OrderedDict()
                for g in top_galaxies_n1:
                    if g.galaxy_id not in ordered_galaxies:
                        ordered_galaxies[g.galaxy_id] = 0
                    ordered_galaxies[g.galaxy_id] += 1

                # # DEBUG
                # pprint.pprint(ordered_galaxies)

                final_sample = []
                final_count = 0
                for gal_id, multiplicity in ordered_galaxies.items():
                    if (final_count + multiplicity) <= total_fibers:
                        final_sample.append((gal_id, multiplicity))
                        final_count += multiplicity
                    else:
                        continue

                for s in final_sample:
                    total_num_galxies += 1

                    gal_id = s[0]
                    num_exposures = s[1]

                    g = galaxy_dict[gal_id]
                    total_prob += g.prob_fraction
                    g.increment_exps(num_exposures)


            elif self.num_exposures == 2:

                # test_n1_pixels = [9001088, 9066655, 9078943, 8927357, 8886405, 8964247]
                # test_n2_pixels = [9058458, 8931498, 8984738]
                #
                # contained_pixels = []
                # for i in test_n1_pixels:
                #     contained_pixels.append(pixel_dict[i])
                # for i in test_n2_pixels:
                #     contained_pixels.append(pixel_dict[i])

                # get N=1 list
                n1_galaxies = []
                n1_weights = []
                for p in contained_pixels:
                    gals = list(p.get_available_galaxies_by_multiplicity(1))
                    n1_galaxies += gals
                    for g in gals:
                        n1_weights.append(g.compute_weight(1))

                ordered_indices_n1 = (-np.asarray(n1_weights)).argsort()
                top_galaxies_n1 = list(np.asarray(n1_galaxies)[ordered_indices_n1])
                top_weights_n1 = list(np.asarray(n1_weights)[ordered_indices_n1])

                # get entire N=2 list
                n2_galaxies = []
                n2_weights = []
                for p in contained_pixels:
                    gals = list(p.get_available_galaxies_by_multiplicity(2))
                    n2_galaxies += gals
                    for g in gals:
                        n2_weights.append(g.compute_weight(2))

                ordered_indices_n2 = (-np.asarray(n2_weights)).argsort()
                top_galaxies_n2 = list(np.asarray(n2_galaxies)[ordered_indices_n2])
                top_weights_n2 = list(np.asarray(n2_weights)[ordered_indices_n2])


                # DEBUG
                # print(top_weights_n1)
                # print("\n")
                # print(top_weights_n2)
                # print("\n")


                for i, n2 in enumerate(top_weights_n2):

                    found = False

                    for j, (w1, w2) in enumerate(zip(top_weights_n1[:-1], top_weights_n1[1:])):
                        combined_weight = w1 + w2

                        if w1 == 0.0:
                            continue
                        elif n2 > combined_weight:
                            top_weights_n1.insert(j, top_weights_n2[i])
                            top_weights_n1.insert(j + 1, 0.0) # place holder

                            top_galaxies_n1.insert(j, top_galaxies_n2[i])
                            top_galaxies_n1.insert(j + 1, top_galaxies_n2[i])

                            found = True
                            break

                    if not found:
                        top_galaxies_n1.append(top_galaxies_n2[i])
                        top_galaxies_n1.append(top_galaxies_n2[i])

                        top_weights_n1.append(top_weights_n2[i])
                        top_weights_n1.append(0.0)

                # print("\n")
                # print(top_weights_n1)

                # Get final list.
                ordered_galaxies = OrderedDict()
                for g in top_galaxies_n1:
                    if g.galaxy_id not in ordered_galaxies:
                        ordered_galaxies[g.galaxy_id] = 0
                    ordered_galaxies[g.galaxy_id] += 1

                # # DEBUG
                # pprint.pprint(ordered_galaxies)

                final_sample = []
                final_count = 0
                for gal_id, multiplicity in ordered_galaxies.items():
                    if (final_count + multiplicity) <= total_fibers:
                        final_sample.append((gal_id, multiplicity))
                        final_count += multiplicity
                    else:
                        continue

                for s in final_sample:
                    total_num_galxies += 1

                    gal_id = s[0]
                    num_exposures = s[1]

                    g = galaxy_dict[gal_id]
                    total_prob += g.prob_fraction
                    g.increment_exps(num_exposures)


                # print(final_sample)
                # print(final_count)
                # test = 1


            elif self.num_exposures == 3:

                # test_n1_pixels = [9001088, 9066655, 9078943, 9066656, 9058477, 9029757]
                # test_n2_pixels = [9058458, 8972436, 8927368]
                # test_n3_pixels = [8607874, 9033873]
                #
                # contained_pixels = []
                # for i in test_n1_pixels:
                #     contained_pixels.append(pixel_dict[i])
                # for i in test_n2_pixels:
                #     contained_pixels.append(pixel_dict[i])
                # for i in test_n3_pixels:
                #     contained_pixels.append(pixel_dict[i])

                # get N=1 list x3
                n1_galaxies = []
                n1_weights = []
                for p in contained_pixels:
                    gals = list(p.get_available_galaxies_by_multiplicity(1))
                    n1_galaxies += gals
                    for g in gals:
                        n1_weights.append(g.compute_weight(1))

                ordered_indices_n1 = (-np.asarray(n1_weights)).argsort()
                top_galaxies_n1 = list(np.asarray(n1_galaxies)[ordered_indices_n1])
                top_weights_n1 = list(np.asarray(n1_weights)[ordered_indices_n1])

                # get entire N=2 list -- assuming fewer than 3x370 galaxies in this bin...
                n2_galaxies = []
                n2_weights = []
                for p in contained_pixels:
                    gals = list(p.get_available_galaxies_by_multiplicity(2))
                    n2_galaxies += gals
                    for g in gals:
                        n2_weights.append(g.compute_weight(2))

                ordered_indices_n2 = (-np.asarray(n2_weights)).argsort()
                top_galaxies_n2 = list(np.asarray(n2_galaxies)[ordered_indices_n2])
                top_weights_n2 = list(np.asarray(n2_weights)[ordered_indices_n2])

                # get entire N=3 list -- assuming fewer than 3x370 galaxies in this bin...
                n3_galaxies = []
                n3_weights = []
                for p in contained_pixels:
                    gals = list(p.get_available_galaxies_by_multiplicity(3))
                    n3_galaxies += gals
                    for g in gals:
                        n3_weights.append(g.compute_weight(3))

                ordered_indices_n3 = (-np.asarray(n3_weights)).argsort()
                top_galaxies_n3 = list(np.asarray(n3_galaxies)[ordered_indices_n3])
                top_weights_n3 = list(np.asarray(n3_weights)[ordered_indices_n3])

                # print(top_weights_n1)
                # print("\n")
                # print(top_weights_n2)
                # print("\n")
                # print(top_weights_n3)

                # bin running_top_galaxies by twos and check if any N=2 galaxy is > any two N=1 galaxies
                for i, n2 in enumerate(top_weights_n2):

                    found = False

                    for j, (w1, w2) in enumerate(zip(top_weights_n1[:-1], top_weights_n1[1:])):
                        combined_weight = w1 + w2

                        if w1 == 0.0:
                            continue
                        elif n2 > combined_weight:
                            top_weights_n1.insert(j, top_weights_n2[i])
                            top_weights_n1.insert(j + 1, 0.0) # place holder

                            top_galaxies_n1.insert(j, top_galaxies_n2[i])
                            top_galaxies_n1.insert(j + 1, top_galaxies_n2[i])

                            found = True
                            break

                    if not found:
                        top_galaxies_n1.append(top_galaxies_n2[i])
                        top_galaxies_n1.append(top_galaxies_n2[i])

                        top_weights_n1.append(top_weights_n2[i])
                        top_weights_n1.append(0.0)

                # bin running_top_galaxies by twos and check if any N=2 galaxy is > any two N=1 galaxies
                for i, n3 in enumerate(top_weights_n3):

                    found = False

                    for j, (w1, w2, w3) in enumerate(zip(top_weights_n1[:-2], top_weights_n1[1:-1], top_weights_n1[2:])):
                        combined_weight = w1 + w2 + w3
                        if w1 == 0.0:
                            # Don't check vs the w2 placeholder
                            continue
                        elif n3 > combined_weight:

                            top_weights_n1.insert(j, top_weights_n3[i])
                            top_weights_n1.insert(j + 1, 0.0) # place holder
                            top_weights_n1.insert(j + 2, 0.0)  # place holder

                            top_galaxies_n1.insert(j, top_galaxies_n3[i])
                            top_galaxies_n1.insert(j + 1, top_galaxies_n3[i])
                            top_galaxies_n1.insert(j + 2, top_galaxies_n3[i])

                            found = True
                            break

                    if not found:
                        top_galaxies_n1.append(top_galaxies_n3[i])
                        top_galaxies_n1.append(top_galaxies_n3[i])
                        top_galaxies_n1.append(top_galaxies_n3[i])

                        top_weights_n1.append(top_weights_n3[i])
                        top_weights_n1.append(0.0)
                        top_weights_n1.append(0.0)

                # print("\n")
                # print(top_weights_n1)

                # Get final list.
                ordered_galaxies = OrderedDict()
                for g in top_galaxies_n1:
                    if g.galaxy_id not in ordered_galaxies:
                        ordered_galaxies[g.galaxy_id] = 0
                    ordered_galaxies[g.galaxy_id] += 1

                # pprint.pprint(ordered_galaxies)
                final_sample = []
                final_count = 0
                for gal_id, multiplicity in ordered_galaxies.items():
                    if (final_count + multiplicity) <= total_fibers:
                        final_sample.append((gal_id, multiplicity))
                        final_count += multiplicity
                    else:
                        continue

                for s in final_sample:
                    total_num_galxies += 1

                    gal_id = s[0]
                    num_exposures = s[1]

                    g = galaxy_dict[gal_id]
                    total_prob += g.prob_fraction
                    g.increment_exps(num_exposures)

                test = 1








            else:
                raise Exception("Too many exposures!")

            return total_num_galxies, total_prob


            # Read region file
    filename = 'test2.reg'
    filepath = path_format.format(ps1_strm_dir, filename)

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

                aaomega_tiles.append(AAOmega_Tile(c.ra.degree, c.dec.degree, map_nside, radius_deg, num_exposures=exp_num,
                                                  tile_num=tile_num))
                # aaomega_tiles.append(AAOmega_Tile(c.ra.degree, c.dec.degree, map_nside, radius_deg, num_exposures=3))
                # # aaomega_tiles.append(AAOmega_Tile(c.ra.degree, c.dec.degree, map_nside, radius_deg, num_exposures=2))
                # break
            elif "polygon" in l:
                poly_vertices = []

                # EX: polygon(1:28:03.984,-33:49:25.649, ... 1:28:14.531,-33:46:44.047)
                tokens = l.replace("polygon(", "").replace(")\n", "").split(",")

                i = 0
                while i < len(tokens):
                    ra = tokens[i]
                    dec = tokens[i + 1]
                    c = coord.SkyCoord(ra, dec, unit=(u.hour, u.deg))
                    poly_vertices.append([c.ra.degree, c.dec.degree])
                    i += 2

                localization_poly.append(SQL_Polygon([geometry.Polygon(poly_vertices)], aaomega_detector))


    total_exps = 0
    total_slews = 0
    sortedTiles = sorted(aaomega_tiles, key=lambda x: x.tile_num)
    all_gals, all_prob = 0, 0
    for t in sortedTiles:
        total_slews += 1
        total_exps += t.num_exposures

        gals, prob = t.compute_best_target()
        all_gals += gals
        all_prob += prob
        print("Tile #%s (%s exps): %s; %s" % (t.tile_num, t.num_exposures, gals, prob))

    print("\n")
    print(all_gals, all_prob)
    total_hours = (total_exps*40 + total_slews*15)/60.
    total_nights = total_hours/11.0
    print("Total hours: %0.2f; total 11 hour nights: %0.2f" % (total_hours, total_nights))


    # gals1, prob1 = aaomega_tiles[0].compute_best_target()
    #
    # gals2, prob2 = aaomega_tiles[1].compute_best_target()
    # print(gals2, prob2)
    # #
    # print(gals1+gals2, prob1+prob2)
    # print("\n\n\n\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    raise Exception("Strop")







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

    for pix_index, p in pixel_dict.items():
        p.plot(m, ax, edgecolor="black", facecolor="None", linewidth="0.5", alpha=0.15)

    for l in localization_poly:
        l.plot(m, ax, edgecolor='green', linewidth=1.5, facecolor='None')

    for t in aaomega_tiles:
        t.plot(m, ax, edgecolor="red", facecolor="None", linewidth="1.0")



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

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)

    ax.invert_xaxis()

    plt.ylabel(r'$\mathrm{Declination}$', fontsize=label_size, labelpad=36)
    plt.xlabel(r'$\mathrm{Right\;Ascension}$', fontsize=label_size, labelpad=30)
    # endregion

    fig.savefig('GW190814_PS1_region_test.png', bbox_inches='tight')  # ,dpi=840
    plt.close('all')
    print("... Done.")



