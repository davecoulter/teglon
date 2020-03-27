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

load_crossmatch = True
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








