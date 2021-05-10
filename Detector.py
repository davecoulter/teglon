import numpy as np
import healpy as hp
from astropy import units as u
import astropy.coordinates as coord
from Contour import distance_metric_squared, rbf_interpolate
import math
from Tile import *
from astropy.coordinates import Distance
import healpy as hp
from astropy.cosmology import WMAP5, WMAP7, LambdaCDM
from shapely.geometry import Point
from matplotlib.patches import CirclePolygon
from shapely.ops import transform as shapely_transform
from ligo.skymap import distance
from scipy.special import erf
import copy
from Tile import *
from Pixel_Element import *
from SQL_Polygon import *
import pprint
import time
import astropy.time
import math
from matplotlib.patches import Polygon
from shapely import geometry
from mpl_toolkits.basemap import Basemap

from configparser import RawConfigParser
import mysql.connector
from mysql.connector import Error
import os
import urllib.request
import requests
import json
import csv
from shapely.geometry import JOIN_STYLE
import pickle
import MySQLdb as my
import sys

import multiprocessing as mp
from scipy import spatial


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
configFile = os.path.join(__location__, 'Settings.ini')

db_config = RawConfigParser()
db_config.read(configFile)

db_name = db_config.get('database', 'DATABASE_NAME')
db_user = db_config.get('database', 'DATABASE_USER')
db_pwd = db_config.get('database', 'DATABASE_PASSWORD')
db_host = db_config.get('database', 'DATABASE_HOST')
db_port = db_config.get('database', 'DATABASE_PORT')

# region db CRUD
def query_db(query_list, commit=False):
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
    except Error as e:
        print('Error:', e)
    finally:
        cursor.close()
        db.close()

    return results

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
# endregion






# def vertex_rotation_matrix(theta_deg):
#     theta_rad = np.deg2rad(theta_deg)
#     return np.matrix([
#         [np.cos(theta_rad), -np.sin(theta_rad)],
#         [np.sin(theta_rad), np.cos(theta_rad)]
#     ])

# Project collection of input polygons, in degrees, centered at the origin, of form:
#    [
#      (ra_1, dec_1), (ra_2, dec_2), ... (ra_N, dec_N)
#    ]
# to desired RA/Dec (ra_0, dec_0), and rotate by theta_0 degrees. Return projected polygon.
# def project_vertices(input_polygon_list, ra_0, dec_0, theta_0):
#
#     output_list = []
#
#     for vertex_collection in input_polygon_list:
#         # rotation matrix is before vector to reverse the rotation -- Position Angle is defined as degrees East of North
#         rotated = list(map(tuple, map(lambda vertex: np.asarray(vertex_rotation_matrix(theta_0) @ vertex)[0], vertex_collection)))
#         translated = list(map(lambda vertex: (vertex[0] + ra_0, vertex[1] + dec_0), rotated))
#
#         output_list.append(translated)
#
#     return output_list

# def generate_mysql_multipolygon_string_from_shapely_poly(input_shapely_poly):
#     mp_str = "MULTIPOLYGON("
#     multipolygon = []
#     for geom in input_shapely_poly:
#
#         mp = "(("
#         ra_deg, dec_deg = zip(*[(coord_deg[0], coord_deg[1]) for coord_deg in geom.exterior.coords])
#
#         # For the SRS in the DB, we need to emulate lat,lon
#         for i in range(len(ra_deg)):
#             mp += "%s %s," % (dec_deg[i], ra_deg[i] - 180.0)
#
#         mp = mp[:-1]  # trim the last ","
#         mp += ")),"
#         multipolygon.append(mp)
#
#     # Use the multipolygon string to create the WHERE clause
#     multipolygon[-1] = multipolygon[-1][:-1]  # trim the last "," from the last object
#
#     for mp in multipolygon:
#         mp_str += mp
#     mp_str += ")"

class Detector(Teglon_Shape):

    # returns a collection of N=len(resolution) vertex polygon, in degrees, centered at the origin, and
    #   defined in clockwise order from the first-quadrant:
    #    [
    #      (ra_1, dec_1), (ra_2, dec_2), ... (ra_N, dec_N)
    #    ]
    @staticmethod
    def get_detector_vertices_circular(deg_radius, resolution=50):

        # q1 = []
        # q2 = []
        # q3 = []
        # q4 = []
        #
        # x_range = np.linspace(0.0, deg_radius, resolution)
        # y_range = np.sqrt(np.full_like(x_range, 1.0) * deg_radius ** 2.0 - x_range[::-1] ** 2.0)
        #
        # # traversing clockwise through cartesian quadrants
        # for x, y in zip(x_range, y_range[::-1]):
        #     q1.append((x, y))
        #
        # for x, y in zip(x_range[::-1], -1.0 * y_range):
        #     q4.append((x, y))
        #
        # for x, y in zip(-1.0 * x_range, -1.0 * y_range[::-1]):
        #     q3.append((x, y))
        #
        # for x, y in zip(-1.0 * x_range[::-1], y_range):
        #     q2.append((x, y))
        #
        # clockwise_vertices = q1 + q4 + q3 + q2 + [q1[0]]
        #
        # return np.asarray([np.asarray(clockwise_vertices)])

        c1 = Point(0, 0).buffer(deg_radius)
        # c2 = shapely_transform(lambda x, y, z=None: ((self.ra_deg - (self.ra_deg - x) /
        #                                               np.abs(np.cos(np.radians(y)))), y), c1)

        # clip the last 2 coords off to protect against degenerate corners
        return np.asarray([np.asarray([(c[0], c[1]) for c in c1.exterior.coords[:-2]])])

    # returns a collection of N=4 vertex polygon, in degreees, centered at the origin, and
    #   defined in clockwise order from the first-quadrant:
    #    [
    #      (ra_1, dec_1), (ra_2, dec_2), ... (ra_N, dec_N)
    #    ]
    @staticmethod
    def get_detector_vertices_rectangular(deg_width, deg_height):

        x_step = deg_width / 2.0
        y_step = deg_height / 2.0

        clockwise_square = [(x_step, y_step),
                            (x_step, -1.0 * y_step),
                            (-1.0 * x_step, -1.0 * y_step),
                            (-1.0 * x_step, y_step),
                            (x_step, y_step)]

        return np.asarray([np.asarray(clockwise_square)])

    # parses a list of M, N-vertex polygon strings, in degrees, centered at the origin, from Treasure Map of the form:
    #   "POLYGON ((ra_1 dec_1, ra_2 dec_2, ... ra_N dec_N))"
    # returns a collection of M, N-vertex polygons, in degrees, centered at the origin of the form:
    #    [
    #      [(ra_1, dec_1), (ra_2, dec_2), ... (ra_N, dec_N)], ...
    #    ]
    @staticmethod
    def get_detector_vertices_from_treasuremap_footprint_string(string_collection):

        output_polygons = []

        for i, s in enumerate(string_collection):
            polygon_vertices = []

            vertex_str = s.split("((")[1].split("))")[0].split(",")
            vertex_tokens = [v.strip().split(" ") for v in vertex_str]

            for v in vertex_tokens:
                polygon_vertices.append((float(v[0]), float(v[1])))

            output_polygons.append(np.asarray(polygon_vertices))

        return np.asarray(output_polygons)

    @staticmethod
    def get_detector_vertices_from_teglon_db(mp_string):

        output_poly = []
        string_list_of_poly = mp_string.replace('MULTIPOLYGON(', '')[:-1].split(")),((")
        for subpoly in string_list_of_poly:
            polygon_vertices = []
            vertex_str = subpoly.replace("((", "").replace("))", "").split(",")
            vertex_tokens = [v.strip().split(" ") for v in vertex_str]

            for v in vertex_tokens:
                polygon_vertices.append((float(v[0]), float(v[1])))
            output_poly.append(np.asarray(polygon_vertices))

        return np.asarray(output_poly)

    # @staticmethod
    # def create_known_detector(detector_name):
    #     known_detector_characteristics = {
    #         "SWOPE": (0.49493333333333334, 0.4968666666666667, None, 0.24591587555555555, -90, 30),
    #         "ANDICAM": (0.1055288888888889, 0.1055288888888889, None, 0.011136346390123458, -90, 30),
    #         "THACHER": (0.34645333333333334, 0.34645333333333334, None, 0.12002991217777778, -30, 90),
    #         "NICKEL": (0.2093511111111111, 0.2093511111111111, None, 0.04382788772345678, -30, 66.75),
    #         "MOSFIRE": (0.1023333333, 0.1023333333, None, 0.010472111104288888, -70, 90),
    #         "KAIT": (0.1133333333, 0.1133333333, None, 0.01284444443688889, -30, 90),
    #         "SINISTRO": (0.4416666667, 0.4416666667, None, 0.19506944447388888, -90, 90),
    #         "2dF": (None, None, 1.05, 3.4636059006, -90, 30),
    #         "WISE": (0.94833333333, 0.94833333333, None, 0.8993361111, -30, 90)
    #     }
    # 
    #     d = known_detector_characteristics[detector_name]
    #     is_rectangular = d[0] is not None
    #     detector_vertex_collection = None
    # 
    #     if is_rectangular:
    #         detector_deg_width = d[0]
    #         detector_deg_height = d[1]
    #         detector_vertex_collection = Detector.get_detector_vertices_rectangular(detector_deg_width,
    #                                                                                 detector_deg_height)
    #     else:
    #         detector_radius = d[2]
    #         detector_vertex_collection = Detector.get_detector_vertices_circular(detector_radius)
    # 
    #     output_detector = Detector(detector_name, detector_vertex_collection)
    #     return output_detector

    # @staticmethod
    # def create_treasuremap_detector(detector_name, treasuremap_footprint_string_collection):
    #     detector_vertex_collection = Detector.get_detector_vertices_from_treasuremap_footprint_string(
    #         treasuremap_footprint_string_collection)
    #
    #     output_detector = Detector(detector_name, detector_vertex_collection)
    #     return output_detector

    def __init__(self,
                 detector_name,
                 detector_vertex_list_collection,
                 detector_width_deg=None,
                 detector_height_deg=None,
                 detector_radius_deg=None,
                 detector_id=None):

        self.id = detector_id
        self.name = detector_name

        # Width, Height, and Radius are holdovers before we started using polygons to represent Detectors directly.
        # They can be included so that they are visible in the database, but radius and area calculation will happen
        # directly from the polygon itself
        self.deg_width = detector_width_deg
        self.deg_height = detector_height_deg
        self.deg_radius = detector_radius_deg
        self.detector_vertex_list_collection = detector_vertex_list_collection

        self.__multipolygon = None
        self.__query_polygon = None
        self.__query_polygon_string = None
        self.__radius_proxy = None

        running_area = 0.0 # sq deg
        for p in self.multipolygon:
            running_area += p.area
        self.area = running_area
        self.__radius_proxy = np.sqrt(self.area/np.pi)


    @property
    def radius_proxy(self):
        return self.__radius_proxy

    @property
    def multipolygon(self):
        if not self.__multipolygon:
            self.__multipolygon = []
            for vertex_list in self.detector_vertex_list_collection:
                self.__multipolygon.append(geometry.Polygon(vertex_list))
        return self.__multipolygon

    @property
    def projected_multipolygon(self):
        return self.multipolygon

    @property
    def query_polygon(self):
        return self.multipolygon

    @property
    def query_polygon_string(self):
        if not self.__query_polygon_string:
            mp_str = "MULTIPOLYGON("
            multipolygon = []
            for geom in self.multipolygon:

                mp = "(("
                # ra_deg, dec_deg = zip(*[(coord_deg[0], coord_deg[1]) for coord_deg in geom.exterior.coords])
                ra_deg, dec_deg = Teglon_Shape.get_coord_lists(geom, convert_radian=False)

                # Here we don't care about emulating lat/lon. These polygons will only be used downstream
                # to instantiate a Tile and project it to a celestial position
                for i in range(len(ra_deg)):
                    mp += "%s %s," % (ra_deg[i], dec_deg[i])

                mp = mp[:-1]  # trim the last ","
                mp += ")),"
                multipolygon.append(mp)

            # Use the multipolygon string to create the WHERE clause
            multipolygon[-1] = multipolygon[-1][:-1]  # trim the last "," from the last object

            for mp in multipolygon:
                mp_str += mp
            mp_str += ")"

            self.__query_polygon_string = mp_str

        return self.__query_polygon_string

    def __str__(self):
        return str(self.__dict__)






is_debug = False
if is_debug:
    # HACK - these are both diagnostic AND actual update/insert commands. Refactor this to it's own initialization file
    update_detectors = False
    query_treasure_map_detectors = False
    build_sql_poly = False

    start = time.time()

    if update_detectors:
        detector_select = '''
            SELECT id, Name, Deg_width, Deg_height, Deg_radius FROM Detector;
        '''

        detector_update = "UPDATE Detector SET Poly = ST_GEOMFROMTEXT('%s', 4326) WHERE id = %s"

        detectors_update_objs = []
        detector_result = query_db([detector_select])[0]
        for d in detector_result:
            d_id = int(d[0])
            d_name = d[1]
            d_width = d[2]
            d_height = d[3]
            d_radius = d[4]

            # Sanity
            if d_width is None and d_height is None and d_radius is None:
                # skip these -- these are not "known detectors" -- these are Treasuremap Detectors
                continue

            is_rectangular = d[2] is not None

            if is_rectangular:
                detector_deg_width = float(d_width)
                detector_deg_height = float(d_height)
                detector_vertex_collection = Detector.get_detector_vertices_rectangular(detector_deg_width,
                                                                                        detector_deg_height)
            else:
                detector_radius = float(d_radius)
                detector_vertex_collection = Detector.get_detector_vertices_circular(detector_radius)

            output_detector = Detector(d_name, detector_vertex_collection)
            detectors_update_objs.append((d_id, output_detector.query_polygon_string))

        for d in detectors_update_objs:
            q = detector_update % (d[1], d[0])
            query_db([q], commit=True)

    def get_skycoord_from_tm_point(tm_point_string):
        ra_dec_list = tm_point_string.split("(")[1].split(")")[0].split(" ")

        ra = float(ra_dec_list[0])
        dec = float(ra_dec_list[1])

        return ra, dec

    # Some constants used in testing
    # 0425 map ID
    healpix_map_id = 8
    healpix_map_nside = 256
    ztf_id = 47
    xrt_id = 13
    ztf_pa = 0.0
    bat_fov_id = 49

    sql_inner_poly = None
    sql_outer_poly = None
    if build_sql_poly:
        map_pixel_select = '''
            SELECT 
                id, HealpixMap_id, Pixel_Index, Prob, Distmu, Distsigma, Distnorm, Mean, Stddev, Norm, N128_SkyPixel_id 
            FROM HealpixPixel 
            WHERE HealpixMap_id = %s;'''
        map_pix_result = query_db([map_pixel_select % healpix_map_id])[0]

        map_pix_objs = [Pixel_Element(int(mp[2]), healpix_map_nside, float(mp[3]), pixel_id=int(mp[0]))
                   for mp in map_pix_result]

        map_pix_sorted = sorted(map_pix_objs, key=lambda p: p.prob, reverse=True)

        cutoff_inner = 0.50
        cutoff_outer = 0.90
        index_inner = 0
        index_outer = 0

        _90th_pix_indices = []

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
        smoothed_inner_poly = joined_inner_poly.buffer(eps, 1, join_style=JOIN_STYLE.mitre).buffer(-eps, 1,
                                                                                                   join_style=JOIN_STYLE.mitre)
        try:
            test_iter = iter(smoothed_inner_poly)
            merged_inner_poly = smoothed_inner_poly
        except TypeError as te:
            merged_inner_poly.append(smoothed_inner_poly)

        print("Number of sub-polygons in `merged_inner_poly`: %s" % len(merged_inner_poly))
        sql_inner_poly = SQL_Polygon(merged_inner_poly)

        net_outer_polygon = []
        for p in map_pix_sorted[0:index_outer]:
            _90th_pix_indices.append(p.index)
            net_outer_polygon += p.query_polygon
        joined_outer_poly = unary_union(net_outer_polygon)

        # Fix any seams
        merged_outer_poly = []
        smoothed_outer_poly = joined_outer_poly.buffer(eps, 1, join_style=JOIN_STYLE.mitre).buffer(-eps, 1,
                                                                                                   join_style=JOIN_STYLE.mitre)
        try:
            test_iter = iter(smoothed_outer_poly)
            merged_outer_poly = smoothed_outer_poly
        except TypeError as te:
            merged_outer_poly.append(smoothed_outer_poly)

        print("Number of sub-polygons in `merged_outer_poly`: %s" % len(merged_outer_poly))
        sql_outer_poly = SQL_Polygon(merged_outer_poly)
        print("... done.")



        with open('sql_inner_poly_test.pkl', 'wb') as handle:
            pickle.dump(sql_inner_poly, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('sql_outer_poly_test.pkl', 'wb') as handle:
            pickle.dump(sql_outer_poly, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('sql_inner_poly_test.pkl', 'rb') as handle:
            sql_inner_poly = pickle.load(handle)
            del handle

        with open('sql_outer_poly_test.pkl', 'rb') as handle:
            sql_outer_poly = pickle.load(handle)
            del handle

    treasure_map_detectors = None
    treasure_map_pointings = None

    if query_treasure_map_detectors:

        api_token = "jOwrpZgnXU-nLVi7r8bhL2wliAI0ZXi5IZUudQ"
        base_url = 'http://treasuremap.space/api/v0'

        # get all photometric instruments
        instrument_url = 'instruments'
        instrument_synopsis_request = {
            "api_token": api_token,
            "type": "photometric"
        }
        instrument_list_target_url = "{}/{}?{}".format(base_url, instrument_url,
                                                       urllib.parse.urlencode(instrument_synopsis_request))
        instrument_response = requests.get(url=instrument_list_target_url)
        instrument_result = [json.loads(r) for r in json.loads(instrument_response.text)]
        instrument_names_full = {ir["id"]: (ir["instrument_name"], ir['nickname']) for ir in instrument_result}
        instrument_names = {}
        for inst_id, inst_name_tuple in instrument_names_full.items():
            full_name = inst_name_tuple[0]
            short_name = inst_name_tuple[1]

            if short_name.strip() == '':
                instrument_names[inst_id] = full_name
            else:
                if len(full_name) > 20:
                    instrument_names[inst_id] = short_name
                else:
                    instrument_names[inst_id] = full_name

        # get all detector footprints
        footprint_url = 'footprints'
        instrument_detail_request = {
            "api_token": api_token
        }
        instrument_detail_target_url = "{}/{}?{}".format(base_url, footprint_url,
                                                         urllib.parse.urlencode(instrument_detail_request))
        instrument_detail_response = requests.get(url=instrument_detail_target_url)
        instrument_detail_result = [json.loads(r) for r in json.loads(instrument_detail_response.text)]

        instrument_footprints = {}
        for idr in instrument_detail_result:
            if idr["instrumentid"] not in instrument_footprints:
                instrument_footprints[idr["instrumentid"]] = []
            instrument_footprints[idr["instrumentid"]].append(idr["footprint"])

        instrument_vertices = {}
        for inst_id, inst_fp in instrument_footprints.items():
            vertext_collection_list = Detector.get_detector_vertices_from_treasuremap_footprint_string(inst_fp)
            instrument_vertices[inst_id] = vertext_collection_list

        treasure_map_detectors = {}
        for inst_id, inst_vert in instrument_vertices.items():

            # Sanity -- TM returns instruments that don't exist on the detail pages...
            # ex: instrument id = 21 is returned from the footprint query, however, navigating to:
            #   http://gwtmlb-1777941625.us-east-2.elb.amazonaws.com/instrument_info?id=21
            # results in a 404. Omit these results.
            if inst_id not in instrument_names:
                continue

            # Swift XRT doesn't have any limiting mags for 0425
            if inst_id == xrt_id:
                continue

            name = instrument_names[inst_id]
            tm_detector = Detector(name, inst_vert)

            treasure_map_detectors[inst_id] = tm_detector


        insert_tm_detectors = False
        if insert_tm_detectors:
            insert_tm = "INSERT INTO Detector (Name, Poly, Area, TM_id) VALUES ('%s', ST_GEOMFROMTEXT('%s', 4326), %s, %s);"

            for inst_id, tm_detect in treasure_map_detectors.items():
                q = insert_tm % (tm_detect.name, tm_detect.query_polygon_string, tm_detect.area, inst_id)
                query_db([q], commit=True)


        # get all detector pointings
        pointing_url = 'pointings'
        pointing_request = {
            "api_token": api_token,
            "status": "completed",
            # Omitting bands includes all bands...
            # "bands": [
            #     "U", "B", "V", "g", "r", "i"
            # ],
            "graceid": "GW190425"
        }
        pointing_target_url = "{}/{}?{}".format(base_url, pointing_url,
                                                urllib.parse.urlencode(pointing_request))
        pointing_response = requests.get(url=pointing_target_url)
        pointing_result = [json.loads(r) for r in json.loads(pointing_response.text)]

        treasure_map_pointings = {}
        for pr in pointing_result:
            pointing_id = pr["id"]
            inst_id = pr["instrumentid"]
            pa = pr["pos_angle"]
            band = pr["band"]
            obs_time = pr["time"]
            mag_lim = pr["depth"]


            # Hack -- we don't want to plot the Swift BAT field of view. Also, XRT has no "depth" stat. Skip these...
            if inst_id == bat_fov_id or inst_id == xrt_id:
                continue

            # HACK -- ZTF is reported at a PA=45 degrees; correct this to PA=0
            if inst_id == ztf_id:
                pa = ztf_pa

            if inst_id not in treasure_map_pointings:
                treasure_map_pointings[inst_id] = []

            treasure_map_pointings[inst_id].append((pa, get_skycoord_from_tm_point(pr["position"]), band, obs_time,
                                                    pointing_id, mag_lim))

        with open('tm_detector_test.pkl', 'wb') as handle:
            pickle.dump(treasure_map_detectors, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('tm_pointing_test.pkl', 'wb') as handle:
            pickle.dump(treasure_map_pointings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        treasure_map_detectors = {}

        # Get from pickle...
        # with open('tm_detector_test.pkl', 'rb') as handle:
        #     treasure_map_detectors = pickle.load(handle)
        #     del handle

        # Get from db...
        # Select TM Detectors from DB and hydrate into Detectors with correct origin-centered polygons
        select_tm_detect = '''
            SELECT Name, ST_AsText(Poly), TM_id FROM Detector WHERE TM_id IS NOT NULL;
        '''
        tm_detect_result = query_db([select_tm_detect])[0]
        for tm_d in tm_detect_result:
            tm_name = tm_d[0]
            tm_poly = tm_d[1]
            tm_id = int(tm_d[2])
            tm_vertices = Detector.get_detector_vertices_from_teglon_db(tm_poly)
            tm_detect = Detector(tm_name, tm_vertices)
            treasure_map_detectors[tm_id] = tm_detect

        with open('tm_pointing_test.pkl', 'rb') as handle:
            treasure_map_pointings = pickle.load(handle)
            del handle

    treasure_map_tiles = {}
    for inst_id, tm_pointing_list in treasure_map_pointings.items():
        detect = treasure_map_detectors[inst_id]

        for tm_pointing_tuple in tm_pointing_list:
            pa = tm_pointing_tuple[0]
            ra, dec = tm_pointing_tuple[1]

            ra_0 = ra
            if ra_0 < 0:
                ra_0 += 360.0
            dec_0 = dec

            if inst_id not in treasure_map_tiles:
                treasure_map_tiles[inst_id] = []

            t = Tile(ra_0, dec_0, detect, healpix_map_nside, position_angle_deg=pa)
            treasure_map_tiles[inst_id].append(t)


    # # Parse the data that MC gave us...
    # ztf_tiles = "./Events/S190425z/ObservedTiles/GW190425_allfields_data.csv"
    # ztf_ra = []
    # ztf_dec = []

    # with open('%s' % ztf_tiles, 'r') as csvfile:
    #     csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
    #     # Skip Header
    #     next(csvreader)
    #
    #     for row in csvreader:
    #         ra = float(row[5])
    #         dec = float(row[6])
    #
    #         ztf_ra.append(ra)
    #         ztf_dec.append(dec)
    #
    # phi, theta = np.radians(ztf_ra), 0.5 * np.pi - np.radians(ztf_dec)
    # pix_i = hp.ang2pix(256, theta, phi)
    # ztf_indices = {}
    # for i, pi in enumerate(pix_i):
    #     ztf_indices[pi] = (ztf_ra[i], ztf_dec[i])
    # _90th_pointings = list(set(pix_i).intersection(set(_90th_pix_indices)))
    #
    # for _9pp in _90th_pointings:
    #     ra_dec_tup = ztf_indices[_9pp]
    #     instr_pointings[ztf_id].append((0.0, ra_dec_tup))

    # projected_pointings = {}
    #
    # bat_fov_id = 49
    # for inst_id, inst_point in treasure_map_pointings.items():
    #
    #     # skip Swift/BAT FOV
    #     if inst_id == bat_fov_id:
    #         continue
    #
    #     if inst_id not in projected_pointings:
    #         projected_pointings[inst_id] = []
    #
    #     instrument_footprint = instrument_vertices[inst_id]
    #
    #     for ip in inst_point:
    #         theta_0 = ip[0]
    #         ra_0 = ip[1][0]
    #         if ra_0 < 0:
    #             ra_0 += 360.0
    #         dec_0 = ip[1][1]

            # projected_pointings[inst_id].append(project_vertices(instrument_footprint, ra_0, dec_0, theta_0))

    clrs = {
        12: "cyan",
        47: "magenta",
        71: "pink",
        22: "yellow",
        11: "royalblue",
        13: "orange"
    }

    do_full_sky = False
    if do_full_sky:
        # fig = plt.figure(figsize=(20, 15))
        # ax = fig.add_subplot(111)
        # m = Basemap(projection='moll', lon_0=180.0)

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
        mwe_pix_result = query_db([select_mwe_pix % ("2.285", "0.5")])[0]
        mwe_pix = [Pixel_Element(int(mp[0]), 32, 0.0) for mp in mwe_pix_result]
        # for p in mwe_pix:
        #     # p.plot(m, ax, edgecolor='cornflowerblue', linewidth=0.5, facecolor='None', alpha=0.15)
        #     p.plot(m, ax, edgecolor='None', linewidth=0.5, facecolor='cornflowerblue', alpha=0.2)
        #
        # sql_inner_poly.plot(m, ax, edgecolor='k', linewidth=0.75, facecolor='None')
        # sql_outer_poly.plot(m, ax, edgecolor='k', linewidth=0.5, facecolor='None')
        #
        # for inst_id, tile_list in treasure_map_tiles.items():
        #
        #     zo = 9999
        #     alph = 1.0
        #     lw = 0.5
        #     if inst_id == ztf_id:
        #         zo = 9998
        #         alph = 0.15
        #         lw=0.2
        #
        #     for t in tile_list:
        #         t.plot(m, ax, edgecolor=clrs[inst_id], facecolor='None', linestyle='-', linewidth=lw, zorder=zo, alpha=alph)
        #
        # parallels = np.arange(-90., 90., 12.)
        # dec_ticks = m.drawparallels(parallels, labels=[0, 1, 0, 0], linewidth=0.5, color="silver", alpha=0.25)
        # meridians = np.arange(0., 360., 24.0)
        # ra_ticks = m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0.5, color="silver", alpha=0.25)
        #
        # ax.invert_xaxis()
        #
        # fig.savefig("GW0425_TM_data.png", bbox_inches='tight')
        # plt.close('all')


        ############################
        # select_mwe_pix = '''
        #                             SELECT sp.Pixel_Index
        #                             FROM SkyPixel sp
        #                             WHERE sp.id IN
        #                             (
        #                                 SELECT sp.Parent_Pixel_id
        #                                 FROM SkyPixel sp
        #                                 WHERE sp.id IN
        #                                 (
        #                                     SELECT sp.Parent_Pixel_id
        #                                     FROM SkyPixel sp
        #                                     JOIN SkyPixel_EBV sp_ebv ON sp_ebv.N128_SkyPixel_id = sp.id
        #                                     WHERE sp_ebv.EBV*%s > %s and NSIDE = 128
        #                                 ) and NSIDE = 64
        #                             ) and NSIDE = 32
        #                         '''
        # mwe_pix_result = query_db([select_mwe_pix % ("2.285", "0.5")])[0]
        # mwe_pix = [Pixel_Element(int(mp[0]), 32, 0.0) for mp in mwe_pix_result]
        # print(len(mwe_pix))
        #
        #
        # fig = plt.figure(figsize=(10, 10), dpi=1000)
        # ax = fig.add_subplot(111)
        # m = Basemap(projection='ortho', lon_0=240.0, lat_0=20.0)
        #
        #
        # for p in mwe_pix:
        #     # _x, _y = m(p.coord.ra.degree, p.coord.dec.degree)
        #     # m.plot(_x, _y, marker=".", color="cornflowerblue", alpha=0.10, markersize=20.0)
        #     # lon_0 = 240.0, lat_0 = 20.0
        #     if p.coord.ra.degree < 345 and p.coord.ra.degree > 195 and p.coord.dec.degree < 85.0 and p.coord.dec.degree > -60:
        #         p.plot(m, ax, edgecolor='None', linewidth=0.5, facecolor='cornflowerblue', alpha=0.20)
        #         # m.plot(_x, _y, marker=".", color="cornflowerblue", alpha=0.10)
        #
        # sql_inner_poly.plot(m, ax, edgecolor='k', linewidth=0.75, facecolor='None')
        # sql_outer_poly.plot(m, ax, edgecolor='k', linewidth=0.5, facecolor='None')
        # for inst_id, tile_list in treasure_map_tiles.items():
        #
        #     zo = 9999
        #     alph = 1.0
        #     lw = 0.5
        #     if inst_id == ztf_id:
        #         zo = 9998
        #         alph = 0.15
        #         lw=0.2
        #
        #     for t in tile_list:
        #         t.plot(m, ax, edgecolor=clrs[inst_id], facecolor='None', linestyle='-', linewidth=lw, zorder=zo, alpha=alph)
        #
        # parallels = np.arange(-90., 90., 12.)
        # dec_ticks = m.drawparallels(parallels, labels=[0, 1, 0, 0], linewidth=0.5, color="silver", alpha=0.25)
        # meridians = np.arange(0., 360., 24.0)
        # ra_ticks = m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0.5, color="silver", alpha=0.25)
        #
        # ax.invert_xaxis()
        #
        # fig.savefig("GW0425_TM_eastern_ortho.png", bbox_inches='tight')
        # plt.close('all')
        #
        fig = plt.figure(figsize=(10, 10), dpi=1000)
        ax = fig.add_subplot(111)
        m = Basemap(projection='ortho', lon_0=60.0, lat_0=-30.0)

        sql_inner_poly.plot(m, ax, edgecolor='k', linewidth=0.75, facecolor='None')
        sql_outer_poly.plot(m, ax, edgecolor='k', linewidth=0.5, facecolor='None')

        for p in mwe_pix:
            # or (p.coord.ra.degree > 359.999 and p.coord.ra.degree < 360)
            if ((p.coord.ra.degree < 130 and p.coord.ra.degree >= 0.0)) and p.coord.dec.degree < 48.0 and p.coord.dec.degree > -90:
                p.plot(m, ax, edgecolor='None', linewidth=0.5, facecolor='cornflowerblue', alpha=0.20)

        for inst_id, tile_list in treasure_map_tiles.items():
            zo = 9999
            alph = 1.0
            lw = 0.5
            if inst_id == ztf_id:
                zo = 9998
                alph = 0.15
                lw = 0.2

            for t in tile_list:
                t.plot(m, ax, edgecolor=clrs[inst_id], facecolor='None', linestyle='-', linewidth=lw, zorder=zo,
                       alpha=alph)

        parallels = np.arange(-90., 90., 12.)
        dec_ticks = m.drawparallels(parallels, labels=[0, 1, 0, 0], linewidth=0.5, color="silver", alpha=0.25)
        meridians = np.arange(0., 360., 24.0)
        ra_ticks = m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0.5, color="silver", alpha=0.25)

        ax.invert_xaxis()

        fig.savefig("GW0425_TM_western_ortho.png", bbox_inches='tight')
        plt.close('all')


    # test random detector tiles...
    do_sterographic_test = False
    if do_sterographic_test:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        lat_0 = -20.0
        lon_0 = 15.0

        m = Basemap(projection='stere',
                    lon_0=lon_0,
                    lat_0=lat_0,
                    # llcrnrlat=-35.0,
                    # urcrnrlat=-19.5,
                    # llcrnrlon=8.0,
                    # urcrnrlon=24.5)
                    llcrnrlat=-37.0,
                    urcrnrlat=-19.5,
                    llcrnrlon=8.0,
                    urcrnrlon=26.5)

        ra_0 = 20.0
        dec_0 = -25.0
        theta_0 = 45.0

        circle = Detector.get_detector_vertices_circular(deg_radius=1.0)
        square = Detector.get_detector_vertices_rectangular(deg_width=1.0, deg_height=1.0)


        circle_detector = Detector(detector_name="Circle", detector_vertex_list_collection=circle)
        square_detector = Detector(detector_name="Square", detector_vertex_list_collection=square)
        # swope_id = 1
        # swope_detector = Detector(swope_name, swope_vertices, detector_id=swope_id)
        #
        # swope_ra = 57.0707203619
        # swope_dec = -89.75155666666666
        # swope_geom = geometry.Polygon(swope_vertices[0])
        # ra_deg, dec_deg = Telgon_Shape.get_coord_lists(swope_geom, convert_radian=False)
        # projected_subpolygon = []
        # for ra, dec in zip(ra_deg, dec_deg):
        #
        #     rotation_matrix = Telgon_Shape.vertex_rotation_matrix(0.0)
        #     rotated = rotation_matrix @ [ra, dec]
        #
        #     rot_ra = rotated[0, 0]
        #     rot_dec = rotated[0, 1]
        #     translated_dec = rot_dec + swope_dec
        #
        #     # transform to spherical projection
        #     translated = [(rot_ra / np.cos(np.radians(translated_dec)) + swope_ra) % 360.0, translated_dec]
        #
        #     projected_subpolygon.append(translated)
        #
        # newPoly = geometry.Polygon(projected_subpolygon)
        #
        # test = 1


        # swope_tile1 = Tile(central_ra_deg=swope_ra, central_dec_deg=swope_dec, detector=swope_detector,
        #                    nside=healpix_map_nside, tile_id=1, tile_mwe=0.35)
        # t = swope_tile1.enclosed_pixel_indices

        # tile_select = "SELECT id, Detector_id, FieldName, RA, _Dec, Coord, Poly, EBV, N128_SkyPixel_id FROM StaticTile WHERE Detector_id = %s;"
        # swope_static_tile_rows = query_db([tile_select % swope_id])[0]
        # swope_tiles = []
        # for r in swope_static_tile_rows:
        #     t = Tile(central_ra_deg=float(r[3]), central_dec_deg=float(r[4]), detector=swope_detector,
        #              nside=healpix_map_nside, tile_id=int(r[0]), tile_mwe=float(r[7]))
        #     swope_tiles.append(t)


        circle_tile1 = Tile(central_ra_deg=ra_0, central_dec_deg=dec_0, detector=circle_detector,
                            nside=healpix_map_nside, position_angle_deg=theta_0)
        square_tile1 = Tile(central_ra_deg=ra_0, central_dec_deg=dec_0, detector=square_detector,
                            nside=healpix_map_nside, position_angle_deg=theta_0)

        ztf_tile1 = Tile(central_ra_deg=ra_0, central_dec_deg=dec_0, detector=treasure_map_detectors[ztf_id],
                         nside=healpix_map_nside, position_angle_deg=theta_0)

        circle_tile1.plot(m, ax, edgecolor='b', facecolor='None', linestyle='-')
        square_tile1.plot(m, ax, edgecolor='r', facecolor='None', linestyle='-')
        # swope_tile1.plot(m, ax, edgecolor='magenta', facecolor='None', linestyle='-')
        ztf_tile1.plot(m, ax, edgecolor='k', facecolor='None', linestyle='-')

        # test shift and rotate
        ra_1 = 13.0
        dec_1 = -32.0
        theta_1 = 0

        ztf_tile2 = Tile(central_ra_deg=ra_1, central_dec_deg=dec_1, detector=treasure_map_detectors[ztf_id],
                         nside=healpix_map_nside, position_angle_deg=theta_1)

        ra_2 = 23
        dec_2 = -32.0
        theta_2 = 20
        circle_tile2 = Tile(central_ra_deg=ra_2, central_dec_deg=dec_2, detector=circle_detector, nside=healpix_map_nside,
                            position_angle_deg=theta_2)

        ra_3 = 22.0
        dec_3 = -34
        theta_3 = 45
        square_tile2 = Tile(central_ra_deg=ra_3, central_dec_deg=dec_3 , detector=square_detector, nside=healpix_map_nside,
                            position_angle_deg=theta_3)

        circle_tile2.plot(m, ax, edgecolor='b', facecolor='None', linestyle=':')
        square_tile2.plot(m, ax, edgecolor='r', facecolor='None', linestyle=':')
        ztf_tile2.plot(m, ax, edgecolor='k', facecolor='None', linestyle=':')

        # plot the enclosed pixels for these tiles...
        for epi in circle_tile2.enclosed_pixel_indices:
            p = Pixel_Element(epi, healpix_map_nside, 0.0)
            p.plot(m, ax, edgecolor='b', facecolor='None', linestyle='-')

        for epi in square_tile2.enclosed_pixel_indices:
            p = Pixel_Element(epi, healpix_map_nside, 0.0)
            p.plot(m, ax, edgecolor='r', facecolor='None', linestyle='-')

        for epi in ztf_tile2.enclosed_pixel_indices:
            p = Pixel_Element(epi, healpix_map_nside, 0.0)
            p.plot(m, ax, edgecolor='k', facecolor='None', linestyle='-')


        parallels = np.arange(-90., 90., 10.)
        dec_ticks = m.drawparallels(parallels, labels=[0, 1, 0, 0])
        meridians = np.arange(0., 360., 7.5)
        ra_ticks = m.drawmeridians(meridians, labels=[0, 0, 0, 1])

        ax.invert_xaxis()

        print("Saving figure...")
        fig.savefig("test_detector_polygons.png", bbox_inches='tight')
        plt.close('all')

    do_swope_static_tiles = False
    if do_swope_static_tiles:

        select_swope_detect = '''
                SELECT id, Name, ST_AsText(Poly) FROM Detector WHERE Name = 'SWOPE';
            '''
        swope_detect_result = query_db([select_swope_detect])[0][0]
        swope_id = swope_detect_result[0]
        swope_name = swope_detect_result[1]
        swope_poly = swope_detect_result[2]
        swope_vertices = Detector.get_detector_vertices_from_teglon_db(swope_poly)
        swope_detector = Detector(swope_name, swope_vertices, detector_id=swope_id)

        tile_select = "SELECT id, Detector_id, FieldName, RA, _Dec, Coord, Poly, EBV, N128_SkyPixel_id FROM StaticTile WHERE Detector_id = %s;"
        swope_static_tile_rows = query_db([tile_select % swope_id])[0]
        swope_tiles = []
        for r in swope_static_tile_rows:
            t = Tile(central_ra_deg=float(r[3]), central_dec_deg=float(r[4]), detector=swope_detector,
                     nside=healpix_map_nside, tile_id=int(r[0]), tile_mwe=float(r[7]))
            swope_tiles.append(t)

        fig = plt.figure(figsize=(20, 20), dpi=800)
        ax = fig.add_subplot(111)
        # m = Basemap(projection='moll', lon_0=180.0)
        lat_0 = 70.0
        lon_0 = 15.0

        m = Basemap(projection='stere',
                    lon_0=lon_0,
                    lat_0=lat_0,
                    llcrnrlat=80.0,
                    urcrnrlat=85.0,
                    llcrnrlon=10.0,
                    urcrnrlon=20.0)

        plotted = []
        for t in swope_tiles:
            if t.dec_deg < 86.0 and t.dec_deg > 79.0 and t.ra_deg < 30.0 and t.ra_deg > 5.0:
                t.plot(m, ax, edgecolor='k', facecolor='None', linestyle='-', linewidth=0.25)
                plotted.append(t)
            # elif t.dec_deg < -80:
            #     t.plot(m, ax, edgecolor='k', facecolor='None', linestyle='-', linewidth=0.25)

        for p in plotted:
            for pp in p.enclosed_pixel_indices:
                ppp = Pixel_Element(pp, healpix_map_nside, 0.0)
                ppp.plot(m, ax, edgecolor='g', facecolor='None', linestyle=':', linewidth=0.25)

        parallels = np.arange(-90., 90., 12.)
        dec_ticks = m.drawparallels(parallels, labels=[0, 1, 0, 0], linewidth=0.5, color="silver", alpha=0.5)
        meridians = np.arange(0., 360., 24.0)
        ra_ticks = m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0.5, color="silver", alpha=0.5)

        ax.invert_xaxis()

        fig.savefig("swope_static_tile_test.png", bbox_inches='tight')
        plt.close('all')


    # Generate Observed Tile lists for TM pointings...
    do_tm_observed_tiles = True
    if do_tm_observed_tiles:

        # DEBUG print all bands for all points:
        unique_bands = []
        for inst_id, pointing_list in treasure_map_pointings.items():

            # what is open?
            for p in pointing_list:
                if p[2] == "open":
                    print(inst_id)
                    break

            unique_bands += list(set([p[2] for p in pointing_list]))
        unique_bands = list(set(unique_bands))
        print(unique_bands)

        # treasure_map_detectors == keyed by TM id
        # treasure_map_pointings == (pa, (ra, dec), band, obs_time_string, pointing_id) keyed by TM id
        treasure_map_detector_for_obs_tile = {} # keyed by teglon detector id
        treasure_map_obs_tile = {}  # keyed by teglon detector id
        tm_to_teglon_detector_map = {
            47: 76,
            11: 62,
            12: 63,
            22: 65,
            71: 59
        }

        select_tm_detect = '''
                    SELECT id, Name, ST_AsText(Poly), TM_id FROM Detector WHERE TM_id IS NOT NULL;
                '''
        tm_detect_result = query_db([select_tm_detect])[0]
        for tm_d in tm_detect_result:
            teglon_id = int(tm_d[0])
            tm_name = tm_d[1]
            tm_poly = tm_d[2]
            tm_id = int(tm_d[3])
            tm_vertices = Detector.get_detector_vertices_from_teglon_db(tm_poly)
            tm_detect = Detector(tm_name, tm_vertices)

            treasure_map_detector_for_obs_tile[teglon_id] = tm_detect

        for inst_id, pointing_list in treasure_map_pointings.items():

            tm_detector = treasure_map_detector_for_obs_tile[tm_to_teglon_detector_map[inst_id]]
            if tm_detector.name not in treasure_map_obs_tile:
                treasure_map_obs_tile[tm_detector.name] = []

            for pointing in pointing_list:
                # treasure_map_pointings[inst_id].append((pa, get_skycoord_from_tm_point(pr["position"]), band, obs_time,
                #                                         pointing_id, mag_lim))
                pa = float(pointing[0])
                _ra_deg = float(pointing[1][0])
                ra_deg = _ra_deg
                if ra_deg < 0:
                    ra_deg += 360.0
                dec_deg = float(pointing[1][1])

                band = pointing[2]
                datetime_string = pointing[3]
                mdj = (astropy.time.Time(datetime_string)).mjd
                pointing_id = int(pointing[4])
                lim_mag = float(pointing[5])

                # file_name, field_name, ra, dec, mjd, band, exp_time, lim_mag, position_angle
                pointing_row = ("treasure_map", pointing_id, ra_deg, dec_deg, mdj, band, 0.0, lim_mag, pa)

                treasure_map_obs_tile[tm_detector.name].append(pointing_row)

        for tm_detect_name, obs_list in treasure_map_obs_tile.items():
            with open("%s_observed_tiles.txt" % tm_detect_name.replace("/", "_"), "w") as csvfile:
                writer = csv.writer(csvfile, delimiter=" ")
                writer.writerows(obs_list)










    end = time.time()
    duration = (end - start)
    print("\n********* start DEBUG ***********")
    print("Teglon `TestDectector` execution time: %s" % duration)
    print("********* end DEBUG ***********\n")









    # # test on basemap
    # fig = plt.figure(figsize=(10, 10), dpi=600)
    # ax = fig.add_subplot(111)
    #
    # lat_0 = -20.0
    # lon_0 = 15.0
    #
    # m = Basemap(projection='stere',
    #             lon_0=lon_0,
    #             lat_0=lat_0,
    #             llcrnrlat=-35.0,
    #             urcrnrlat=-19.5,
    #             llcrnrlon=8.0,
    #             urcrnrlon=24.5)
    #
    # ra_0 = 20.0
    # dec_0 = -25.0
    # theta_0 = 45.0
    #
    # transformed_ztf_2 = project_polygon(ztf_multi_poly, ra_0, dec_0, theta_0)
    # for geom in transformed_ztf_2:
    #     ra_deg, dec_deg = zip(*[(coord_deg[0], coord_deg[1]) for coord_deg in geo_poly.exterior.coords])
    #
    #     x2, y2 = m(ra_deg, dec_deg)
    #     lat_lons = np.vstack([x2, y2]).transpose()
    #     ax.add_patch(Polygon(lat_lons, edgecolor='k', facecolor='None', linestyle='-'))
    #
    # #
    # # ztf_geometries = []
    # # for z_p in ztf_multi_poly:
    # #
    # #     rotated = list(map(tuple, map(lambda vertex: np.asarray(vertex @ rot(theta))[0], z_p)))
    # #     translated = list(map(lambda vertex: (vertex[0]/np.cos(np.radians(vertex[1] + dec_0)) + ra_0,
    # #                                                  vertex[1] + dec_0), rotated))
    # #     geo_poly = geometry.Polygon(translated)
    # #     ztf_geometries.append(geo_poly)
    # #     ra_deg, dec_deg = zip(*[(coord_deg[0], coord_deg[1]) for coord_deg in geo_poly.exterior.coords])
    # #
    # #     x2, y2 = m(ra_deg, dec_deg)
    # #     lat_lons = np.vstack([x2, y2]).transpose()
    # #     ax.add_patch(Polygon(lat_lons, edgecolor='k', facecolor='None', linestyle='-'))
    #
    #
    #
    # # Test the ability to query based on polygon. Retrieve the galaxies
    # all_gals_select = '''
    #     SELECT  gd2.id as GalaxyDistance2_id
    #     FROM GalaxyDistance2 gd2
    #     WHERE
    #         RA BETWEEN 0 AND 30 and
    #         _Dec BETWEEN -40 and -15 and
    #         z_dist IS NOT NULL and
    #         B IS NOT NULL and
    #         ST_WITHIN(Coord,
    #             (SELECT Poly FROM Detector d WHERE d.Name = "ZTF"))
    # '''
    # all_gal_result = query_db([all_gals_select])[0]
    # ids = ",".join([str(agr[0]) for agr in all_gal_result])
    # gal_query_2 = "SELECT RA, _Dec FROM GalaxyDistance2 WHERE id IN (%s)" % ids
    # all_gala = query_db([gal_query_2])[0]
    # for g in all_gala:
    #     _x, _y = m(g[0], g[1])
    #     m.plot(_x, _y, 'r.', alpha=0.2)
    #
    #
    #
    # ra_1 = 13.0
    # dec_1 = -28.0
    # theta_1 = 0.0
    #
    # rotated_square = list(map(tuple, map(lambda vertex: np.asarray(vertex @ rot(theta_1))[0], square)))
    # translated_square = list(map(lambda vertex: (vertex[0]/np.cos(np.radians(vertex[1] + dec_1)) + ra_1,
    #                                              vertex[1] + dec_1), rotated_square))
    #
    # geo_square = geometry.Polygon(translated_square)
    # ra_deg, dec_deg = zip(*[(coord_deg[0], coord_deg[1]) for coord_deg in geo_square.exterior.coords])
    # x2, y2 = m(ra_deg, dec_deg)
    # lat_lons = np.vstack([x2, y2]).transpose()
    # ax.add_patch(Polygon(lat_lons, edgecolor='r', facecolor='None', linestyle='-'))
    #
    #
    # parallels = np.arange(-90., 90., 10.)
    # dec_ticks = m.drawparallels(parallels, labels=[0, 1, 0, 0])
    # meridians = np.arange(0., 360., 7.5)
    # ra_ticks = m.drawmeridians(meridians, labels=[0, 0, 0, 1])
    #
    # ax.invert_xaxis()
    #
    # fig.savefig("test_detector_onsky.png", bbox_inches='tight')
    # plt.close('all')
    #
    #
    #
    #
    #
    #
    # print(mp_str)
    #
    # print("\n")
    #
    # swope_old_detect = Tile(90.0, 50.0, 0.5, 0.5, 1024)
    # print(swope_old_detect.query_polygon_string)
    #
    #
    #
    #
    # # detector_insert = "INSERT INTO Detector (Name, Deg_width, Deg_height, Deg_radius, Area, Poly) VALUES (%s, %s, %s, %s, %s, ST_GEOMFROMTEXT(%s, 4326));"
    # # detector_data = [
    # #     ("ZTF", None, None, None, None, mp_str)
    # # ]
    # #
    # # print("\nInserting %s detectors..." % len(detector_data))
    # # if insert_records(detector_insert, detector_data):
    # #     print("Success!")
    # # else:
    # #     raise("Error with INSERT! Exiting...")
    # # print("...Done")
    #
    # # t = '''MultiPolygon(
    # #         (
    # #             (0 0,0 3,3 3,3 0,0 0),
    # #             (1 1,1 2,2 2,2 1,1 1)
    # #         )
    # #     )'''
    #
    # xyz_polygons = []
    # for i, geom in enumerate(ztf_geometries):
    #
    #     _xyz_vertices = []
    #     for coord_deg in geom.exterior.coords[:-1]:
    #         ra_deg = coord_deg[0]
    #         dec_deg = coord_deg[1]
    #
    #         theta = 0.5 * np.pi - np.deg2rad(dec_deg)
    #         phi = np.deg2rad(ra_deg)
    #
    #         _xyz_vertices.append(hp.ang2vec(theta, phi))
    #
    #     xyz_polygons.append(np.asarray(_xyz_vertices))
    #
    # xyz_polygons = np.asarray(xyz_polygons)
    #
    # pix = []
    # for p in xyz_polygons:
    #     # print(p)
    #     internal_pix = hp.query_polygon(1024, p, inclusive=False)
    #     # print(internal_pix)
    #     for i in internal_pix:
    #         pix.append(Pixel_Element(i, 1024, 0.0))
    #
    #
    #
    #
    # # Test the ability to resolve the multipolygon into healpix pixels
    # fig = plt.figure(figsize=(10, 10), dpi=600)
    # ax = fig.add_subplot(111)
    #
    # lat_0 = -20.0
    # lon_0 = 15.0
    #
    # m = Basemap(projection='stere',
    #             lon_0=lon_0,
    #             lat_0=lat_0,
    #             llcrnrlat=-35.0,
    #             urcrnrlat=-19.5,
    #             llcrnrlon=8.0,
    #             urcrnrlon=24.5)
    #
    # ra_0 = 20.0
    # dec_0 = -25.0
    # theta_0 = 45.0
    #
    # transformed_ztf_3 = project_polygon(ztf_multi_poly, ra_0, dec_0, theta_0)
    # for geom in transformed_ztf_2:
    #     ra_deg, dec_deg = zip(*[(coord_deg[0], coord_deg[1]) for coord_deg in geo_poly.exterior.coords])
    #
    #     x2, y2 = m(ra_deg, dec_deg)
    #     lat_lons = np.vstack([x2, y2]).transpose()
    #     ax.add_patch(Polygon(lat_lons, edgecolor='k', facecolor='None', linestyle='-'))
    #
    # for geo_poly in ztf_geometries:
    #     ra_deg, dec_deg = zip(*[(coord_deg[0], coord_deg[1]) for coord_deg in geo_poly.exterior.coords])
    #
    #     x2, y2 = m(ra_deg, dec_deg)
    #     lat_lons = np.vstack([x2, y2]).transpose()
    #     ax.add_patch(Polygon(lat_lons, edgecolor='k', facecolor='None', linestyle='-'))
    #
    # for px in pix:
    #     px.plot(m, ax, edgecolor="green", facecolor="None", linestyle="-", linewidth=0.5)
    #
    # parallels = np.arange(-90., 90., 10.)
    # dec_ticks = m.drawparallels(parallels, labels=[0, 1, 0, 0])
    # meridians = np.arange(0., 360., 7.5)
    # ra_ticks = m.drawmeridians(meridians, labels=[0, 0, 0, 1])
    #
    # ax.invert_xaxis()
    #
    # fig.savefig("test_detector_onsky_pixels.png", bbox_inches='tight')
    # plt.close('all')

