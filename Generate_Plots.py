import matplotlib

matplotlib.use("Agg")

import astropy as aa
import numpy as np
from astropy import units as u
import astropy.coordinates as coord
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from astropy.io import fits
from matplotlib.patches import Polygon
from astroquery.irsa_dust import IrsaDust
import glob
import random
from scipy.special import erf
from scipy.optimize import minimize, minimize_scalar
import scipy.stats as st
from matplotlib.pyplot import cm
import copy
import random
import pprint
from pprint import pformat
from ligo.skymap import distance
from functools import reduce
import matplotlib as mpl
from scipy.integrate import simps
import os
import pickle
from scipy.interpolate import interp2d
from shapely import geometry
from itertools import groupby
import re

from astropy.coordinates.angles import Angle

import urllib.request
from bs4 import BeautifulSoup
from oauth2client import file, client, tools
from apiclient.discovery import build
from httplib2 import Http

from HEALPix_Helpers import *
from Tile import *
from Pixel_Element import *
from SQL_Polygon import *

import xml.etree.ElementTree as ET

import math
import csv

import ephem
from dateutil.parser import parse
from datetime import tzinfo, timedelta, datetime
import pytz as pytz

from matplotlib.patches import CirclePolygon
from shapely.ops import transform as shapely_transform
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.geometry import JOIN_STYLE

from astropy import cosmology
from astropy.cosmology import WMAP5, WMAP7, LambdaCDM
from astropy.coordinates import Distance
import csv

from matplotlib import colors
import statistics
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Plotter import *
from Database_Helpers import *

from dustmaps.config import config
from dustmaps.sfd import SFDQuery

import time

LOCAL_PORT = '3306'


class GTT:
    def __init__(self):

        swope_deg_width = 4096 * 0.435 / 3600.  # pix * plate scale / arcsec/pix => degrees
        swope_deg_height = 4112 * 0.435 / 3600.

        andicam_deg_width = 1024 * 0.371 / 3600.
        andicam_deg_height = 1024 * 0.371 / 3600.

        thacher_deg_width = 2048 * 0.609 / 3600.
        thacher_deg_height = 2048 * 0.609 / 3600.

        nickel_deg_width = 2048 * 0.368 / 3600.
        nickel_deg_height = 2048 * 0.368 / 3600.

        mosfire_deg_width = 6.14 / 60.0
        mosfire_deg_height = 6.14 / 60.0

        self.telescope_mapping = {
            "S": Detector("SWOPE", swope_deg_width, swope_deg_height),
            "A": Detector("ANDICAM", andicam_deg_width, andicam_deg_height),
            "T": Detector("THACHER", thacher_deg_width, thacher_deg_height),
            "N": Detector("NICKEL", nickel_deg_width, nickel_deg_height),
            "M": Detector("MOSFIRE", mosfire_deg_width, mosfire_deg_height)
        }

    def add_options(self, parser=None, usage=None, config=None):
        import optparse
        if parser == None:
            parser = optparse.OptionParser(usage=usage, conflict_handler="resolve")

        parser.add_option('--telescope_abbreviation', default="", type="str",
                          help='Abbreviation for telescope. Built-in telescopes: S - Swope, A - Andicam, T - Thacher, N - Nickel. \nIf using a non-built in, use another character/char combination and provide detector width and height.')

        parser.add_option('--telescope_name', default="", type="str",
                          help='Name of telescope. Ignore for built-in telescopes: S - Swope, A - Andicam, T - Thacher, N - Nickel. \nIf using a non-built in, specify name.')

        parser.add_option('--detector_width_deg', default=0, type="float",
                          help='Detector width in degrees. For circular apertures, use radius for width and height.')

        parser.add_option('--detector_height_deg', default=0, type="float",
                          help='Detector height in degrees. For circular apertures, use radius for width and height.')

        parser.add_option('--filter', default='r', type="string",
                          help='Filter choice (default=%default)')

        parser.add_option('--exp_time', default='60.0', type="float",
                          help='Exposure time (default=%default)')

        parser.add_option('--healpix_dir', default='./', type="str",
                          help='Directory for where to look for the healpix file.')

        parser.add_option('--healpix_file', default="", type="str",
                          help='healpix filename.')

        parser.add_option('--working_dir', default='./', type="str",
                          help='Working directory for where to look for files and where to put output.')

        parser.add_option('--gw_id', default="", type="str",
                          help='LIGO superevent name, e.g. `S190425z` ')

        parser.add_option('--event_number', default=1, type="int",
                          help='event number (default=%default)')

        parser.add_option('--schedule_designation', default='AA', type="str",
                          help='schedule designation (default=%default)')

        parser.add_option('--percentile', default='-1', type="float",
                          help='Percentile to tile to -- use this flag to set the percentile to < 0.90')

        parser.add_option('--skip_completeness', action="store_true", help='If True, purely tiles sky map',
                          default=False)

        return (parser)

    def main(self):

        hpx_path = "%s/%s" % (self.options.healpix_dir, self.options.healpix_file)

        # If you specify a telescope that's not in the default list, you must provide the rest of the information
        detector = None
        is_error = False
        is_custom_percentile = False
        custom_percentile = None

        # Check the inputs for errors...
        if self.options.telescope_abbreviation not in self.telescope_mapping.keys():
            print("Running for custom telescope. Checking required parameters...")

            if self.options.telescope_abbreviation == "":
                is_error = True
                print("For custom telescope, `telescope_abbreviation` is required!")

            if self.options.telescope_name == "":
                is_error = True
                print("For custom telescope, `telescope_name` is required!")

            if self.options.detector_width_deg <= 0.0:
                is_error = True
                print("For custom telescope, `detector_width_deg` is required, and must be > 0!")

            if self.options.detector_height_deg <= 0.0:
                is_error = True
                print("For custom telescope, `detector_height_deg` is required, and must be > 0!")

            if not is_error:
                detector = Detector(self.options.telescope_name, self.options.detector_width_deg,
                                    self.options.detector_height_deg)
        else:
            detector = self.telescope_mapping[self.options.telescope_abbreviation]

        if self.options.percentile > 0.9:
            is_error = True
            print("User-defined percentile must be <= 0.90")
        elif self.options.percentile > 0.0:
            is_custom_percentile = True
            custom_percentile = self.options.percentile

        if is_error:
            print("Exiting...")
            return 1

        print("\n\nTelescope: `%s -- %s`, width: %s [deg]; height %s [deg]" % (self.options.telescope_abbreviation,
                                                                               detector.name, detector.deg_width,
                                                                               detector.deg_height))
        fov_area = (detector.deg_width * detector.deg_height)
        print("%s FOV area: %s" % (detector.name, fov_area))

        print("Loading base cartography...")
        base_cartography = None
        with open('%s/%s_base_cartography.pkl' % (self.options.working_dir, self.options.gw_id), 'rb') as handle:
            base_cartography = pickle.load(handle)

        # print("Loading sql pixel map...")
        # sql_pixel_map = None
        # with open('%s/%s_sql_pixel_map.pkl' % (self.options.working_dir, self.options.gw_id), 'rb') as handle:
        # 	sql_pixel_map = pickle.load(handle)

        # print("Loading sql cartography...")
        # sql_tile_cartography = None
        # with open('%s/%s_sql_cartography.pkl' % (self.options.working_dir, self.options.gw_id), 'rb') as handle:
        # 	sql_tile_cartography = pickle.load(handle)

        # print("Loading galaxy query...")
        # query = None
        # with open('%s/%s_query.pkl' % (self.options.working_dir, self.options.gw_id), 'rb') as handle:
        # 	query = pickle.load(handle)

        if not self.options.skip_completeness:
            print("Loading sql multipolygon...")
            sql_poly = None
            with open('%s/%s_sql_poly.pkl' % (self.options.working_dir, self.options.gw_id), 'rb') as handle:
                sql_poly = pickle.load(handle)

            # print("Loading contained galaxies...")
            # contained_galaxies = None
            # with open('%s/%s_contained_galaxies.pkl' % (self.options.working_dir, self.options.gw_id), 'rb') as handle:
            # 	contained_galaxies = pickle.load(handle)

            # for g in (sorted(contained_galaxies, key=lambda x: x.relative_prob, reverse=True))[:99]:
            # 	print(g.relative_prob)

            print("Loading redistributed cartography...")
            redistributed_cartography = None
            with open('%s/%s_redstributed_cartography.pkl' % (self.options.working_dir, self.options.gw_id),
                      'rb') as handle:
                redistributed_cartography = pickle.load(handle)

        # print("Loading observed tiles file...")
        # observed_tiles = {
        # 	"S190728q":{"Thacher":{"ut190728":[]},
        # 				"Swope":{"ut190728":[]}
        # 			   }
        # }

        # running_prob = 0
        # unique_pixels = []
        # # Thacher
        # thacher_width = 2048*0.609/3600. #arcseconds -> deg
        # thacher_height = 2048*0.609/3600.
        # with open('%s/ut190727_28_Thacher_observed.txt' % self.options.working_dir,'r') as csvfile:
        # 	csvreader = csv.reader(csvfile, delimiter=',',skipinitialspace=True)

        # 	for row in csvreader:
        # 		name = row[0]
        # 		t = Tile(coord.SkyCoord(row[1], row[2], unit=(u.hour, u.deg)), thacher_width, thacher_height, redistributed_cartography.unpacked_healpix.nside)
        # 		t.field_name = name

        # 		t_prob = t.enclosed_pixel_indices
        # 		for ti in t_prob:
        # 			if ti not in unique_pixels:
        # 				unique_pixels.append(ti)

        # 		t.net_prob = np.sum(redistributed_cartography.unpacked_healpix.prob[t_prob])
        # 		running_prob += t.net_prob
        # 		print("%s - net prob: %s" % (name, t.net_prob))
        # 		observed_tiles["S190728q"]["Thacher"]["ut190728"].append(t)

        # print("\n")
        # swope_width = 4096*0.435/3600. #arcseconds -> deg
        # swope_height = 4112*0.435/3600.
        # with open('%s/ut190727_28_Swope_observed.txt' % self.options.working_dir,'r') as csvfile:
        # 	csvreader = csv.reader(csvfile, delimiter=',',skipinitialspace=True)

        # 	for row in csvreader:
        # 		name = row[0]
        # 		t = Tile(coord.SkyCoord(row[1], row[2], unit=(u.hour, u.deg)), swope_width, swope_height, redistributed_cartography.unpacked_healpix.nside)
        # 		t.field_name = name

        # 		t_prob = t.enclosed_pixel_indices

        # 		for ti in t_prob:
        # 			if ti not in unique_pixels:
        # 				unique_pixels.append(ti)

        # 		t.net_prob = np.sum(redistributed_cartography.unpacked_healpix.prob[t_prob])
        # 		running_prob += t.net_prob
        # 		print("%s - net prob: %s" % (name, t.net_prob))
        # 		observed_tiles["S190728q"]["Swope"]["ut190728"].append(t)

        # tile_set = [
        # 	("Thacher_0728",observed_tiles["S190728q"]["Thacher"]["ut190728"],('royalblue','None')),
        # 	("Swope_0728",observed_tiles["S190728q"]["Swope"]["ut190728"],('green','None')),
        # ]
        # print("Total non-corrected prob captured by observed Thacher+Swope tiles: %s" % running_prob)
        # print("Total corrected prob captured by observed Thacher+Swope tiles: %s" % np.sum(redistributed_cartography.unpacked_healpix.prob[np.asarray(unique_pixels)]))

        print("\n\nBuilding MWE...")
        config["data_dir"] = "./DataIngestion"

        print("Initializing MWE n128 pix...")
        f99_r = 2.285
        nside128 = 128
        nside128_npix = hp.nside2npix(nside128)
        nside128_pixels = []
        nside128_RA = []
        nside128_DEC = []

        mw_pixels = []
        for i in range(nside128_npix):
            pe = Pixel_Element(i, nside128, 0.0)
            nside128_pixels.append(pe)

            theta, phi = hp.pix2ang(pe.nside, pe.index)
            dec = np.degrees(0.5 * np.pi - theta)
            ra = np.degrees(phi)

            nside128_RA.append(ra)
            nside128_DEC.append(dec)

        c1 = coord.SkyCoord(nside128_RA, nside128_DEC, unit=(u.deg, u.deg))
        sfd = SFDQuery()
        ebv1 = sfd(c1)
        for i, e in enumerate(ebv1):
            if e * f99_r >= 0.5:
                mw_pixels.append(nside128_pixels[i])

        tiles_to_plot = None
        pixels_to_plot = None
        sql_poly_to_plot = None
        if not self.options.skip_completeness:
            tiles_to_plot = redistributed_cartography.tiles
            pixels_to_plot = redistributed_cartography.unpacked_healpix.pixels_90
            sql_poly_to_plot = sql_poly
        else:
            tiles_to_plot = base_cartography.tiles
            pixels_to_plot = base_cartography.unpacked_healpix.pixels_90

        ra_tiles = []
        dec_tiles = []
        for i, t in enumerate(tiles_to_plot):
            field_name = "%s%s%sE%s" % (self.options.telescope_abbreviation,
                                        str(self.options.event_number).zfill(3),
                                        self.options.schedule_designation,
                                        str(i + 1).zfill(5))

            t.field_name = field_name
            # ra_tiles.append(t.coord.ra.degree)
            # dec_tiles.append(t.coord.dec.degree)
            ra_tiles.append(t.ra_deg)
            dec_tiles.append(t.dec_deg)

        c = coord.SkyCoord(ra_tiles, dec_tiles, unit=(u.deg, u.deg))
        sfd = SFDQuery()
        ebv = sfd(c)

        new_tiles = []
        for i, e in enumerate(ebv):
            t = tiles_to_plot[i]
            # if e*f99_r < 0.5 and t.coord.dec.degree > -30.0:
            if e * f99_r < 0.5 and t.dec_deg < 30.0:
                new_tiles.append(t)
        # if t.coord.dec.degree > -30.0:
        # 	t.mwe = e
        # 	new_tiles.append(t)

        print(len(new_tiles))
        prob = 0.0
        for t in new_tiles:
            prob += t.net_prob

        print("Enclosed prob = %s" % prob)

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

        with open('%s/%s_%s_MWE_below_30_AA.txt' % (self.options.working_dir, self.options.gw_id, detector.name),
                  'w') as csvfile:

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
            # cols.append('A_R')
            csvwriter.writerow(cols)

            for i, st in enumerate(new_tiles):
                # coord_str = GetSexigesimalString(st.coord)
                c = coord.SkyCoord(st.ra_deg, st.dec_deg, unit=(u.deg, u.deg))
                coord_str = GetSexigesimalString(c)

                cols = []

                cols.append(st.field_name)
                cols.append(coord_str[0])
                cols.append(coord_str[1])
                cols.append(detector.name)
                cols.append(self.options.filter)
                cols.append(self.options.exp_time)
                cols.append(st.net_prob)
                cols.append('False')
                # cols.append(st.mwe)
                csvwriter.writerow(cols)

            print("Done")

        # return 0;
        # raise("Stop!")

        # swope_width = 4096*0.435/3600. #arcseconds -> deg
        # swope_height = 4112*0.435/3600.

        # thacher_width = 2048*0.609/3600. #arcseconds -> deg
        # thacher_height = 2048*0.609/3600.

        # nickel_width = 2048*0.368/3600. #arcseconds -> deg
        # nickel_height = 2048*0.368/3600.

        # nickel_width = 2048*0.368/3600. #arcseconds -> deg
        # nickel_height = 2048*0.368/3600.

        # andicam_ccd_width = 1024*0.371/3600. #arcseconds -> deg
        # andicam_ccd_height = 1024*0.371/3600.

        # swift_uvot_width = 2048*0.502/3600. #arcseconds -> deg
        # swift_uvot_height = 2048*0.502/3600.

        # saguaro_width = 2.26 #deg
        # saguaro_height = 2.26

        # observed_tiles = {
        # 	"S190425z":{"Swope":{"ut190425":[], "ut190428":[], "ut190429":[]},
        # 				"Thacher":{"ut190425":[]},
        # 				"ANDICAM":{"ut190425":[], "ut190428":[]},
        # 				"Nickel":{"ut190425":[], "ut190429":[]},
        # 				"Swift":{"ut190425":[]},
        # 				"SAGUARO":{"ut190425":[]}
        # 			   }
        # }

        # # Just do S190425z for now...
        # # SWOPE
        # with open('../O3_Alerts/GW190408/GW190425_2/S190425z_ut190425_Swope_Tiles_Observed.csv','r') as csvfile:

        # 	csvreader = csv.reader(csvfile, delimiter=' ',skipinitialspace=True)
        # 	next(csvreader)

        # 	for row in csvreader:
        # 		t = Tile(coord.SkyCoord(row[1], row[2], unit=(u.hour, u.deg)), swope_width, swope_height, redistributed_cartography.unpacked_healpix.nside)
        # 		(observed_tiles["S190425z"]["Swope"]["ut190425"]).append(t)

        # with open('../O3_Alerts/GW190408/GW190425_2/S190425z_ut190428_Swope_Tiles_Observed.csv','r') as csvfile:

        # 	csvreader = csv.reader(csvfile, delimiter=' ',skipinitialspace=True)
        # 	next(csvreader)

        # 	for row in csvreader:
        # 		t = Tile(coord.SkyCoord(row[1], row[2], unit=(u.hour, u.deg)), swope_width, swope_height, redistributed_cartography.unpacked_healpix.nside)
        # 		(observed_tiles["S190425z"]["Swope"]["ut190428"]).append(t)

        # with open('../O3_Alerts/GW190408/GW190425_2/S190425z_ut190429_Swope_Tiles_Observed.csv','r') as csvfile:

        # 	csvreader = csv.reader(csvfile, delimiter=' ',skipinitialspace=True)
        # 	next(csvreader)

        # 	for row in csvreader:
        # 		t = Tile(coord.SkyCoord(row[1], row[2], unit=(u.hour, u.deg)), swope_width, swope_height, redistributed_cartography.unpacked_healpix.nside)
        # 		(observed_tiles["S190425z"]["Swope"]["ut190429"]).append(t)

        # # THACHER
        # with open('../O3_Alerts/GW190408/GW190425_2/S190425z_ut190425_Thacher_Tiles_Observed.csv','r') as csvfile:

        # 	csvreader = csv.reader(csvfile, delimiter=' ',skipinitialspace=True)
        # 	next(csvreader)

        # 	for row in csvreader:
        # 		t = Tile(coord.SkyCoord(row[1], row[2], unit=(u.hour, u.deg)), thacher_width, thacher_height, redistributed_cartography.unpacked_healpix.nside)
        # 		(observed_tiles["S190425z"]["Thacher"]["ut190425"]).append(t)

        # # ANDICAM
        # with open('../O3_Alerts/GW190408/GW190425_2/S190425z_ut190425_ANDICAM_Tiles_Observed.csv','r') as csvfile:

        # 	csvreader = csv.reader(csvfile, delimiter=' ',skipinitialspace=True)
        # 	next(csvreader)

        # 	for row in csvreader:
        # 		t = Tile(coord.SkyCoord(row[1], row[2], unit=(u.hour, u.deg)), andicam_ccd_width, andicam_ccd_height, redistributed_cartography.unpacked_healpix.nside)
        # 		(observed_tiles["S190425z"]["ANDICAM"]["ut190425"]).append(t)

        # with open('../O3_Alerts/GW190408/GW190425_2/S190425z_ut190428_ANDICAM_Tiles_Observed.csv','r') as csvfile:

        # 	csvreader = csv.reader(csvfile, delimiter=' ',skipinitialspace=True)
        # 	next(csvreader)

        # 	for row in csvreader:
        # 		t = Tile(coord.SkyCoord(row[1], row[2], unit=(u.hour, u.deg)), andicam_ccd_width, andicam_ccd_height, redistributed_cartography.unpacked_healpix.nside)
        # 		(observed_tiles["S190425z"]["ANDICAM"]["ut190428"]).append(t)

        # # NICKEL
        # with open('../O3_Alerts/GW190408/GW190425_2/S190425z_ut190425_Nickel_Tiles_Observed.csv','r') as csvfile:

        # 	csvreader = csv.reader(csvfile, delimiter=' ',skipinitialspace=True)
        # 	next(csvreader)

        # 	for row in csvreader:
        # 		t = Tile(coord.SkyCoord(row[1], row[2], unit=(u.hour, u.deg)), nickel_width, nickel_height, redistributed_cartography.unpacked_healpix.nside)
        # 		(observed_tiles["S190425z"]["Nickel"]["ut190425"]).append(t)

        # with open('../O3_Alerts/GW190408/GW190425_2/S190425z_ut190429_Nickel_Tiles_Observed.csv','r') as csvfile:

        # 	csvreader = csv.reader(csvfile, delimiter=' ',skipinitialspace=True)
        # 	next(csvreader)

        # 	for row in csvreader:
        # 		t = Tile(coord.SkyCoord(row[1], row[2], unit=(u.hour, u.deg)), nickel_width, nickel_height, redistributed_cartography.unpacked_healpix.nside)
        # 		(observed_tiles["S190425z"]["Nickel"]["ut190429"]).append(t)

        # # Swift
        # with open('../O3_Alerts/GW190408/GW190425_2/S190425z_ut190425_Swift_Tiles_Observed.csv','r') as csvfile:

        # 	csvreader = csv.reader(csvfile, delimiter=' ',skipinitialspace=True)
        # 	next(csvreader)

        # 	for row in csvreader:
        # 		t = Tile(coord.SkyCoord(row[1], row[2], unit=(u.deg, u.deg)), swift_uvot_width, swift_uvot_height, redistributed_cartography.unpacked_healpix.nside)
        # 		(observed_tiles["S190425z"]["Swift"]["ut190425"]).append(t)

        # for k1,v1 in observed_tiles.items():
        # 	print("%s" % k1)

        # 	for k2,v2 in observed_tiles[k1].items():
        # 		print("\t%s" % k2)

        # 		for k3,v3 in observed_tiles[k1][k2].items():
        # 			print("\t\t%s - # of tiles: %s" % (k3, len(observed_tiles[k1][k2][k3])))

        # 	print("\n")

        # tile_set = [
        # 	("Swope_0425",observed_tiles["S190425z"]["Swope"]["ut190425"],('r','None')),
        # 	("Swope_0428",observed_tiles["S190425z"]["Swope"]["ut190428"],('r','None')),
        # 	("Swope_0429",observed_tiles["S190425z"]["Swope"]["ut190429"],('r','None')),

        # 	("Thacher_0425",observed_tiles["S190425z"]["Thacher"]["ut190425"],('royalblue','None')),

        # 	("ANDICAM_0425",observed_tiles["S190425z"]["ANDICAM"]["ut190425"],('forestgreen','None')),
        # 	("ANDICAM_0428",observed_tiles["S190425z"]["ANDICAM"]["ut190428"],('forestgreen','None')),

        # 	("Nickel_0425",observed_tiles["S190425z"]["Nickel"]["ut190425"],('mediumorchid','None')),
        # 	("Nickel_0429",observed_tiles["S190425z"]["Nickel"]["ut190429"],('mediumorchid','None')),

        # 	("Swift_0425",observed_tiles["S190425z"]["Swift"]["ut190425"],('k','None'))
        # ]

        # # In case you want to plot contours...
        # print("Computing contours for '%s'...\n" % base_cartography.unpacked_healpix.file_name)
        # base_cartography.unpacked_healpix.compute_contours()

        # print("Computing contours for '%s'...\n" % redistributed_cartography.unpacked_healpix.file_name)
        # redistributed_cartography.unpacked_healpix.compute_contours()

        # # Thacher
        # thacher_tiles = []
        # with open('%s/S190521g_AA_THACHER_Tiles.txt' % self.options.working_dir,'r') as csvfile:

        #     csvreader = csv.reader(csvfile, delimiter=',')
        #     next(csvreader)

        #     for row in csvreader:
        #         t = Tile(coord.SkyCoord(row[1], row[2], unit=(u.hour, u.deg)), detector.deg_width, detector.deg_height,
        #                  redistributed_cartography.unpacked_healpix.nside)
        #         t.field_name = row[0]
        #         t.net_prob = np.sum(redistributed_cartography.unpacked_healpix.prob[t.enclosed_pixel_indices])

        #       		print("Tile #: %s; Net prob: %0.4f" % (t.field_name, t.net_prob))

        #         thacher_tiles.append(t)

        # above_30 = []
        # for t in thacher_tiles:
        # 	if t.coord.dec.degree > -30.0:
        # 		above_30.append(t)

        # sorted_tiles = sorted(above_30, key=lambda t: t.net_prob, reverse=True)
        # print("Number of northern tiles: %s" % len(sorted_tiles))

        # top_200 = sorted_tiles[0:199]
        # top_200_prob = np.sum([t.net_prob for t in top_200])
        # print("Prob of northern, top 200: %s" % top_200_prob)

        # top_200_area = fov_area*200
        # print("Area of northern, top 200: %s" % top_200_area)

        # with open('%s/S190521g_AA_THACHER_Tiles_Northern_200.txt' % self.options.working_dir,'w') as csvfile:

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

        # 	for i, st in enumerate(top_200):

        # 		coord_str = GetSexigesimalString(st.coord)

        # 		cols = []

        # 		cols.append(st.field_name)
        # 		cols.append(coord_str[0])
        # 		cols.append(coord_str[1])
        # 		cols.append(detector.name)
        # 		cols.append(self.options.filter)
        # 		cols.append(self.options.exp_time)
        # 		print(st.net_prob)
        # 		cols.append(st.net_prob)
        # 		cols.append('False')
        # 		csvwriter.writerow(cols)

        # 	print("Done")

        # test = []
        # for t in base_cartography.tiles:
        # 	test.append(t.polygon)

        # # print("here")
        # # print(list(test[0].exterior.coords))
        # # print("\n\n")

        # test2 = unary_union(test)
        # # print(type(test2))

        # eps = 0.00001
        # test3 = test2.buffer(eps, 1, join_style=JOIN_STYLE.mitre).buffer(-eps, 1, join_style=JOIN_STYLE.mitre)

        # plot_probability_map("%s/%s_SQL_Pixels" % (self.options.working_dir, self.options.gw_id),
        # 			 pixels_filled=base_cartography.unpacked_healpix.pixels_90,
        # 			 # tiles=base_cartography.tiles,
        # 			 # pixels_empty=base_cartography.unpacked_healpix.pixels_90,
        # 			 colormap=plt.cm.viridis,
        # 			 linear_rings=test3)

        # pixels_90 = [Pixel_Element(i90, redistributed_cartography.unpacked_healpix.nside,
        # 							redistributed_cartography.unpacked_healpix.prob[i90])
        # 							for i90 in redistributed_cartography.unpacked_healpix.indices_of_90]

        plot_probability_map(
            "%s/%s_%s_Redistributed_90th_Tiles" % (self.options.working_dir, self.options.gw_id, detector.name),
            # pixels_filled=redistributed_cartography.unpacked_healpix.pixels_90,
            pixels_filled=pixels_to_plot,
            # galaxies=contained_galaxies,
            galaxies=mw_pixels,
            tiles=new_tiles,
            # tiles=redistributed_cartography.tiles,
            # tiles=base_cartography.tiles,
            # sql_poly=sql_poly,
            # sql_poly=sql_poly_to_plot,
            # linear_rings=sql_poly.polygon,
            # healpix_obj_for_contours=base_cartography.unpacked_healpix,
            # tile_set=tile_set
            )

    # plot_probability_map_2("%s/%s_%s_Redistributed_90th_Tiles" % (self.options.working_dir, self.options.gw_id, detector.name),
    # 				central_lon=330,
    # 				 central_lat=30,
    # 				 lower_left_corner_lat=15,
    # 				 lower_left_corner_lon=345,
    # 				 upper_right_corner_lat=45,
    # 				 upper_right_corner_lon=315,
    # 			 pixels=redistributed_cartography.unpacked_healpix.pixels_90,
    # 			 tiles=redistributed_cartography.tiles)


if __name__ == "__main__":
    import os
    import optparse

    useagestring = """python Generate_Plots.py [options]

Default telescope example:
python Generate_Tiles.py --gw_id S190425z --healpix_dir S190521g --healpix_file bayestar.fits.gz,0 --working_dir S190521g_Swope  --event_number 14 --telescope_abbreviation S --filter r --exp_time 180 --schedule_designation AA

Custom telescope example:
python Generate_Tiles.py --gw_id S190425z --healpix_dir S190521g --healpix_file bayestar.fits.gz,0 --working_dir S190521g_SOAR  --event_number 14 --telescope_abbreviation SO --telescope_name SOAR --detector_width_deg 0.12 --detector_height_deg 0.12 --filter r --exp_time 180 --schedule_designation AA
"""

    start = time.time()

    gtt = GTT()
    parser = gtt.add_options(usage=useagestring)
    options, args = parser.parse_args()
    gtt.options = options

    gtt.main()

    end = time.time()
    duration = (end - start)
    print("Execution time: %s" % duration)
