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

from astropy import cosmology
from astropy.cosmology import WMAP5, WMAP7, LambdaCDM
from astropy.coordinates import Distance
import csv

from matplotlib import colors
import statistics
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Plotter import *
from Database_Helpers import *

import time

LOCAL_PORT = '3306'

class GTT:
	def __init__(self):

		swope_deg_width = 4096*0.435/3600. # pix * plate scale / arcsec/pix => degrees
		swope_deg_height = 4112*0.435/3600.

		andicam_deg_width = 1024*0.371/3600.
		andicam_deg_height = 1024*0.371/3600.

		thacher_deg_width = 2048*0.609/3600.
		thacher_deg_height = 2048*0.609/3600.

		nickel_deg_width = 2048*0.368/3600.
		nickel_deg_height = 2048*0.368/3600.

		self.telescope_mapping = {
		"S":Detector("SWOPE", swope_deg_width, swope_deg_height),
		"A":Detector("ANDICAM", andicam_deg_width, andicam_deg_height),
		"T":Detector("THACHER", thacher_deg_width, thacher_deg_height),
		"N":Detector("NICKEL", nickel_deg_width, nickel_deg_height)
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

		parser.add_option('--healpix_file', default="", type="str",
						help='healpix filename.')

		parser.add_option('--working_dir', default='./', type="str",help='Working directory for where to look for files and where to put output.')

		parser.add_option('--gw_id', default="", type="str",
						help='LIGO superevent name, e.g. `S190425z` ')

		parser.add_option('--event_number', default=1, type="int",
							  help='event number (default=%default)')

		parser.add_option('--schedule_designation', default='AA', type="str",
							  help='schedule designation (default=%default)')

		return(parser)

	def main(self):

		hpx_path = "%s/%s" % (self.options.working_dir, self.options.healpix_file)

		# If you specify a telescope that's not in the default list, you must provide the rest of the information
		detector = None
		is_error = False

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

			if is_error:
				print("Exiting...")
				return 1

			detector = Detector(self.options.telescope_name, self.options.detector_width_deg, self.options.detector_height_deg)
		else:
			detector = self.telescope_mapping[self.options.telescope_abbreviation]


		print("Telescope: `%s -- %s`, width: %s [deg]; height %s [deg]" % (self.options.telescope_abbreviation,
			detector.name, detector.deg_width, detector.deg_height))
		fov_area = (detector.deg_width * detector.deg_height)
		print("\n\n%s FOV area: %s" % (detector.name, fov_area))


		print("Loading base cartography...")
		base_cartography = None
		with open('%s/%s_base_cartography.pkl' % (self.options.working_dir, self.options.gw_id), 'rb') as handle:
			base_cartography = pickle.load(handle)

		print("Loading sql pixel map...")
		sql_pixel_map = None
		with open('%s/%s_sql_pixel_map.pkl' % (self.options.working_dir, self.options.gw_id), 'rb') as handle:
			sql_pixel_map = pickle.load(handle)

		print("Loading sql cartography...")
		sql_tile_cartography = None
		with open('%s/%s_sql_cartography.pkl' % (self.options.working_dir, self.options.gw_id), 'rb') as handle:
			sql_tile_cartography = pickle.load(handle)

		print("Loading galaxy query...")
		query = None
		with open('%s/%s_query.pkl' % (self.options.working_dir, self.options.gw_id), 'rb') as handle:
			query = pickle.load(handle)

		print("Loading contained galaxies...")
		contained_galaxies = None
		with open('%s/%s_contained_galaxies.pkl' % (self.options.working_dir, self.options.gw_id), 'rb') as handle:
			contained_galaxies = pickle.load(handle)

		print("Loading redistributed cartography...")
		redistributed_cartography = None
		with open('%s/%s_redstributed_cartography.pkl' % (self.options.working_dir, self.options.gw_id), 'rb') as handle:
			redistributed_cartography = pickle.load(handle)



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

		# def GetSexigesimalString(c):
		# 	ra = c.ra.hms
		# 	dec = c.dec.dms

		# 	ra_string = "%02d:%02d:%05.2f" % (ra[0],ra[1],ra[2])
		# 	if dec[0] >= 0:
		# 		dec_string = "+%02d:%02d:%05.2f" % (dec[0],np.abs(dec[1]),np.abs(dec[2]))
		# 	else:
		# 		dec_string = "%03d:%02d:%05.2f" % (dec[0],np.abs(dec[1]),np.abs(dec[2]))

		# 	# Python has a -0.0 object. If the deg is this (because object lies < 60 min south), the string formatter will drop the negative sign
		# 	if c.dec < 0.0 and dec[0] == 0.0:
		# 		dec_string = "-00:%02d:%05.2f" % (np.abs(dec[1]),np.abs(dec[2]))
		# 	return (ra_string, dec_string)

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


		plot_probability_map("%s/%s_SQL_Tiles" % (self.options.working_dir, self.options.gw_id),
					 pixels=sql_pixel_map.pixels_90,
					 tiles=sql_tile_cartography.tiles, 
					 colormap=plt.cm.viridis)

		plot_probability_map("%s/%s_%s_Redistributed_90th_Tiles" % (self.options.working_dir, self.options.gw_id, detector.name),
					 pixels=redistributed_cartography.unpacked_healpix.pixels_90,
					 tiles=redistributed_cartography.tiles)


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

	useagestring="""python Generate_Tiles.py [options]

Default telescope example:
python Generate_Tiles.py --gw_id S190425z --working_dir GW190425_Swope --healpix_file LALInference.fits.gz,0 --event_number 5 --telescope_abbreviation S --filter r --exp_time 180 --schedule_designation AA

Custom telescope example:
python Generate_Tiles.py --gw_id S190425z --working_dir GW190425_Swope --healpix_file LALInference.fits.gz,0 --event_number 5 --telescope_abbreviation SO --telescope_name SOAR --detector_width_deg 0.12 --detector_height_deg 0.12 --filter r --exp_time 180 --schedule_designation AA
"""
	
	start = time.time()


	gtt = GTT()
	parser = gtt.add_options(usage=useagestring)
	options,  args = parser.parse_args()
	gtt.options = options
	
	gtt.main()

	end = time.time()
	duration = (end - start)
	print("Execution time: %s" % duration)


