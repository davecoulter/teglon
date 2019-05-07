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


class GTT:
	def __init__(self):
		self.telescope_mapping = {
		"S":"SWOPE",
		"A":"ANDICAM",
		"T":"THACHER",
		"N":"NICKEL"}

	def add_options(self, parser=None, usage=None, config=None):
		import optparse
		if parser == None:
			parser = optparse.OptionParser(usage=usage, conflict_handler="resolve")

		parser.add_option('--plate_scale', default=0.435, type="float",
							  help='plate scale (default=%default)')

		parser.add_option('--filter', default='i', type="string",
							  help='Filter choice (default=%default)')

		parser.add_option('--exp_time', default='60.0', type="float",
							  help='Exposure time (default=%default)')

		parser.add_option('--detector_width_npix', default=4096, type="float",
							  help='detector width, in pixels (default=%default)')

		parser.add_option('--detector_height_npix', default=4112, type="float",
							  help='detector height, in pixels (default=%default)')

		parser.add_option('--healpix_file', default="", type="str",
						help='healpix filename.')

		parser.add_option('--working_dir', default='./', type="str",help='Working directory for where to look for files and where to put output.')

		parser.add_option('--gw_id', default="", type="str",
						help='LIGO superevent name, e.g. `S190425z` ')

		parser.add_option('--telescope_abbreviation', default="S", type="str",
							  help='one-letter abbreviation for telescope, S - Swope, A - Andicam, T - Thacher, N - Nickel (default=%default)')

		parser.add_option('--event_number', default=1, type="int",
							  help='event number (default=%default)')

		parser.add_option('--schedule_designation', default='AA', type="str",
							  help='schedule designation (default=%default)')

		return(parser)

	def main(self):

		# hpx_path = "%s/%s" % (self.options.working_dir, self.options.healpix_file)
		detector = Detector(self.telescope_mapping[self.options.telescope_abbreviation],
			float(self.options.detector_width_npix), 
			float(self.options.detector_height_npix),
			float(self.options.plate_scale))

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


		plot_probability_map("%s/%s_SQL_Tiles" % (self.options.working_dir, self.options.gw_id),
					 pixels=sql_pixel_map.pixels_90,
					 tiles=sql_tile_cartography.tiles, 
					 colormap=plt.cm.viridis)

		plot_probability_map("%s/%s_%s_Redistributed_90th_Tiles" % (self.options.working_dir, self.options.gw_id, detector.name),
					 pixels=redistributed_cartography.unpacked_healpix.pixels_90,
					 tiles=redistributed_cartography.tiles)


if __name__ == "__main__":
	
	import os
	import optparse

	useagestring="""python Generate_Tiles.py [options]

example:
python Generate_Tiles.py --working_dir GW170817/ --healpix_file LALInference_v2.fits.gz,0 --event_name GW170817 --event_number 1 --telescope_abbreviation S"""
	
	start = time.time()


	gtt = GTT()
	parser = gtt.add_options(usage=useagestring)
	options,  args = parser.parse_args()
	gtt.options = options
	
	gtt.main()

	end = time.time()
	duration = (end - start)
	print("Execution time: %s" % duration)


