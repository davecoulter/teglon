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

		hpx_path = "%s/%s" % (self.options.working_dir, self.options.healpix_file)
		detector = Detector(self.telescope_mapping[self.options.telescope_abbreviation],
			float(self.options.detector_width_npix), 
			float(self.options.detector_height_npix),
			float(self.options.plate_scale))


		print("\n\nUnpacking '%s':%s..." % (self.options.gw_id, hpx_path))
		prob, distmu, distsigma, distnorm, header_gen = hp.read_map(hpx_path, field=(0,1,2,3), h=True)
		header = dict(header_gen)

		npix = len(prob)
		print("\nNumber of pix in '%s': %s" % (hpx_path,len(prob)))

		sky_area = 4 * 180**2 / np.pi
		area_per_px = sky_area / npix
		print("Sky Area per pix in '%s': %s [sq deg]" % (hpx_path, area_per_px))

		sky_area_radians = 4*np.pi
		steradian_per_pix = sky_area_radians/npix
		pixel_radius_radian = np.sqrt(steradian_per_pix/np.pi)
		print("Steradian per px for '%s': %s" % (hpx_path, steradian_per_pix))

		# nside = the lateral resolution of the HEALPix map
		nside = hp.npix2nside(npix)
		print("Resolution (nside) of '%s': %s\n" % (hpx_path, nside))

		print("Processing for %s..." % detector.name)
		num_px_per_field = (detector.deg_height*detector.deg_width)/area_per_px
		print("Pix per (%s) field for '%s': %s" % (detector.name, hpx_path, num_px_per_field))
		
		unpacked_healpix = Unpacked_Healpix(hpx_path,
											prob, 
											distmu, 
											distsigma, 
											distnorm, 
											header, 
											nside, 
											npix, 
											area_per_px, 
											linestyle="-")

		num_50 = len(unpacked_healpix.indices_of_50)
		num_90 = len(unpacked_healpix.indices_of_90)
		area_50 = num_50*area_per_px
		area_90 = num_90*area_per_px

		print("\nArea of 50th: %s" % area_50)
		print("Area of 90th: %s\n" % area_90)


		# Get Swope all-sky tile pattern
		print("Generating all sky coords for %s" % detector.name)

		# Get pixel element for highest prob pixel
		index_of_max_prob = np.argmax(unpacked_healpix.prob)
		max_pixel = Pixel_Element(index_of_max_prob, unpacked_healpix.nside, unpacked_healpix.prob[index_of_max_prob])

		# Center grid on highest prob pixel
		detector_all_sky_coords = Cartographer.generate_all_sky_coords(detector, 
			max_pixel.coord.ra.degree, 
			max_pixel.coord.dec.degree)

		base_cartography = Cartographer(self.options.gw_id, unpacked_healpix, detector, detector_all_sky_coords)


		# Save cartograpy
		with open('%s/%s_base_cartography.pkl' % (self.options.working_dir, self.options.gw_id), 'wb') as handle:
			pickle.dump(base_cartography, handle, protocol=pickle.HIGHEST_PROTOCOL)


		# Generate SQL query
		sql_pixel_size = Detector("sql_pixel",3600*5.0,3600*5.0,1.0) # 5 degrees in arcseconds
		sql_tile_size = Detector("sql_tile",3600*10.0,3600*10.0,1.0) # 5 degrees in arcseconds

		# Generate all sky coords for the large SQL tiles
		print("Generating all sky coords for %s" % sql_tile_size.name)
		sql_tile_all_sky_coords = Cartographer.generate_all_sky_coords(sql_tile_size, 
																	   max_pixel.coord.ra.degree, 
																	   max_pixel.coord.dec.degree)

		# Generate a more granular sampling of the healpix with the sql_pixel_size
		sql_tile_cartography = Cartographer(self.options.gw_id, 
													 unpacked_healpix, 
													 sql_tile_size, 
													 sql_tile_all_sky_coords, 
													 generate_tiles=False)

		sql_pixel_map = Cartographer.downsample_map(sql_pixel_size, unpacked_healpix)
		sql_prob, sql_tiles = Cartographer.generate_tiles(unpacked_healpix, 
														  sql_pixel_map.pixels_90, 
														  sql_tile_size, 
														  sql_tile_all_sky_coords)

		sql_tile_cartography.tiles = sql_tiles
		sql_tile_cartography.cumlative_prob_in_tiles = sql_prob

		# Save cartograpy
		with open('%s/%s_sql_pixel_map.pkl' % (self.options.working_dir, self.options.gw_id), 'wb') as handle:
			pickle.dump(sql_pixel_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
		with open('%s/%s_sql_cartography.pkl' % (self.options.working_dir, self.options.gw_id), 'wb') as handle:
			pickle.dump(sql_tile_cartography, handle, protocol=pickle.HIGHEST_PROTOCOL)


		# # if debug
		# plot_probability_map("%s/%s_SQL_Tiles" % (self.options.working_dir, self.options.gw_id),
		# 			 pixels=sql_pixel_map.pixels_90,
		# 			 tiles=sql_tile_cartography.tiles, 
		# 			 colormap=plt.cm.viridis)


		ra_between = []
		dec_between = []

		for t in sql_tile_cartography.tiles:

			ra_min = np.min([c.ra.degree for c in t.corner_coords])
			ra_min = ra_min + 360.0 if ra_min < 0 else ra_min

			ra_max = np.max([c.ra.degree for c in t.corner_coords])
			ra_max = ra_max + 360.0 if ra_max < 0 else ra_max

			dec_min = np.min([c.dec.degree for c in t.corner_coords])
			dec_max = np.max([c.dec.degree for c in t.corner_coords])

			ra_between.append((ra_min, ra_max))
			dec_between.append((dec_min, dec_max))

		print("Tiles in %s: %s" % (sql_tile_cartography.gwid, 
								   len(sql_tile_cartography.tiles)))

		query = "SELECT * from Galaxy WHERE\n("

		for i,ra in enumerate(ra_between):
			query += "(RA BETWEEN %0.5f AND %0.5f AND _Dec BETWEEN %0.5f AND %0.5f)" % (ra_between[i][0],
																				   ra_between[i][1],
																				   dec_between[i][0],
																				   dec_between[i][1])
			if (i+1) < len(ra_between):
				query += " OR\n"
			else:
				query += ")\n"

		query += "AND dist IS NOT NULL AND B IS NOT NULL;"   
		print(query)
		print("\n*****************************\n")

		# Save cartograpy
		with open('%s/%s_query.pkl' % (self.options.working_dir, self.options.gw_id), 'wb') as handle:
			pickle.dump(query, handle, protocol=pickle.HIGHEST_PROTOCOL)



		result = QueryDB(query, port=LOCAL_PORT)

		contained_galaxies = []
		for row in result:
			g = glade_galaxy(row, base_cartography.unpacked_healpix)
			contained_galaxies.append(g)

		print("Query returned %s galaxies" % len(contained_galaxies))




		# Save cartograpy
		with open('%s/%s_contained_galaxies.pkl' % (self.options.working_dir, self.options.gw_id), 'wb') as handle:
			pickle.dump(contained_galaxies, handle, protocol=pickle.HIGHEST_PROTOCOL)

		avg_dist = average_distance_prior(base_cartography.unpacked_healpix)
		catalog_completeness = GLADE_completeness(avg_dist)
		print("Completeness: %s" % catalog_completeness)

		working_galaxies = copy.deepcopy(contained_galaxies)
			
		print("Assigning relative prob...")
		Cartographer.assign_galaxy_relative_prob(base_cartography.unpacked_healpix, 
												 working_galaxies,
												 sql_tile_cartography.cumlative_prob_in_tiles,
												 catalog_completeness)


		print("Redistribute prob...")
		redistributed_map = Cartographer.redistribute_probability(base_cartography.unpacked_healpix,
																  working_galaxies, 
																  sql_tile_cartography.tiles,
																  catalog_completeness)

		# S190425z_swope_cartography = Cartographer("S190425z", unpacked_healpix, swope, swope_all_sky_coords)
		redistributed_cartography = Cartographer("%s_%s" % (base_cartography.gwid, detector.name), 
													redistributed_map, 
													detector, 
													detector_all_sky_coords)

		# Save cartograpy
		with open('%s/%s_redstributed_cartography.pkl' % (self.options.working_dir, self.options.gw_id), 'wb') as handle:
			pickle.dump(redistributed_cartography, handle, protocol=pickle.HIGHEST_PROTOCOL)


		# plot_probability_map("%s/%s_%s_Redistributed_90th_Tiles" % (self.options.working_dir, self.options.gw_id, detector.name),
		# 			 pixels=redistributed_cartography.unpacked_healpix.pixels_90,
		# 			 tiles=redistributed_cartography.tiles)

		def GetSexigesimalString(c):
			ra = c.ra.hms
			dec = c.dec.dms

			ra_string = "%02d:%02d:%05.2f" % (ra[0],ra[1],ra[2])
			if dec[0] >= 0:
				dec_string = "+%02d:%02d:%05.2f" % (dec[0],np.abs(dec[1]),np.abs(dec[2]))
			else:
				dec_string = "%03d:%02d:%05.2f" % (dec[0],np.abs(dec[1]),np.abs(dec[2]))

			# Python has a -0.0 object. If the deg is this (because object lies < 60 min south), the string formatter will drop the negative sign
			if c.dec < 0.0 and dec[0] == 0.0:
				dec_string = "-00:%02d:%05.2f" % (np.abs(dec[1]),np.abs(dec[2]))
			return (ra_string, dec_string)


		sorted_tiles = sorted(redistributed_cartography.tiles, 
			key=lambda x: x.net_prob, 
			reverse=True)

		with open('%s/%s_%s_%s_Tiles.txt' % (self.options.working_dir, 
			base_cartography.gwid, 
			self.options.schedule_designation, 
			detector.name),'w') as csvfile:

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

			for i, st in enumerate(sorted_tiles):

				coord_str = GetSexigesimalString(st.coord) 

				cols = []
				field_name = "%s%s%sE%s" % (self.options.telescope_abbreviation,
					str(self.options.event_number).zfill(3),
					self.options.schedule_designation,
					str(i+1).zfill(4))

				cols.append(field_name)
				cols.append(coord_str[0])
				cols.append(coord_str[1])
				cols.append(detector.name)
				cols.append(self.options.filter)
				cols.append(self.options.exp_time)
				cols.append(st.net_prob)
				cols.append('False')
				csvwriter.writerow(cols)

			print("Done")


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


