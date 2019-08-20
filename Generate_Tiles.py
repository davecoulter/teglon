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
from SQL_Polygon import *
from Pixel_Element import *

import xml.etree.ElementTree as ET

import csv

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

		mosfire_deg_width = 6.14/60.0
		mosfire_deg_height = 6.14/60.0

		self.telescope_mapping = {
		"S":Detector("SWOPE", swope_deg_width, swope_deg_height),
		"A":Detector("ANDICAM", andicam_deg_width, andicam_deg_height),
		"T":Detector("THACHER", thacher_deg_width, thacher_deg_height),
		"N":Detector("NICKEL", nickel_deg_width, nickel_deg_height),
		"M":Detector("MOSFIRE", mosfire_deg_width, mosfire_deg_height)
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

		parser.add_option('--healpix_dir', default='./', type="str",help='Directory for where to look for the healpix file.')

		parser.add_option('--healpix_file', default="", type="str",
						help='healpix filename.')

		parser.add_option('--working_dir', default='./', type="str",help='Working directory for where to look for files and where to put output.')

		parser.add_option('--gw_id', default="", type="str",
						help='LIGO superevent name, e.g. `S190425z` ')

		parser.add_option('--event_number', default=1, type="int",
							  help='event number (default=%default)')

		parser.add_option('--schedule_designation', default='AA', type="str",
							  help='schedule designation (default=%default)')

		parser.add_option('--percentile', default='-1', type="float",
							  help='Percentile to tile to -- use this flag to set the percentile to < 0.90')

		parser.add_option('--skip_completeness', action="store_true", help='If True, purely tiles sky map', default=False)

		return(parser)

	def main(self):

		### UNPACKING THE HEALPIX FILES ###

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
				detector = Detector(self.options.telescope_name, self.options.detector_width_deg, self.options.detector_height_deg)
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
			detector.name, detector.deg_width, detector.deg_height))
		print("%s FOV area: %s" % (detector.name, (detector.deg_width * detector.deg_height)))


		if is_custom_percentile:
			print("\nTiling to %s percentile." % self.options.percentile)

		t1 = time.time()
		
		print("Unpacking '%s':%s..." % (self.options.gw_id, hpx_path))
		prob, distmu, distsigma, distnorm, header_gen = hp.read_map(hpx_path, field=(0,1,2,3), h=True)
		header = dict(header_gen)

		t2 = time.time()

		print("\n********* start DEBUG ***********")
		print("`Unpacking healpix` execution time: %s" % (t2 - t1))
		print("********* end DEBUG ***********\n")

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
		

		percentile = self.options.percentile
		unpacked_healpix = Unpacked_Healpix(hpx_path,
											prob, 
											distmu, 
											distsigma, 
											distnorm, 
											header, 
											nside, 
											npix, 
											area_per_px, 
											linestyle="-",
											compute_contours=False,
											custom_percentile=custom_percentile)

		num_50 = len(unpacked_healpix.indices_of_50)
		num_90 = len(unpacked_healpix.indices_of_90)

		area_50 = num_50*area_per_px
		area_90 = num_90*area_per_px
		

		# Debug -- should match Grace DB statistics
		print("\nArea of 50th: %s" % area_50)
		print("Area of 90th: %s\n" % area_90)

		if is_custom_percentile:
			num_custom = len(unpacked_healpix.pixels_custom)
			area_custom = num_custom*area_per_px
			print("\n[User defined] Area of %s: %s\n" % (custom_percentile, area_custom))





		### GENERATE THE ALL-SKY TILING FOR A DETECTOR & GET THE ENCLOSED GALAXIES ###

		print("Generating all sky coords for %s" % detector.name)
		detector_all_sky_coords = Cartographer.generate_all_sky_coords(detector)
		base_cartography = Cartographer(self.options.gw_id, unpacked_healpix, detector, detector_all_sky_coords, generate_tiles=True)

		# Save cartograpy
		with open('%s/%s_base_cartography.pkl' % (self.options.working_dir, self.options.gw_id), 'wb') as handle:
			pickle.dump(base_cartography, handle, protocol=pickle.HIGHEST_PROTOCOL)


		if not self.options.skip_completeness:
			# Build the spatial query
			t1 = time.time()

			# Join all tiles together into a composite polygon to simplify the SQL query
			net_polygon = []
			for t in base_cartography.tiles:
				net_polygon += t.query_polygon
			joined_poly = unary_union(net_polygon)
			
			# Fix any seams
			eps = 0.00001
			merged_poly = joined_poly.buffer(eps, 1, join_style=JOIN_STYLE.mitre).buffer(-eps, 1, join_style=JOIN_STYLE.mitre)


			try:
				print("Number of sub-polygons in query: %s" % len(merged_poly))
			except TypeError as e:
				print("Warning...")
				print(e)
				print("\nOnly one polygons in query! Wrapping `merged_poly` and resuming...")

				merged_poly = [merged_poly]

			# Create and save sql_poly
			sql_poly = SQL_Polygon(merged_poly, detector)
			with open('%s/%s_sql_poly.pkl' % (self.options.working_dir, self.options.gw_id), 'wb') as handle:
				pickle.dump(sql_poly, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
			# # Use the sql_poly string to create the WHERE clause
			mp_where = "ST_WITHIN(Coord, ST_GEOMFROMTEXT('"
			mp_where += sql_poly.query_polygon_string
			mp_where += "', 4326));"

			t2 = time.time()

			print("\n********* start DEBUG ***********")
			print("`Generating multipolygon` execution time: %s" % (t2 - t1))
			print("********* end DEBUG ***********\n")

			
			# Database I/O
			t1 = time.time()

			query = "SELECT * from GalaxyDistance2 WHERE z_dist IS NOT NULL AND z_dist_err IS NOT NULL AND B IS NOT NULL AND "
			query += mp_where

			print(query)
			print("\n*****************************\n")

			# Save cartograpy
			with open('%s/%s_query.pkl' % (self.options.working_dir, self.options.gw_id), 'wb') as handle:
				pickle.dump(query, handle, protocol=pickle.HIGHEST_PROTOCOL)

			result = QueryDB(query, port=LOCAL_PORT)

			t2 = time.time()

			print("\n********* start DEBUG ***********")
			print("`Query database` execution time: %s" % (t2 - t1))
			print("********* end DEBUG ***********\n")

			# Instantiate galaxies
			contained_galaxies = []

			# What is the angular radius for our given map?
			cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
			pixel_diameter_arcsec = np.degrees(pixel_radius_radian)*2.0*3600.0
			proper_radius = 20.0 # kpc

			z = np.linspace(1e-18,0.3,1000)
			arcsec_of_proper_diam = cosmo.arcsec_per_kpc_proper(z)*proper_radius*2.0
			z_index = find_nearest(arcsec_of_proper_diam, pixel_diameter_arcsec)
			target_z = z[z_index]
			dist_cuttoff = cosmo.luminosity_distance(target_z).value # Mpc

			print("Redshift cutoff: %s" % target_z)
			print("Distance cutoff: %s" % dist_cuttoff)

			for row in result:
				g = glade_galaxy(row, base_cartography.unpacked_healpix, cosmo, dist_cuttoff, proper_radius)
				contained_galaxies.append(g)

			print("Query returned %s galaxies" % len(contained_galaxies))

			avg_dist = average_distance_prior(base_cartography.unpacked_healpix)
			catalog_completeness = GLADE_completeness(avg_dist)
			print("Completeness: %s" % catalog_completeness)





			### REDISTRIBUTE THE PROBABILITY IN THE MAP TO THE GALAXIES ###

			print("Assigning relative prob...")
			Cartographer.assign_galaxy_relative_prob(base_cartography.unpacked_healpix, 
													 contained_galaxies,
													 base_cartography.cumlative_prob_in_tiles,
													 catalog_completeness)


			# Save contained galaxies
			with open('%s/%s_contained_galaxies.pkl' % (self.options.working_dir, self.options.gw_id), 'wb') as handle:
				pickle.dump(contained_galaxies, handle, protocol=pickle.HIGHEST_PROTOCOL)

			print("Redistribute prob...")
			# redistributed_map
			redistributed_prob, enclosed_indices = Cartographer.redistribute_probability(base_cartography.unpacked_healpix,
																	  contained_galaxies,
																	  base_cartography.tiles,
																	  catalog_completeness)

			# Copy base cartography (have to do this?) and update the prob of the 90th perecentil
			redistributed_cartography = copy.deepcopy(base_cartography)
			redistributed_cartography.unpacked_healpix.prob = redistributed_prob
			redistributed_cartography.unpacked_healpix.pixels_90 = [Pixel_Element(ei, 
																				  redistributed_cartography.unpacked_healpix.nside, 
																				  redistributed_cartography.unpacked_healpix.prob[ei]) 
																	for ei in enclosed_indices]
			

			# Update the original tiles with the new probability
			redistributed_cartography.assign_tiles(base_cartography.tiles)
			# Save cartograpy
			with open('%s/%s_redstributed_cartography.pkl' % (self.options.working_dir, self.options.gw_id), 'wb') as handle:
				pickle.dump(redistributed_cartography, handle, protocol=pickle.HIGHEST_PROTOCOL)

		else:
			print("\n\nPure tiling...")



		### SAVE THE RESULTS ###
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


		tiles_to_serialize = None
		if not self.options.skip_completeness:
			tiles_to_serialize = redistributed_cartography.tiles
		else:
			tiles_to_serialize = base_cartography.tiles


		t1 = time.time()
		sorted_tiles = sorted(tiles_to_serialize, key=lambda x: x.net_prob, reverse=True)

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

				c = coord.SkyCoord(st.ra_deg, st.dec_deg, unit=(u.deg, u.deg))
				coord_str = GetSexigesimalString(c) 

				cols = []
				field_name = "%s%s%sE%s" % (self.options.telescope_abbreviation,
					str(self.options.event_number).zfill(3),
					self.options.schedule_designation,
					str(i+1).zfill(5))

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

		t2 = time.time()

		print("\n********* start DEBUG ***********")
		print("`Serialize tiles` execution time: %s" % (t2 - t1))
		print("********* end DEBUG ***********\n")


if __name__ == "__main__":
	
	import os
	import optparse

	useagestring="""python Generate_Tiles.py [options]

Default telescope example:
python Generate_Tiles.py --gw_id S190425z --healpix_dir S190521g --healpix_file bayestar.fits.gz,0 --working_dir S190521g_Swope  --event_number 14 --telescope_abbreviation S --filter r --exp_time 180 --schedule_designation AA

Custom telescope example:
python Generate_Tiles.py --gw_id S190425z --healpix_dir S190521g --healpix_file bayestar.fits.gz,0 --working_dir S190521g_SOAR  --event_number 14 --telescope_abbreviation SO --telescope_name SOAR --detector_width_deg 0.12 --detector_height_deg 0.12 --filter r --exp_time 180 --schedule_designation AA
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


