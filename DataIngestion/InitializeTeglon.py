import matplotlib
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
from configparser import RawConfigParser
import multiprocessing as mp
import mysql.connector
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

import healpy as hp
from ligo.skymap import distance

from HEALPix_Helpers import *
from Tile import *
from SQL_Polygon import *
from Pixel_Element import *
from Completeness_Objects import *


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
configFile = os.path.join(__location__, 'Settings.ini')

db_config = RawConfigParser()
db_config.read(configFile)


db_name = db_config.get('database', 'DATABASE_NAME')
db_user = db_config.get('database', 'DATABASE_USER')
db_pwd = db_config.get('database', 'DATABASE_PASSWORD')
db_host = db_config.get('database', 'DATABASE_HOST')
db_port = db_config.get('database', 'DATABASE_PORT')

isDEBUG = True
build_skydistances = False
build_skypixels = False
build_detectors = False
build_bands = False
build_MWE = True
build_galaxy_skypixel_associations = False



# Database SELECT
def QueryDB(query):
	results = []

	cnx = mysql.connector.connect(user=db_user, password=db_pwd, host=db_host, port=db_port, database=db_name)
	cursor = cnx.cursor()
	cursor.execute(query)
	results = cursor.fetchall()
	
	cursor.close()
	cnx.close()
	
	return results

def MultiQueryDB(queries):

	# lenofresult = 0
	query_list = ";".join(queries)
	results = []

	cnx = mysql.connector.connect(user=db_user, password=db_pwd, host=db_host, port=db_port, database=db_name)
	cursor = cnx.cursor()
	
	for result in cursor.execute(query_list, multi=True):
		r = result.fetchall()

		# lenofresult += len(r)
		results += r
		# results.append(r)
	
	cursor.close()
	cnx.close()
	
	# print("Len: %s" % lenofresult)

	return results

# Database INSERT
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

	number_of_inserts = len(insert_data)//batch_size
	if len(insert_data) % batch_size > 0:
		number_of_inserts += 1

	insert_num = 1
	payload = []
	while j < k:

		t1 = time.time()

		print("%s:%s" % (i,j))
		payload = insert_data[i:j]

		if insert_records(insert_statement, payload):
			i = j
			j += batch_size
		else:
			raise("Error inserting batch! Exiting...")
		
		t2 = time.time()

		print("\n********* start DEBUG ***********")
		print("INSERT %s/%s complete - execution time: %s" % (insert_num, number_of_inserts, (t2 - t1)))
		print("********* end DEBUG ***********\n")

		insert_num += 1

	print("Out of loop...")

	t1 = time.time()

	print("\n%s:%s" % (i,k))
	
	payload = insert_data[i:k]
	if not insert_records(insert_statement, payload):
		raise("Error inserting batch! Exiting...")
	
	t2 = time.time()

	print("\n********* start DEBUG ***********")
	print("INSERT %s/%s complete - execution time: %s" % (insert_num, number_of_inserts, (t2 - t1)))
	print("********* end DEBUG ***********\n")

	_tend = time.time()

	print("\n********* start DEBUG ***********")
	print("batch_insert execution time: %s" % (_tend - _tstart))
	print("********* end DEBUG ***********\n")


initialize_start = time.time()

# Generate all pixel indices
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

nside2 = 2
nside2_npix = hp.nside2npix(nside2)
frac_sky_nside2 = (1/nside2_npix)

nside4 = 4
nside4_npix = hp.nside2npix(nside4)
frac_sky_nside4 = (1/nside4_npix)

nside8 = 8
nside8_npix = hp.nside2npix(nside8)
frac_sky_nside8 = (1/nside8_npix)

nside16 = 16
nside16_npix = hp.nside2npix(nside16)
frac_sky_nside16 = (1/nside16_npix)

nside32 = 32
nside32_npix = hp.nside2npix(nside32)
frac_sky_nside32 = (1/nside32_npix)

nside64 = 64
nside64_npix = hp.nside2npix(nside64)
frac_sky_nside64 = (1/nside64_npix)

nside128 = 128
nside128_npix = hp.nside2npix(nside128)
frac_sky_nside128 = (1/nside128_npix)

nside_npix = [nside2_npix, nside4_npix, nside8_npix, nside16_npix, nside32_npix, nside64_npix, nside128_npix]
nsides = [nside2, nside4, nside8, nside16, nside32, nside64, nside128]
frac = [frac_sky_nside2, frac_sky_nside4, frac_sky_nside8, frac_sky_nside16, frac_sky_nside32, frac_sky_nside64, frac_sky_nside128]

distance_at_resolution_change = [45,125,200,400,700,900,1220] # Mpc
steps_in_resolution_bin = [6,24,19,37,38,13,8]


if isDEBUG:
	print("NSIDE: %s, npix: %s, frac/pix: %0.3E" % (nside2, nside2_npix, frac_sky_nside2))
	print("NSIDE: %s, npix: %s, frac/pix: %0.3E" % (nside4, nside4_npix, frac_sky_nside4))
	print("NSIDE: %s, npix: %s, frac/pix: %0.3E" % (nside8, nside8_npix, frac_sky_nside8))
	print("NSIDE: %s, npix: %s, frac/pix: %0.3E" % (nside16, nside16_npix, frac_sky_nside16))
	print("NSIDE: %s, npix: %s, frac/pix: %0.3E" % (nside32, nside32_npix, frac_sky_nside32))
	print("NSIDE: %s, npix: %s, frac/pix: %0.3E" % (nside64, nside64_npix, frac_sky_nside64))
	print("NSIDE: %s, npix: %s, frac/pix: %0.3E" % (nside128, nside128_npix, frac_sky_nside128))
	print("\nDistances at resolution change: %s [Mpc]" % distance_at_resolution_change)

## Build Sky Distance Objects ##
# start_comoving_volume is Quantity [Mpc^3]
# end_comoving_volume is Quantity [Mpc^3]
# start_distance is Quantity [Mpc]
def compute_sky_distance(start_comoving_volume, end_comoving_volume, steps, start_distance, sky_fraction, nside):
	sky_distances = []
	volume_steps = np.linspace(start_comoving_volume, end_comoving_volume, steps)
	d1 = start_distance
	d2 = d1
	for i in range(len(volume_steps)):
		
		if i == 0:
			continue

		z = z_at_value(cosmo.comoving_volume, volume_steps[i], zmin=2e-5)
		d2 = cosmo.luminosity_distance(z)
		partial_vol = (volume_steps[i]-volume_steps[i-1])*sky_fraction
		sky_distances.append(Sky_Distance(d1,d2,partial_vol,nside))
		d1 = d2
	return sky_distances

################################################

if not build_skydistances:
	print("Skipping Sky Distances...")
else:
	## Build Sky Distance Objects ##
	print("\n\nBuilding Sky Distances...")

	sky_distances = []
	for i in range(len(distance_at_resolution_change)):

		start_vol = 0.0*u.Mpc**3
		end_vol = cosmo.comoving_volume((Distance(distance_at_resolution_change[i], u.Mpc)).z)
		start_distance = 0.0*u.Mpc
		
		if i > 0:
			start_vol = cosmo.comoving_volume((Distance(distance_at_resolution_change[i-1], u.Mpc)).z)
			start_distance = sky_distances[-1].D2*u.Mpc

		sky_distances += compute_sky_distance(start_vol, end_vol, steps_in_resolution_bin[i], start_distance, frac[i], nsides[i])

	if isDEBUG:
		print("\nGenerated distances:\n")
		for d in sky_distances:
			print(d)

	sky_distance_insert = "INSERT INTO SkyDistance (D1, D2, dCoV, Sch_L10, NSIDE) VALUES (%s,%s,%s,%s,%s)"
	sky_distance_data = []

	for d in sky_distances:
		sky_distance_data.append((d.D1, d.D2, d.dCoV, d.Sch_L10, d.NSIDE))

	print("\nInserting %s sky distances..." % len(sky_distance_data))
	if insert_records(sky_distance_insert, sky_distance_data):
		print("Success!")
	else:
		raise("Error with INSERT! Exiting...")
	print("...Done")

################################################

sky_pixels = None
if not build_skypixels:
	print("Skipping Sky Pixels...")
	print("\tLoading existing pixels...")
	with open('sky_pixels.pkl', 'rb') as handle:
		sky_pixels = pickle.load(handle)
else:
	## Build Sky Pixel Objects ##
	print("\n\nBuilding Sky Pixels...")
	sky_pixels = OrderedDict() # Ordered Dict because it will matter the order once we start iterating during the INSERT

	for i, nside in enumerate(nsides):

		print("Initializing NSIDE=%s pix..." % nside)
		sky_pixels[nside] = {}

		for j in range(nside_npix[i]):
			pe = Pixel_Element(j, nside, 0.0)
			pe.query_polygon_string # initialize
			sky_pixels[nside][j] = pe

		if i > 0:
			for j, (pi, pe) in enumerate(sky_pixels[nside].items()):
				# Get Parent Pixel
				theta, phi = hp.pix2ang(pe.nside, pe.index)
				parent_index = hp.ang2pix(nsides[i-1], theta, phi)

				if not parent_index in sky_pixels[nsides[i-1]]:
					raise("Orphaned NSIDE=%s pixel! Orphaned index, theta, phi: (%s, %s, %s)" % (nside, pe.index, theta, phi))
				else:
					parent_pixel = sky_pixels[nsides[i-1]][parent_index]
					sky_pixels[nside][j].parent_pixel = parent_pixel
		print("... created %s NSIDE=%s pix." % (len(sky_pixels[nside]), nside))

	sky_pixel_insert = "INSERT INTO SkyPixel (RA, _Dec, Coord, Poly, NSIDE, Pixel_Index, Parent_Pixel_id) VALUES (%s,%s,ST_PointFromText(%s, 4326),ST_GEOMFROMTEXT(%s, 4326),%s,%s,%s);"
	sky_pixel_select = "SELECT id, Pixel_Index FROM SkyPixel WHERE NSIDE = %s;"

	for i, (nside, pixel_dict) in enumerate(sky_pixels.items()):
		
		sky_pixel_data = []
		if i == 0: # NSIDE=2 has no parents to associate
			for pi, pe in pixel_dict.items():

				sky_pixel_data.append((pe.coord.ra.degree, pe.coord.dec.degree,
				"POINT(%s %s)" % (pe.coord.dec.degree, pe.coord.ra.degree - 180.0), # Dec, RA order due to MySQL convention for lat/lon
				pe.query_polygon_string, pe.nside, pe.index, None))

		else: # Associate parents

			parent_ids = QueryDB(sky_pixel_select % (nsides[i-1]))
			
			for row in parent_ids:
				sky_pixels[nsides[i-1]][int(row[1])].id = int(row[0]) # Set pixel id in parent...

			for pi, pe in pixel_dict.items():

				parent_id = sky_pixels[nsides[i-1]][pe.parent_pixel.index].id

				sky_pixel_data.append((pe.coord.ra.degree, pe.coord.dec.degree,
				"POINT(%s %s)" % (pe.coord.dec.degree, pe.coord.ra.degree - 180.0), # Dec, RA order due to MySQL convention for lat/lon
				pe.query_polygon_string, pe.nside, pe.index, parent_id))

		print("\nInserting %s sky pixels (NSIDE=%s)..." % (len(sky_pixel_data), nside))
		batch_insert(sky_pixel_insert, sky_pixel_data)

	# Get the n128 pixel ids
	n128_pixel_tuples = QueryDB(sky_pixel_select % nside128)
	for tup in n128_pixel_tuples:
		sky_pixels[nside128][int(tup[1])].id = int(tup[0])

	with open('sky_pixels.pkl', 'wb') as handle:
		pickle.dump(sky_pixels, handle, protocol=pickle.HIGHEST_PROTOCOL)

	if isDEBUG:
		print("\n\nTest pixel parent-child relationships...")
		# Pick a random nside32 pixel and plot all of its parents
		test_index = 110435
		print("Plotting n128 index[%s]" % test_index)
		n128 = sky_pixels[nside128][test_index]
		n64 = n128.parent_pixel
		n32 = n64.parent_pixel
		n16 = n32.parent_pixel
		n8 = n16.parent_pixel
		n4 = n8.parent_pixel
		n2 = n4.parent_pixel


		lon0=180.0
		fig = plt.figure(figsize=(30,30), dpi=1000)
		ax = fig.add_subplot(111)
		m = Basemap(projection='moll',lon_0=lon0)
			
		n2.plot(m, ax, facecolor='blue', edgecolor='blue', linewidth=0.5)
		n4.plot(m, ax, facecolor='red', edgecolor='red', linewidth=0.5)
		n8.plot(m, ax, facecolor='green', edgecolor='green', linewidth=0.5)
		n16.plot(m, ax, facecolor='magenta', edgecolor='magenta', linewidth=0.5)
		n32.plot(m, ax, facecolor='orange', edgecolor='orange', linewidth=0.5)
		n64.plot(m, ax, facecolor='pink', edgecolor='pink', linewidth=0.5)
		n128.plot(m, ax, facecolor='cyan', edgecolor='cyan', linewidth=0.5)

		meridians = np.arange(0.,360.,60.)
		m.drawparallels(np.arange(-90.,91.,30.),fontsize=14,labels=[True,True,False,False],dashes=[2,2],linewidth=0.5, xoffset=2500000)
		m.drawmeridians(meridians,labels=[False,False,False,False],dashes=[2,2],linewidth=0.5)
			
		for mer in meridians[1:]: # np.str(mer)
			plt.annotate("%0.0f" % mer,xy=m(mer,0),xycoords='data', fontsize=14, zorder=9999)

		for axis in ['top','bottom','left','right']:
			ax.spines[axis].set_linewidth(2.0)

		ax.invert_xaxis()

		fig.savefig('Pixel_Relation_Check.png', bbox_inches='tight') #,dpi=840
		plt.close('all')

		print("... Done.")

################################################

if not build_detectors:
	print("Skipping Detectors...")
else:
	## Build Detector Objects ##
	print("\n\nBuilding Detectors...")

	swope_deg_width = 4096*0.435/3600. # pix * plate scale / arcsec/pix => degrees
	swope_deg_height = 4112*0.435/3600.

	andicam_deg_width = 1024*0.371/3600.
	andicam_deg_height = 1024*0.371/3600.

	thacher_deg_width = 2048*0.609/3600.
	thacher_deg_height = 2048*0.609/3600.

	nickel_deg_width = 2048*0.368/3600.
	nickel_deg_height = 2048*0.368/3600.

	detector_data = [
		("SWOPE", swope_deg_width, swope_deg_height, None, swope_deg_width*swope_deg_height),
		("ANDICAM", andicam_deg_width, andicam_deg_height, None, andicam_deg_width*andicam_deg_height),
		("THACHER", thacher_deg_width, thacher_deg_height, None, thacher_deg_width*thacher_deg_height),
		("NICKEL", nickel_deg_width, nickel_deg_height, None, nickel_deg_width*nickel_deg_height)
	]

	detector_insert = "INSERT INTO Detector (Name, Deg_width, Deg_height, Deg_radius, Area) VALUES (%s, %s, %s, %s, %s);"
	print("\nInserting %s detectors..." % len(detector_data))
	if insert_records(detector_insert, detector_data):
		print("Success!")
	else:
		raise("Error with INSERT! Exiting...")
	print("...Done")

################################################

if not build_bands:
	print("Skipping Bands...")
else:
	## Build Band Objects ##
	print("\n\nBuilding Photometric Bands...")
	# Schlafly & Finkbeiner, 2018: https://arxiv.org/pdf/1012.4804.pdf

	# Name, Effective_Wavelength, F99 coefficient (Rv = 3.1); See Table 6.
	band_data = [
		("SDSS u", 3586.8, 4.239),
		("SDSS g", 4716.7, 3.303),
		("SDSS r", 6165.1, 2.285),
		("SDSS i", 7475.9, 1.698),
		("SDSS z", 8922.9, 1.263),
		("CTIO B", 4308.9, 3.641),
		("CTIO V", 5516.6, 2.682),
		("CTIO R", 6520.2, 2.119),
		("CTIO I", 8006.9, 1.516),
		("UKIRT J", 12482.9, 0.709),
		("UKIRT H", 16588.4, 0.449),
		("UKIRT K", 21897.7, 0.302)
	]

	band_insert = "INSERT INTO Band (Name, Effective_Wavelength, F99_Coefficient) VALUES (%s, %s, %s);"

	print("\nInserting %s bands..." % len(band_data))
	if insert_records(band_insert, band_data):
		print("Success!")
	else:
		raise("Error with INSERT! Exiting...")
	print("...Done")

################################################

if not build_MWE:
	print("Skipping MWE...")
else:
	## Build E(B-V) Map ##
	print("\n\nBuilding MWE...")

	config["data_dir"] = "./"

	nside128_RA = []
	nside128_DEC = []
	nside128_ids = []

	t1 = time.time()

	for n128, pe in sky_pixels[nside128].items():
		nside128_RA.append(pe.coord.ra.degree)
		nside128_DEC.append(pe.coord.dec.degree)
		nside128_ids.append(pe.id)

	c = coord.SkyCoord(nside128_RA, nside128_DEC, unit=(u.deg,u.deg))
	sfd = SFDQuery()
	ebv = sfd(c)

	t2 = time.time()

	print("\n********* start DEBUG ***********")
	print("EBV query - execution time: %s" % (t2 - t1))
	print("********* end DEBUG ***********\n")
	
	print("E(B-V) data length: %s" % len(ebv))
	min_ebv = np.min(ebv)
	max_ebv = np.max(ebv)
	print("Max E(B-V): %s" % min_ebv)
	print("Min E(B-V): %s" % max_ebv)

	mwe_insert = "INSERT INTO SkyPixel_EBV (N128_SkyPixel_id, EBV) VALUES (%s, %s);"
	mwe_data = [(nside128_ids[i], str(e)) for i,e in enumerate(ebv)]

	# batch_insert(mwe_insert, mwe_data)

	# Debug plot of of MWE
	if isDEBUG:
		print("Plotting MWE E(B-V) extinction...")

		lon0=180.0
		fig = plt.figure(figsize=(20,20), dpi=800)
		ax = fig.add_subplot(111)
		m = Basemap(projection='moll',lon_0=lon0)

		norm = colors.LogNorm(min_ebv, max_ebv)
		for pi,pe in sky_pixels[nside128].items():
		    pe.plot(m, ax, facecolor=plt.cm.inferno(norm(ebv[i])), edgecolor=plt.cm.inferno(norm(ebv[i])), linewidth=0.5, alpha=0.8, zorder=9900)

		meridians = np.arange(0.,360.,60.)
		m.drawparallels(np.arange(-90.,91.,30.),fontsize=14,labels=[True,True,False,False],dashes=[2,2],linewidth=0.5, xoffset=2500000)
		m.drawmeridians(meridians,labels=[False,False,False,False],dashes=[2,2],linewidth=0.5)

		for mer in meridians[1:]:
		    plt.annotate("%0.0f" % mer,xy=m(mer,0),xycoords='data', fontsize=14, zorder=9999)

		sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.inferno)
		sm.set_array([]) # can be an empty list

		tks = np.logspace(np.log10(min_ebv), np.log10(max_ebv), 11)
		tks_strings = []
		for t in tks:
		    tks_strings.append('%0.2f' % t)

		cb = fig.colorbar(sm, ax=ax, ticks=tks, orientation='horizontal', fraction=0.08951, pad=0.02, alpha=0.80) 
		cb.ax.set_xticklabels(tks_strings, fontsize=10)
		cb.set_label("E(B-V) per pixel", fontsize=14, labelpad=10.0)
		cb.outline.set_linewidth(1.0)

		for axis in ['top','bottom','left','right']:
		    ax.spines[axis].set_linewidth(2.0)
		ax.invert_xaxis()

		fig.savefig("MWE_Debug_Plot.png", bbox_inches='tight')
		plt.close('all')




if not build_galaxy_skypixel_associations:
	print("Skipping SkyPixel - GalaxyDistance2 Associations...")
else:
	## Build SkyPixel - Galaxy associations
	print("\n\nBuilding SkyPixel - GalaxyDistance2 Associations...")

	galaxy_association_insert = "INSERT INTO SkyPixel_GalaxyDistance2 (SkyPixel_id, GalaxyDistance2_id) VALUES (%s, %s);"

	top_level_galaxy_association_query = '''
		SELECT 
			%s as SkyPixel_id, id as GalaxyDistance2_id 
		FROM GalaxyDistance2 
		WHERE z_dist IS NOT NULL and B IS NOT NULL and ST_CONTAINS(ST_GEOMFROMTEXT('%s', 4326), Coord)
	''' 
	child_galaxy_association_query = '''
		SELECT 
			%s as SkyPixel_id, gd2.id as GalaxyDistance2_id 
		FROM GalaxyDistance2 gd2 
		INNER JOIN SkyPixel_GalaxyDistance2 sp_gd2 ON sp_gd2.GalaxyDistance2_id = gd2.id 
		INNER JOIN (SELECT DISTINCT Parent_Pixel_id FROM SkyPixel WHERE id in %s) filtered_parent on filtered_parent.Parent_Pixel_id = sp_gd2.SkyPixel_id 
		WHERE 
			gd2.B IS NOT NULL AND 
			gd2.z_dist IS NOT NULL and 
			ST_CONTAINS((SELECT child.Poly FROM SkyPixel child WHERE child.id=%s), gd2.Coord) 
	'''

	for i, (nside, pixel_dict) in enumerate(sky_pixels.items()):

		print("\nNumber of sky pixels: %s (NSIDE=%s)\n" % (len(pixel_dict), nside))
		galaxy_association_data = []
		batch_queries = []

		if i == 0: # NSIDE = 2
			
			for j, (pi, sp) in enumerate(pixel_dict.items()):
				batch_queries.append(top_level_galaxy_association_query % (sp.id, sp.query_polygon_string))
		else:
			for j, (pi, sp) in enumerate(pixel_dict.items()):

				# Get this pixel's neighbors. This will allow to get a distinct list of shared parents
				sibling_indices = list(hp.get_all_neighbours(sp.nside, sp.index)) + [sp.index]

				sibling_ids = []
				for si in sibling_indices:
					if si > -1:
						sibling_ids.append(sky_pixels[nside][si].id)


				sibling_ids_string = "("
				for si in sibling_ids:
					sibling_ids_string += "%s," % si
				sibling_ids_string = sibling_ids_string[:-1] + ")"

				batch_queries.append(child_galaxy_association_query % (sp.id, sibling_ids_string, sp.id))

		_tstart = time.time()

		batch_size = 500
		ii = 0
		jj = batch_size
		kk = len(batch_queries)

		print("\nLength of data to query: %s" % kk)
		print("Query batch size: %s" % batch_size)
		print("Starting loop...")

		number_of_queries = len(batch_queries)//batch_size
		if len(batch_queries) % batch_size > 0:
			number_of_queries += 1

		query_num = 1
		payload = []
		while jj < kk:

			t1 = time.time()

			print("%s:%s" % (ii,jj))
			payload = batch_queries[ii:jj]

			galaxy_association_data += MultiQueryDB(payload)

			ii = jj
			jj += batch_size
			
			t2 = time.time()

			print("\n********* start DEBUG ***********")
			print("Query %s/%s complete - execution time: %s" % (query_num, number_of_queries, (t2 - t1)))
			print("********* end DEBUG ***********\n")

			query_num += 1

		print("Out of loop...")

		t1 = time.time()

		print("\n%s:%s" % (ii,kk))
		
		payload = batch_queries[ii:kk]
		galaxy_association_data += MultiQueryDB(payload)
		
		
		t2 = time.time()

		print("\n********* start DEBUG ***********")
		print("Query %s/%s complete - execution time: %s" % (query_num, number_of_queries, (t2 - t1)))
		print("********* end DEBUG ***********\n")

		_tend = time.time()

		print("\n********* start DEBUG ***********")
		print("Batch Query execution time: %s" % (_tend - _tstart))
		print("********* end DEBUG ***********\n")

		print("\n\n Batch INSERT (%s) records..." % len(galaxy_association_data))
		batch_insert(galaxy_association_insert, galaxy_association_data)





# print("\n\nBuilding MWE. For now, loading from a pickel file...")

# # Rehydrate pixel elements
# print("Loading IRSA_N32_pixels.pkl...")
# MWE = None
# with open('IRSA_N32_pixels.pkl', 'rb') as handle:
# 	MWE = pickle.load(handle)

# band_select = "SELECT id, Name FROM Band;"
# bands = QueryDB(band_select)

# pixel_select = "SELECT id, Pixel_Index FROM SkyPixel WHERE NSIDE = 32;"
# pixels = QueryDB(pixel_select)

# mwe_insert = "INSERT INTO MWE (Band_id, A_SandF_mag, N32_SkyPixel_id) VALUES (%s, %s, %s);"

# mwe_data = []
# for pixel_index, filter_dict in MWE.items():
# 	pixel_id = next((p[0] for p in pixels if p[1] == pixel_index), None)

# 	if pixel_id is None:
# 		raise("Didn't find N32 Pixel_Index [%s] in MWE data file!" % pixel_index)

# 	for band, mag in filter_dict.items():
# 		band_id = next((b[0] for b in bands if b[1] == band), None)

# 		if band_id is None:
# 			raise("Didn't find Band `%s` for Pixel_Index [%s] in MWE data file!" % (band, pixel_index))

# 		mwe_data.append((band_id, mag, pixel_id))


# # print("\nInserting %s MWE records..." % len(mwe_data))
# # if insert_records(mwe_insert, mwe_data):
# # 	print("Success!")
# # else:
# # 	raise("Error with INSERT! Exiting...")
# # print("...Done")


# def plot_MWE(filter_key, label_text, MWE_dict, filename):
# 	lon0=180.0
# 	fig = plt.figure(figsize=(8,8), dpi=180)
# 	ax = fig.add_subplot(111)
# 	m = Basemap(projection='moll',lon_0=lon0)

# 	irsa_pixels = []
# 	values = []
# 	for pindex, filter_dict in MWE_dict.items():
# 		a_lambda = filter_dict[filter_key]
# 		values.append(a_lambda)
# 		irsa_pixels.append(Pixel_Element(pindex, nside32, a_lambda))

# 	min_ext = np.min(values)
# 	max_ext = np.max(values)

# 	norm = colors.LogNorm(min_ext, max_ext)

# 	for i,p in enumerate(irsa_pixels):
# 		p.plot(m, ax, facecolor=plt.cm.inferno(norm(p.prob)), edgecolor=plt.cm.inferno(norm(p.prob)), linewidth=0.5, alpha=0.8, zorder=9900)

# 	meridians = np.arange(0.,360.,60.)
# 	m.drawparallels(np.arange(-90.,91.,30.),fontsize=14,labels=[True,True,False,False],dashes=[2,2],linewidth=0.5, xoffset=2500000)
# 	m.drawmeridians(meridians,labels=[False,False,False,False],dashes=[2,2],linewidth=0.5)

# 	for mer in meridians[1:]:
# 		plt.annotate("%0.0f" % mer,xy=m(mer,0),xycoords='data', fontsize=14, zorder=9999)

# 	sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.inferno)
# 	sm.set_array([]) # can be an empty list


# 	tks = np.logspace(np.log10(min_ext), np.log10(max_ext), 11)
# 	tks_strings = []

# 	for t in tks:
# 		tks_strings.append('%0.2f' % t)

# 	top_left = 1.05
# 	delta_y = 0.04

# 	cb = fig.colorbar(sm, ax=ax, ticks=tks, orientation='horizontal', fraction=0.08951, pad=0.02, alpha=0.80) 
# 	cb.ax.set_xticklabels(tks_strings, fontsize=10)
	
	
# 	cb.set_label(label_text, fontsize=14, labelpad=10.0)
# 	cb.outline.set_linewidth(1.0)

# 	for axis in ['top','bottom','left','right']:
# 		ax.spines[axis].set_linewidth(2.0)

# 	ax.invert_xaxis()

# 	fig.savefig("%s" % filename, bbox_inches='tight') #,dpi=840
# 	plt.close('all')

# 	print("...`%s` Done." % filename)

# # # Debug plot of of MWE
# # if isDEBUG:
# # 	print("Plotting MWE r-band extinction...")
# # 	plot_MWE("SDSS r", r"$\mathrm{A_r}$ [mag] per Pixel", MWE, "SDSS_r_MWE.png")

initialize_end = time.time()
print("\n********* start DEBUG ***********")
print("Initialize Teglon execution time: %s" % (initialize_end - initialize_start))
print("********* end DEBUG ***********\n")
