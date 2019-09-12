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
import optparse

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

			if commit: # used for updates, etc
				db.commit()

			streamed_results = []
			print("fetching results...")
			while True:
				r = cursor.fetchmany(1000000)
				count = len(r)
				streamed_results += r
				size_in_mb = sys.getsizeof(streamed_results)/1.0e+6

				print("\tfetched: %s; current length: %s; running size: %0.3f MB" % (count, len(streamed_results), size_in_mb))

				if not r or count < chunk_size:
					break
	
		results.append(streamed_results)
 

		# cnx = mysql.connector.connect(user=db_user, password=db_pwd, host=db_host, port=db_port, database=db_name)
		# cursor = cnx.cursor()
		# for result in cursor.execute(query_string, multi=True):

		# 	streamed_results = []
		# 	print("fetching results...")

		# 	i = 0
		# 	while True:
		# 		i += chunk_size
		# 		print("Fetching: %s records" % i)
				
		# 		partial_result = result.fetchmany(chunk_size)
		# 		count = len(partial_result)
		# 		streamed_results += partial_result
		# 		size_in_mb = sys.getsizeof(streamed_results)/1.0e+6

		# 		print("\tfetched: %s; current length: %s; running size: %0.3f MB" % (count, len(streamed_results), size_in_mb))

		# 		if not partial_result or count < chunk_size:
		# 			break
			
		# 	results.append(streamed_results)

	except Error as e:
		print('Error:', e)
	finally:
		cursor.close()
		# cnx.close()
		db.close()

	# fake = [[[(1)]]]
	return results
	# return fake

def batch_query(query_list):
	return_data = []
	batch_size = 500
	ii = 0
	jj = batch_size
	kk = len(query_list)

	print("\nLength of data to query: %s" % kk)
	print("Query batch size: %s" % batch_size)
	print("Starting loop...")

	number_of_queries = len(query_list)//batch_size
	if len(query_list) % batch_size > 0:
		number_of_queries += 1

	query_num = 1
	payload = []
	while jj < kk:
		t1 = time.time()

		print("%s:%s" % (ii,jj))
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

	print("\n%s:%s" % (ii,kk))
	
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

# Set up dustmaps config
config["data_dir"] = "./"

# Generate all pixel indices
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

def get_healpix_pixel_id(galaxy_info):
	phi, theta = np.radians(float(galaxy_info[8])), 0.5*np.pi - np.radians(float(galaxy_info[9]))
	
	# map NSIDE is last argument of galaxy_info
	# return the galaxy_id with the pixel index in the NSIDE of the healpix map 
	return (galaxy_info[0], hp.ang2pix(int(galaxy_info[-1]), theta, phi)) 

class Teglon:

	def add_options(self, parser=None, usage=None, config=None):
		import optparse
		if parser == None:
			parser = optparse.OptionParser(usage=usage, conflict_handler="resolve")


		parser.add_option('--gw_id', default="", type="str", help='LIGO superevent name, e.g. `S190425z`.')

		parser.add_option('--healpix_file', default="", type="str", help='Healpix filename. Used with `gw_id` to identify unique map.')

		parser.add_option('--healpix_dir', default='../Events/{GWID}', type="str",help='Directory for where to look for the healpix file.')

		parser.add_option('--output_dir', default="../Events/{GWID}/ObservedTiles/TileStats", type="str", help='Directory for where to look for observed tiles to import.')

		return(parser)

	def main(self):

		is_error = False

		# Parameter checks
		if self.options.gw_id == "":
			is_error = True
			print("GWID is required.")

		formatted_healpix_dir = self.options.healpix_dir
		if "{GWID}" in formatted_healpix_dir:
			formatted_healpix_dir = formatted_healpix_dir.replace("{GWID}", self.options.gw_id)

		formatted_output_dir = self.options.output_dir
		if "{GWID}" in formatted_output_dir:
			formatted_output_dir = formatted_output_dir.replace("{GWID}", self.options.gw_id)

		hpx_path = "%s/%s" % (formatted_healpix_dir, self.options.healpix_file)

		if self.options.healpix_file == "":
			is_error = True
			print("You must specify which healpix file to process.")

		if is_error:
			print("Exiting...")
			return 1

		
		# Get Map ID
		healpix_map_select = "SELECT id, NSIDE FROM HealpixMap WHERE GWID = '%s' and Filename = '%s'"
		healpix_map_id = int(query_db([healpix_map_select % (self.options.gw_id, self.options.healpix_file)])[0][0][0])
		healpix_map_nside = int(query_db([healpix_map_select % (self.options.gw_id, self.options.healpix_file)])[0][0][1])
		
		# Get all conposed observed tiles for this map
		observed_tile_select = '''
			SELECT 
				ot.id as ObservedTile_id, 
				ot.FieldName, 
				ot.RA, 
				ot._Dec, 
				ot.MJD, 
				ot.Exp_Time, 
				ot.Mag_Lim, 
				d.Name as Detector_Name, 
				ot.EBV, 
				b.Name as Band_Name, 
				ot.EBV*b.F99_Coefficient as A_lambda, 
				SUM(hp.Prob) as LIGO_2D_Tile_Prob, 
				SUM(hpc.Renorm2DProb) as Renorm_2D_Tile_Prob, 
				SUM(hpc.NetPixelProb) as Teglon_4D_Tile_Prob, # This should equal the contribution from the galaxies + redistributed prob
				AVG(hpc.PixelCompleteness) as Avg_Tile_Completeness, 
				AVG(hp.Mean) as Avg_LIGO_Distance 
			FROM ObservedTile ot 
			JOIN Detector d on d.id = ot.Detector_id 
			JOIN Band b on b.id = ot.Band_id 
			JOIN ObservedTile_HealpixPixel ot_hp on ot_hp.ObservedTile_id = ot.id 
			JOIN HealpixPixel hp on hp.id = ot_hp.HealpixPixel_id 
			JOIN HealpixPixel_Completeness hpc on hpc.HealpixPixel_id = hp.id 
			WHERE ot.HealpixMap_id = %s 
			GROUP BY 
				ot.id, 
				ot.FieldName, 
				ot.RA, 
				ot._Dec, 
				ot.MJD, 
				ot.Exp_Time, 
				ot.Mag_Lim, 
				d.Name, 
				ot.EBV, 
				b.Name, 
				ot.EBV*b.F99_Coefficient 
			ORDER BY ot.id 
		'''
		observed_tile_result = query_db([observed_tile_select % healpix_map_id])[0]


		observed_tile_galaxy_select = '''
			SELECT
				ot.id as ObservedTile_id, 
				ot.FieldName, 
				ot.RA as ObservedTile_RA, 
				ot._Dec as ObservedTile_Dec, 
				hp.id as HealpixPixel_id, 
				gd2.id as Galaxy_id, 
				gd2.RA as Galaxy_RA, 
				gd2._Dec as Galaxy_Dec, 
				gd2.PGC, 
				gd2.Name_GWGC, 
				gd2.Name_HyperLEDA, 
				gd2.Name_2MASS, 
				gd2.Name_SDSS_DR12, 
				hp_gd2_w.GalaxyProb as Galaxy_4D_Prob, 
				gd2.z, 
				gd2.z_dist, 
				gd2.z_dist_err, 
				gd2.B, 
				gd2.K 
			FROM 
				GalaxyDistance2 gd2 
			JOIN 
				HealpixPixel_GalaxyDistance2 hp_gd2 on  hp_gd2.GalaxyDistance2_id = gd2.id 
			JOIN 
				HealpixPixel_GalaxyDistance2_Weight hp_gd2_w on hp_gd2_w.HealpixPixel_GalaxyDistance2_id = hp_gd2.id 
			JOIN 
				HealpixPixel hp on hp.id = hp_gd2.HealpixPixel_id 
			JOIN 
				HealpixMap hm on hm.id = hp.HealpixMap_id 
			JOIN 
				ObservedTile_HealpixPixel ot_hp on ot_hp.HealpixPixel_id = hp.id 
			JOIN 
				ObservedTile ot on ot.id = ot_hp.ObservedTile_id 
			WHERE 
				ot.id = %s
		'''
		for ot in observed_tile_result:
			ot_id = ot[0]
			observed_tile_galaxy_result = query_db([observed_tile_galaxy_select % ot_id])[0]


			field_name = ot[1]
			csv_file_name = "%s_%s.txt" % (ot_id, field_name)
			csv_file_path = "%s/%s" % (formatted_output_dir, csv_file_name)

			print("Creating `%s`" % csv_file_path)
			with open(csv_file_path,'w') as file:

				# Write key-value pairs from the tile to the CSV
				file.write("ID:\t%s\n" % ot[0])
				file.write("OBJECT:\t%s\n" % ot[1])
				file.write("RA:\t%s\n" % ot[2])
				file.write("DEC:\t%s\n" % ot[3])
				file.write("MJD:\t%s\n" % ot[4])
				file.write("EXPTM:\t%s\n" % ot[5])
				file.write("INSTR:\t%s\n" % ot[7])
				file.write("FILTER:\t%s\n" % ot[9])
				file.write("EBV:\t%s\n" % ot[8])
				file.write("MWE:\t%s\n" % ot[10])
				file.write("LIMMAG:\t%s\n" % ot[6])
				file.write("NET2D:\t%s\n" % ot[11])
				file.write("RDST2D:\t%s\n" % ot[12])
				file.write("NET4D:\t%s\n" % ot[13])
				file.write("PXDIST:\t%s\n" % ot[15])
				file.write("PXCOMP:\t%s\n" % ot[14])
				file.write("\n----------------------Contained Galaxies-------------------------\n\n")
				file.write("Galaxy_ID, Galaxy_RA, Galaxy_Dec, PGC, Name_GWGC, Name_HyperLEDA, Name_2MASS, Name_SDSS_DR12, Galaxy_4D_Prob, z, z_Dist, z_Dist_Err, B, K\n")
				csvwriter = csv.writer(file)
				for otgr in observed_tile_galaxy_result:

					csvwriter.writerow((otgr[5], 
						otgr[6], 
						otgr[7], 
						otgr[8], 
						otgr[9], 
						otgr[10], 
						otgr[11], 
						otgr[12], 
						otgr[13], 
						otgr[14], 
						otgr[15], 
						otgr[16], 
						otgr[17], 
						otgr[18]))



if __name__ == "__main__":
	
	useagestring="""python Generate_Tiles.py [options]

Example with healpix_dir defaulted to 'Events/<gwid>':
python LoadMap.py --gw_id <>gwid --healpix_file <filename>

Example with healpix_dir specified:
python LoadMap.py --gw_id <gwid> --healpix_dir Events/<directory name> --healpix_file <filename>
"""
	
	start = time.time()

	teglon = Teglon()
	parser = teglon.add_options(usage=useagestring)
	options,  args = parser.parse_args()
	teglon.options = options
	
	teglon.main()

	end = time.time()
	duration = (end - start)
	print("\n********* start DEBUG ***********")
	print("Teglon `LoadObservedTiles` execution time: %s" % duration)
	print("********* end DEBUG ***********\n")


