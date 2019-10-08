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
			print("\tfetching results...")
			while True:
				r = cursor.fetchmany(1000000)
				count = len(r)
				streamed_results += r
				size_in_mb = sys.getsizeof(streamed_results)/1.0e+6

				print("\t\tfetched: %s; current length: %s; running size: %0.3f MB" % (count, len(streamed_results), size_in_mb))

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


		parser.add_option('--gw_id', default="", type="str",
						help='LIGO superevent name, e.g. `S190425z` ')

		parser.add_option('--healpix_dir', default='../Events/{GWID}', type="str",help='Directory for where to look for the healpix file.')

		parser.add_option('--healpix_file', default="", type="str", help='Healpix filename.')

		parser.add_option('--candidate_file', default="{FILENAME}", type="str", help='Optional filename of candidates to plot.')

		return(parser)

	def main(self):

		is_error = False
		has_candidates = False

		# Parameter checks
		if self.options.gw_id == "":
			is_error = True
			print("GWID is required.")

		formatted_healpix_dir = self.options.healpix_dir
		if "{GWID}" in formatted_healpix_dir:
			formatted_healpix_dir = formatted_healpix_dir.replace("{GWID}", self.options.gw_id)


		hpx_path = "%s/%s" % (formatted_healpix_dir, self.options.healpix_file)
		candidate_file_path = "%s/Candidates/%s" % (formatted_healpix_dir, self.options.candidate_file)

		if self.options.healpix_file == "":
			is_error = True
			print("You must specify which healpix file to process.")

		if os.path.exists(candidate_file_path):
			has_candidates = True
		else:
			print("Skipping plot candidates.")

		if is_error:
			print("Exiting...")
			return 1
	



		# Get Map ID
		healpix_map_select = "SELECT id, NSIDE FROM HealpixMap WHERE GWID = '%s' and Filename = '%s'"
		healpix_map_id = query_db([healpix_map_select % (self.options.gw_id, self.options.healpix_file)])[0][0][0]
		healpix_map_nside = query_db([healpix_map_select % (self.options.gw_id, self.options.healpix_file)])[0][0][1]

		# Get configured Detectors
		detector_select = "SELECT id, Name, Deg_width, Deg_width, Deg_radius, Area, MinDec, MaxDec FROM Detector"
		detector_result = query_db([detector_select])[0]
		detectors = [Detector(dr[1], float(dr[2]), float(dr[2]), detector_id=int(dr[0])) for dr in detector_result]

		# Get all observed tiles for all configured detectors
		observed_tile_select = '''
			SELECT 
				id,
				Detector_id, 
				FieldName, 
				RA, 
				_Dec, 
				EBV, 
				N128_SkyPixel_id, 
				Band_id, 
				MJD, 
				Exp_Time, 
				Mag_Lim, 
				HealpixMap_id 
			FROM
				ObservedTile 
			WHERE
				HealpixMap_id = %s and 
				Detector_id = %s 
		'''
		observed_tiles = {}
		for d in detectors:

			ot_result = query_db([observed_tile_select % (healpix_map_id, d.id)])[0]
			observed_tiles[d.name] = [
				Tile(float(ot[3]), 
					float(ot[4]), 
					d.deg_width, 
					d.deg_height, 
					healpix_map_nside, 
					tile_id=int(ot[0])) for ot in ot_result]

			print("Loaded %s %s tiles..." % (len(observed_tiles[d.name]), d.name))


		# load candidates if they exist
		candidates = []
		if has_candidates:
			with open(candidate_file_path,'r') as csvfile:

				csvreader = csv.reader(csvfile, delimiter=',')
				next(csvreader) # skip header
				
				for row in csvreader:
					name = row[0]
					ra = float(row[1])
					dec = float(row[2])
					flag1 = bool(int(row[3]))
					flag2 = bool(int(row[4]))
					is_keck = bool(int(row[5]))

					# append tuple
					candidates.append((name, coord.SkyCoord(ra,dec,unit=(u.deg,u.deg)), flag1, flag2, is_keck))


		select_pix = '''
			SELECT 
				running_prob.id, 
				running_prob.HealpixMap_id, 
				running_prob.Pixel_Index, 
				running_prob.Prob, 
				running_prob.Distmu, 
				running_prob.Distsigma, 
				running_prob.Mean, 
				running_prob.Stddev, 
				running_prob.Norm, 
				running_prob.N128_SkyPixel_id, 
				running_prob.cum_prob 
			FROM 
				(SELECT 
					hp_prob.id, 
					hp_prob.HealpixMap_id, 
					hp_prob.Pixel_Index,
					hp_prob.Prob, 
					hp_prob.Distmu, 
					hp_prob.Distsigma, 
					hp_prob.Mean, 
					hp_prob.Stddev, 
					hp_prob.Norm, 
					hp_prob.N128_SkyPixel_id, 
					SUM(hp_prob.Prob) OVER(ORDER BY hp_prob.Prob DESC) AS cum_prob 
				FROM 
					(SELECT 
						hp.id, 
						hp.HealpixMap_id, 
						hp.Pixel_Index,
						hp.Prob, 
						hp.Distmu, 
						hp.Distsigma, 
						hp.Mean, 
						hp.Stddev, 
						hp.Norm, 
						hp.N128_SkyPixel_id 
					FROM HealpixPixel hp 
					WHERE hp.HealpixMap_id = %s 
					ORDER BY
						hp.Prob DESC) hp_prob
					GROUP BY
						hp_prob.id, 
						hp_prob.HealpixMap_id, 
						hp_prob.Pixel_Index,
						hp_prob.Prob, 
						hp_prob.Distmu, 
						hp_prob.Distsigma, 
						hp_prob.Mean, 
						hp_prob.Stddev, 
						hp_prob.Norm, 
						hp_prob.N128_SkyPixel_id 
					) running_prob 
			WHERE 
				running_prob.cum_prob <= 0.9 
		'''

		print("Selecting map pixels...")
		map_pix_result = query_db([select_pix % healpix_map_id])[0]
		print("...done")

		print("Building pixel elements...")
		map_pix = [Pixel_Element(int(mp[2]), healpix_map_nside, float(mp[3]), pixel_id=int(mp[0])) for mp in map_pix_result]
		map_pix_sorted = sorted(map_pix, key=lambda x: x.prob, reverse=True)
		print("...done")

		
		cutoff_50th = 0.5
		cutoff_90th = 0.9
		index_50th = 0
		index_90th = 0

		print("Find index for 50th...")
		cum_prob = 0.0
		for i in range(len(map_pix_sorted)):
			cum_prob += map_pix_sorted[i].prob
			index_50th = i

			if (cum_prob >= cutoff_50th):
				break

		print("... %s" % index_50th)


		print("Find index for 90th...")
		cum_prob = 0.0
		for i in range(len(map_pix_sorted)):
			cum_prob += map_pix_sorted[i].prob
			index_90th = i

			if (cum_prob >= cutoff_90th):
				break
		print("... %s" % index_90th)


		# print("Build multipolygons...")
		# net_50_polygon = []
		# for p in map_pix_sorted[0:index_50th]:
		# 	net_50_polygon += p.query_polygon
		# joined_50_poly = unary_union(net_50_polygon)
		
		# # Fix any seams
		# eps = 0.00001
		# merged_50_poly = []
		# smoothed_50_poly = joined_50_poly.buffer(eps, 1, join_style=JOIN_STYLE.mitre).buffer(-eps, 1, join_style=JOIN_STYLE.mitre)
		
		# try:
		# 	test_iter = iter(smoothed_50_poly)
		# 	merged_50_poly = smoothed_50_poly
		# except TypeError as te:
		# 	merged_50_poly.append(smoothed_50_poly)

		# print("Number of sub-polygons in `merged_50_poly`: %s" % len(merged_50_poly))
		# sql_50_poly = SQL_Polygon(merged_50_poly, detectors[0])



		# net_90_polygon = []
		# for p in map_pix_sorted[0:index_90th]:
		# 	net_90_polygon += p.query_polygon
		# joined_90_poly = unary_union(net_90_polygon)
		
		# # Fix any seams
		# merged_90_poly = []
		# smoothed_90_poly = joined_90_poly.buffer(eps, 1, join_style=JOIN_STYLE.mitre).buffer(-eps, 1, join_style=JOIN_STYLE.mitre)
		
		# try:
		# 	test_iter = iter(smoothed_90_poly)
		# 	merged_90_poly = smoothed_90_poly
		# except TypeError as te:
		# 	merged_90_poly.append(smoothed_90_poly)

		# print("Number of sub-polygons in `merged_90_poly`: %s" % len(merged_90_poly))
		# sql_90_poly = SQL_Polygon(merged_90_poly, detectors[0])
		# print("... done.")



		sql_50_poly = None
		with open('sql50.pkl', 'rb') as handle:
			sql_50_poly = pickle.load(handle)

		sql_90_poly = None
		with open('sql90.pkl', 'rb') as handle:
			sql_90_poly = pickle.load(handle)
		



		fig = plt.figure(figsize=(10,10), dpi=1000)
		ax = fig.add_subplot(111)

		m = Basemap(projection='stere',
				 lon_0=15.0,
				 lat_0=-20.0,
				 llcrnrlat=-35.0,
				 urcrnrlat=-19.5,
				 llcrnrlon=8.0,
				 urcrnrlon=24.5)
		# # m = Basemap(projection='moll',lon_0=180.0)

		# print("Plotting `pixels_filled`...")

		# Scale colormap
		pix_90 = map_pix_sorted[0:index_90th]
		pixel_probs = [p.prob for p in pix_90]
		min_prob = np.min(pixel_probs)
		max_prob = np.max(pixel_probs)
		# min_prob = 1.341511144989834e-06
		# max_prob = 6.890843958619067e-05  

		print("min prob: %s" % min_prob)
		print("max prob: %s" % max_prob)

		# fracs = pixels_probs/max_prob
		# norm = colors.Normalize(fracs.min(), fracs.max())
		# norm = colors.LogNorm(min_prob, max_prob)
		norm = colors.Normalize(min_prob, max_prob)
		# norm = colors.LogNorm(1e-18, max_prob)
		# log_lower_limit = 1e-8	
		# norm = colors.LogNorm(log_lower_limit, max_prob)

		


		

		
			

		# clrs = {
		# 	"SWOPE":"yellow",
		# 	"NICKEL":"brown",
		# 	"THACHER":"blue",
		# 	"ANDICAM":"red",
		# 	"MOSFIRE":"green",
		# }

		clrs = {
			"SWOPE":(230.0/256.0, 159/256.0, 0),
			"NICKEL":(0, 114.0/256.0, 178.0/256.0),
			"THACHER":(0, 158.0/256.0, 115.0/256.0),
			"MOSFIRE":(204.0/256.0, 121.0/256.0, 167.0/256.0),
			# "ANDICAM":"red",
			# "MOSFIRE":"green"
		}


		


		# Plot SWOPE, then THACHER, then NICKEL, then MOSFIRE
		x1,y1 = m(0,0)
		m.plot(x1, y1, marker='s', markeredgecolor="k", markerfacecolor=clrs["SWOPE"], markersize=20, label="Swope", linestyle='None')
		m.plot(x1, y1, marker='s', markeredgecolor="k", markerfacecolor=clrs["THACHER"], markersize=16, label="Thacher", linestyle='None')
		m.plot(x1, y1, marker='s', markeredgecolor="k", markerfacecolor=clrs["NICKEL"], markersize=14, label="Nickel", linestyle='None')
		m.plot(x1, y1, marker='s', markeredgecolor="k", markerfacecolor=clrs["MOSFIRE"], markersize=10, label="MOSFIRE", linestyle='None')

		tile_opacity = 0.1
		print("Plotting Tiles for: %s (%s)" % ("SWOPE", clrs["SWOPE"]))
		for i, t in enumerate(observed_tiles["SWOPE"]):
			t.plot(m, ax, edgecolor=clrs["SWOPE"], facecolor=clrs["SWOPE"], linewidth=0.25, alpha=tile_opacity)
			t.plot(m, ax, edgecolor='k', facecolor="None", linewidth=0.25, alpha=1.0)

		print("Plotting Tiles for: %s (%s)" % ("THACHER", clrs["THACHER"]))
		for i, t in enumerate(observed_tiles["THACHER"]):
			t.plot(m, ax, edgecolor=clrs["THACHER"], facecolor=clrs["THACHER"], linewidth=0.25, alpha=tile_opacity)
			t.plot(m, ax, edgecolor='k', facecolor="None", linewidth=0.25, alpha=1.0)

		print("Plotting Tiles for: %s (%s)" % ("NICKEL", clrs["NICKEL"]))
		for i, t in enumerate(observed_tiles["NICKEL"]):
			t.plot(m, ax, edgecolor=clrs["NICKEL"], facecolor=clrs["NICKEL"], linewidth=0.25, alpha=tile_opacity)
			t.plot(m, ax, edgecolor='k', facecolor="None", linewidth=0.25, alpha=1.0)

		print("Plotting Tiles for: %s (%s)" % ("MOSFIRE", clrs["MOSFIRE"]))
		for i, t in enumerate(observed_tiles["MOSFIRE"]):
			t.plot(m, ax, edgecolor=clrs["MOSFIRE"], facecolor=clrs["MOSFIRE"], linewidth=0.25, alpha=1.0, zorder=9999)
			t.plot(m, ax, edgecolor='k', facecolor="None", linewidth=0.25, alpha=1.0, zorder=9999)


		print("Plotting (%s) `pixels`..." % len(pix_90))
		for i,p in enumerate(pix_90):
			p.plot(m, ax, facecolor=plt.cm.Greys(norm(p.prob)), edgecolor='None', linewidth=0.5, alpha=norm(p.prob)*0.8)

		print("Plotting SQL Multipolygons")


		# with open('sql50.pkl', 'wb') as handle:
		# 	pickle.dump(sql_50_poly, handle, protocol=pickle.HIGHEST_PROTOCOL)

		# with open('sql90.pkl', 'wb') as handle:
		# 	pickle.dump(sql_90_poly, handle, protocol=pickle.HIGHEST_PROTOCOL)
		sql_50_poly.plot(m, ax, edgecolor='black', linewidth=1.0, facecolor='None')
		sql_90_poly.plot(m, ax, edgecolor='gray', linewidth=0.75, facecolor='None')




		

		# Plotted off the map so that the legend will have a line item
		if has_candidates:
			for c in candidates:
				x,y = m(c[1].ra.degree, c[1].dec.degree)

				if c[4]:
					m.plot(x, y, marker='*', markeredgecolor='k', markerfacecolor='gray', markersize=20.0, zorder=9999)
				else:
					mkrclr = 'gray'
					if c[2] and c[3]: # passes both Charlie and Ryan cuts
						mkrclr = 'red'
					m.plot(x, y, marker='o', markeredgecolor='k', markerfacecolor=mkrclr, markersize=5.0)


				

			x1,y1 = m(0,0)
			m.plot(x1, y1, marker='.', linestyle="None", markeredgecolor="k", markerfacecolor="red", markersize=20, label="Viable candidate")

			x2,y2 = m(0,0)
			m.plot(x2, y2, marker='.', linestyle="None", markeredgecolor="k", markerfacecolor="grey", markersize=20, label="Excluded")

			x3,y3 = m(0,0)
			m.plot(x3, y3, marker='*', linestyle="None", markeredgecolor="k", markerfacecolor="grey", markersize=20, label="Keck Observed")


		# # # -------------- Use this for mollweide projections -------------- 
		# # meridians = np.arange(0.,360.,60.)
		# # m.drawparallels(np.arange(-90.,91.,30.),fontsize=14,labels=[True,True,False,False],dashes=[2,2],linewidth=0.5) # , xoffset=2500000
		# # m.drawmeridians(meridians,labels=[False,False,False,False],dashes=[2,2],linewidth=0.5)
		# # # ---------------------------------------------------------------- 
		


		# # # -------------- Use this for stereographic projections ----------
		# # # draw meridians
		# # meridians = np.arange(0.,360.,10.)
		# # par = m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=18,zorder=-1,color='gray', linewidth=0.5)

		# # # draw parallels
		# # parallels = np.arange(-90.,90.,10.)
		# # par = m.drawparallels(parallels,labels=[0,1,0,0],fontsize=18,zorder=-1,color='gray', linewidth=0.5, xoffset=230000)
		# # ## ----------------------------------------------------------------

		# # for mer in meridians[1:]:
		# # 	plt.annotate("%0.0f" % mer,xy=m(mer,0),xycoords='data', fontsize=14, zorder=9999)



		


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

		

		ax.annotate('', xy=(_90_x1,_90_y1), xytext=(_90_x2,_90_y2), xycoords='axes fraction', arrowprops={'arrowstyle': '-', 'lw':2.0 })
		ax.annotate('', xy=(_50_x1,_50_y1), xytext=(_50_x2,_50_y2), xycoords='axes fraction', arrowprops={'arrowstyle': '-', 'lw':2.0 })

		ax.annotate('90th percentile', xy=(_50_text_x,_50_text_y), xycoords='axes fraction', fontsize=sm_label_size)
		ax.annotate('50th percentile', xy=(_90_text_x,_90_text_y), xycoords='axes fraction', fontsize=sm_label_size)

		parallels = np.arange(-90.,90.,10.)
		dec_ticks = m.drawparallels(parallels,labels=[0,1,0,0])
		for i,tick_obj in enumerate(dec_ticks):
			a = coord.Angle(tick_obj, unit=u.deg)
			
			for text_obj in dec_ticks[tick_obj][1]:        
				direction = '+' if a.dms[0] > 0.0 else '-'
				text_obj.set_text(r'${0}{1:0g}^{{\degree}}$'.format(direction,np.abs(a.dms[0])))
				text_obj.set_size(sm_label_size)
				x = text_obj.get_position()[0]
				
				new_x = x*(1.0+0.08)
				text_obj.set_x(new_x)


		# draw meridians
		meridians = np.arange(0.,360.,7.5)
		ra_ticks = m.drawmeridians(meridians,labels=[0,0,0,1])

		RA_label_dict = {
			7.5:r'$00^{\mathrm{h}}30^{\mathrm{m}}$',
			15.0:r'$01^{\mathrm{h}}00^{\mathrm{m}}$',
			22.5:r'$01^{\mathrm{h}}30^{\mathrm{m}}$',
			
		}

		for i,tick_obj in enumerate(ra_ticks):
			# a = coord.Angle(tick_obj, unit=u.deg)
			# print(a.hms)
			print(tick_obj)
			for text_obj in ra_ticks[tick_obj][1]:
				if tick_obj in RA_label_dict:
					text_obj.set_text(RA_label_dict[tick_obj])
					text_obj.set_size(sm_label_size)
			

		sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Greys)
		sm.set_array([]) # can be an empty list

		tks = np.linspace(min_prob, max_prob, 6)
		# tks = np.logspace(np.log10(min_prob), np.log10(max_prob), 11)
		tks_strings = []
		
		for t in tks:
			tks_strings.append('%0.2f' % (t*100))


		cb = fig.colorbar(sm, ax=ax, ticks=tks, orientation='vertical', fraction=0.04875, pad=0.02, alpha=0.80) # 0.08951
		cb.ax.set_yticklabels(tks_strings, fontsize=16)
		cb.set_label("2D Pixel Probability", fontsize=label_size, labelpad=9.0)

		cb.ax.tick_params(width=2.0, length=6.0)

		cb.outline.set_linewidth(2.0)

		for axis in ['top','bottom','left','right']:
			ax.spines[axis].set_linewidth(2.0)

		
		ax.invert_xaxis()
		ax.legend(loc='upper left', fontsize=sm_label_size, borderpad=0.35, handletextpad=0.0, labelspacing=0.4)
		# ax.legend(fontsize=14)

		plt.ylabel(r'$\mathrm{Declination}$',fontsize=label_size,labelpad=36)
		plt.xlabel(r'$\mathrm{Right\;Ascension}$',fontsize=label_size,labelpad=30)

		fig.savefig('Keck.png', bbox_inches='tight') #,dpi=840
		plt.close('all')
		print("... Done.")
		
		


if __name__ == "__main__":
	
	useagestring="""python TileCandidatesMap.py [options]

Example with healpix_dir defaulted to 'Events/<gwid>':
python TileCandidatesMap.py --gw_id <gwid> --healpix_file <filename> --candidate_file <filename>
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
	print("Teglon `TileCandidatesMap` execution time: %s" % duration)
	print("********* end DEBUG ***********\n")


