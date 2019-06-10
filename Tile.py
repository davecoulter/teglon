import numpy as np
from astropy import units as u
import astropy.coordinates as coord
from shapely import geometry
import healpy as hp
from matplotlib.patches import Polygon
import statistics
from shapely.ops import linemerge, unary_union, polygonize, split
from Teglon_Shape import *

class Tile(Telgon_Shape):
	def __init__(self, coord, width, height, nside, net_prob=0.0):
		
		self.coord = coord
		self.width = width
		self.height = height

		self.nside = nside
		self.net_prob = net_prob
		self.mwe = 0.0

		# Gets set after construction
		self.field_name = None

		self.contained_galaxies = []
		self.mean_pixel_distance = None
		self.tile_glade_completeness = -999

		# Properties
		self.__corner_coords = np.array([])
		self.__corner_xyz = np.array([])
		
		# Used to check if this object is within this "radius proxy" of the coordinate signularity
		self.__radius_proxy = 5.0*np.sqrt(self.width*self.height/np.pi)
		self.__polygon = None
		self.__query_polygon = None
		# self.__query_polygon_string = None

		self.__enclosed_pixel_indices = np.array([])
		
	def __str__(self):
		return str(self.__dict__)

	@property
	def corner_coords(self):
		
		if len(self.__corner_coords) == 0:
		
			# Get correction factor:
			d = self.coord.dec.radian
			width_corr = self.width/np.abs(np.cos(d))

			# Define the tile offsets:
			ra_offset = coord.Angle(width_corr/2., unit=u.deg)
			dec_offset = coord.Angle(self.height/2., unit=u.deg)

			SW = coord.SkyCoord(self.coord.ra - ra_offset, self.coord.dec - dec_offset)
			NW = coord.SkyCoord(self.coord.ra - ra_offset, self.coord.dec + dec_offset)

			SE = coord.SkyCoord(self.coord.ra + ra_offset, self.coord.dec - dec_offset)
			NE = coord.SkyCoord(self.coord.ra + ra_offset, self.coord.dec + dec_offset)

			self.__corner_coords = np.asarray([SW,NW,NE,SE])
			
		return self.__corner_coords
	
	@property
	def corner_xyz(self):
		
		if len(self.__corner_xyz) == 0:
		
			coord_vertices = self.corner_coords
			xyz_vertices = []

			for c in coord_vertices:
				theta = 0.5 * np.pi - np.deg2rad(c.dec.degree)
				phi = np.deg2rad(c.ra.degree)

				xyz_vertices.append(hp.ang2vec(theta, phi))

			self.__corner_xyz = np.asarray(xyz_vertices)
			
		return self.__corner_xyz
	
	# Telgon_Shape properties
	@property
	def radius_proxy(self):
		return self.__radius_proxy

	@property # Returns list of polygon
	def polygon(self):

		if not self.__polygon:
			tile_vertices = [[coord.ra.radian, coord.dec.radian] for coord in self.corner_coords]
			self.__polygon = geometry.Polygon(tile_vertices)

		return [self.__polygon]

	# NOTE -- always assuming RA coordinate of the form [0, 360]
	@property
	def query_polygon(self):

		if not self.__query_polygon:
			self.__query_polygon = self.create_query_polygon(initial_poly_in_radian=True)
		return self.__query_polygon
	
	@property
	def enclosed_pixel_indices(self):
		
		if len(self.__enclosed_pixel_indices) == 0:
			internal_pix = hp.query_polygon(self.nside, self.corner_xyz, inclusive=False)
			self.__enclosed_pixel_indices = internal_pix
			
		return self.__enclosed_pixel_indices
	
	def plot(self, bmap, ax_to_plot, **kwargs): #bmap, 

		query_polygon = self.query_polygon
		for p in query_polygon:

			ra_deg,dec_deg = zip(*[(coord_deg[0], coord_deg[1]) 
								   for coord_deg in p.exterior.coords])

			x2,y2 = bmap(ra_deg,dec_deg)
			lat_lons = np.vstack([x2,y2]).transpose()
			ax_to_plot.add_patch(Polygon(lat_lons, **kwargs))
