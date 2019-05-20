import numpy as np
from astropy import units as u
import astropy.coordinates as coord
from shapely import geometry
import healpy as hp
from matplotlib.patches import Polygon

class Tile:
	def __init__(self, coord, width, height, nside, net_prob=0.0):
		
		self.coord = coord
		self.width = width
		self.height = height
		self.nside = nside
		self.net_prob = net_prob
		self.mwe = 0.0

		# Gets set after construction
		self.field_name = None

		# Properties
		self.__corner_coords = np.array([])
		self.__corner_xyz = np.array([])
		self.__polygon = None
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
	
	@property
	def polygon(self):

		if not self.__polygon:
			tile_coords = self.corner_coords
			# tile_vertices = [[coord.ra.radian*np.cos(coord.dec.radian), 
			# 				  coord.dec.radian] for coord in tile_coords]
			tile_vertices = [[coord.ra.radian, 
							  coord.dec.radian] for coord in tile_coords]

			self.__polygon = geometry.Polygon(tile_vertices)
		
		return self.__polygon
		
		# if not self.__polygon:        
		# 	tile_coords = self.corner_coords
			
		# 	# Hack -- corner cords need to be in +/- 180 format, not 0-
		# 	# tile_vertices = [[coord.ra.degree - 360.0, coord.dec.degree] for coord in tile_coords]

		# 	# Hack: convert the RA of the galaxy from [0,360] to [-180,180]
		# 	# tile_vertices = []
		# 	# for c in tile_coords:
		# 	# 	t_ra = c.ra.degree
		# 	# 	if t_ra > 180:
		# 	# 		t_ra = c.ra.degree - 360.

		# 	# 	tile_vertices.append([t_ra, c.dec.degree])


		# 	tile_vertices = [[coord.ra.degree, coord.dec.degree] for coord in tile_coords]
			
		# 	self.__polygon = geometry.Polygon(tile_vertices)
		
		# return self.__polygon
	
	@property
	def enclosed_pixel_indices(self):
		
		if len(self.__enclosed_pixel_indices) == 0:
			internal_pix = hp.query_polygon(self.nside, self.corner_xyz, inclusive=False)
			self.__enclosed_pixel_indices = internal_pix
			
		return self.__enclosed_pixel_indices
	
	def plot(self, bmap, ax_to_plot, **kwargs): #bmap, 

		# ra_deg,dec_deg = zip(*[(coord_deg[0], coord_deg[1])
		# 					   for coord_deg in self.polygon.exterior.coords])

		# ra_deg,dec_deg = zip(*[(np.degrees(coord_rad[0]/np.cos(coord_rad[1])), np.degrees(coord_rad[1])) 
		# 					   for coord_rad in self.polygon.exterior.coords])

		ra_deg,dec_deg = zip(*[(np.degrees(coord_rad[0]), np.degrees(coord_rad[1])) 
							   for coord_rad in self.polygon.exterior.coords])
		
		x,y = bmap(ra_deg,dec_deg)
		lat_lons = np.vstack([x,y]).transpose()
		ax_to_plot.add_patch(Polygon(lat_lons, **kwargs))

# Assumes rectangular tiles and contiguous contours...
def build_tile_list(contour_polygon, healpix_obj, tile_width, tile_height):
	
	tiles = []
	
	# Get list of RA and DEC at boundary of contour
	ra_deg,dec_deg = zip(*[(coord[0], coord[1]) for coord in contour_polygon.exterior.coords])
	
	# Define width offsets for tiles
	dec_offset = coord.Angle(tile_height, unit=u.deg)
	dec_offset_half = coord.Angle(tile_height/2., unit=u.deg)
	dec_offset_threehalf = coord.Angle(3*tile_height/2., unit=u.deg)    
	
	# Convert to highest prob tile
	ipix_max = np.argmax(healpix_obj.prob)
	theta, phi = hp.pix2ang(healpix_obj.nside, ipix_max)
	ra_max = coord.Angle(phi*u.rad).wrap_at(180*u.deg).degree
	dec_max = coord.Angle((0.5*np.pi - theta)*u.rad).degree
	
	max_tile = Tile(coord.SkyCoord(ra_max,
								   dec_max,
								   unit=(u.deg,u.deg)),
					tile_width, tile_height, healpix_obj.nside)

	print("\tBuilding tiles to the north...")
	
	current_max_tile = max_tile
	while not contour_polygon.intersection(current_max_tile.polygon).is_empty:
		tiles.append(current_max_tile)
		
		# Get offset in RA for this declination...
		d = current_max_tile.coord.dec.radian
		width_corr = tile_width/np.abs(np.cos(d))
		ra_offset = coord.Angle(width_corr, unit=u.deg)

		current_tile = current_max_tile
		left_tile = Tile(coord.SkyCoord(current_tile.coord.ra + ra_offset, 
										current_tile.coord.dec), 
						 tile_width, tile_height, healpix_obj.nside)

		while not contour_polygon.intersection(left_tile.polygon).is_empty:
			tiles.append(left_tile)
			current_tile = left_tile
			left_tile = Tile(coord.SkyCoord(current_tile.coord.ra.degree + ra_offset.degree, 
											current_tile.coord.dec.degree, unit=(u.deg,u.deg)), 
							 tile_width, tile_height, healpix_obj.nside)


		right_tile = Tile(coord.SkyCoord(current_max_tile.coord.ra - ra_offset, 
										 current_max_tile.coord.dec), 
						  tile_width, tile_height, healpix_obj.nside)

		while not contour_polygon.intersection(right_tile.polygon).is_empty:
			tiles.append(right_tile)
			current_tile = right_tile
			right_tile = Tile(coord.SkyCoord(current_tile.coord.ra.degree - ra_offset.degree, 
											 current_tile.coord.dec.degree, unit=(u.deg,u.deg)), 
							  tile_width, tile_height, healpix_obj.nside)

	
	
		# Go up to get new max...
		dec_slice = [(min(ra_deg), (current_max_tile.coord.dec + dec_offset_half).degree),
					 (min(ra_deg), (current_max_tile.coord.dec + dec_offset_threehalf).degree),
					 (max(ra_deg), (current_max_tile.coord.dec + dec_offset_threehalf).degree),
					 (max(ra_deg), (current_max_tile.coord.dec + dec_offset_half).degree)]

		dec_slice_xyz = []
		for c in dec_slice:
			theta = 0.5 * np.pi - np.deg2rad(c[1])
			phi = np.deg2rad(c[0])

			dec_slice_xyz.append(hp.ang2vec(theta, phi))

		internal_pix = hp.query_polygon(healpix_obj.nside, 
										np.asarray(dec_slice_xyz), inclusive=False)

		dec_slice_pix = healpix_obj.prob[internal_pix]
		ipix_max = np.argmax(dec_slice_pix)
		theta, phi = hp.pix2ang(healpix_obj.nside, internal_pix[ipix_max])

		# Convert to highest prob tile
		new_ra_max = coord.Angle(phi*u.rad).wrap_at(180*u.deg).degree
		current_max_tile = Tile(coord.SkyCoord(new_ra_max,current_max_tile.coord.dec + dec_offset, unit=(u.deg,u.deg)),
										   tile_width, tile_height, healpix_obj.nside)
	
	print("\tBuilding tiles to the south...")

	# Go down to get new max...
	dec_slice = [(min(ra_deg), (max_tile.coord.dec - dec_offset_half).degree),
				 (min(ra_deg), (max_tile.coord.dec - dec_offset_threehalf).degree),
				 (max(ra_deg), (max_tile.coord.dec - dec_offset_threehalf).degree),
				 (max(ra_deg), (max_tile.coord.dec - dec_offset_half).degree)]
								
	dec_slice_xyz = []
	for c in dec_slice:
		theta = 0.5 * np.pi - np.deg2rad(c[1])
		phi = np.deg2rad(c[0])
		dec_slice_xyz.append(hp.ang2vec(theta, phi))
	
	internal_pix = hp.query_polygon(healpix_obj.nside, 
										np.asarray(dec_slice_xyz), inclusive=False)
	dec_slice_pix = healpix_obj.prob[internal_pix]
	ipix_max = np.argmax(dec_slice_pix)
	theta, phi = hp.pix2ang(healpix_obj.nside, internal_pix[ipix_max])

	# Convert to highest prob tile
	new_ra_max = coord.Angle(phi*u.rad).wrap_at(180*u.deg).degree
	current_max_tile = Tile(coord.SkyCoord(new_ra_max,max_tile.coord.dec - dec_offset, unit=(u.deg,u.deg)),
										   tile_width, tile_height, healpix_obj.nside)

	
	while not contour_polygon.intersection(current_max_tile.polygon).is_empty:
		tiles.append(current_max_tile)
		
		# Get offset in RA for this declination...
		d = current_max_tile.coord.dec.radian
		width_corr = tile_width/np.abs(np.cos(d))
		ra_offset = coord.Angle(width_corr, unit=u.deg)

		current_tile = current_max_tile
		left_tile = Tile(coord.SkyCoord(current_tile.coord.ra + ra_offset, 
										current_tile.coord.dec), 
						 tile_width, tile_height, healpix_obj.nside)

		while not contour_polygon.intersection(left_tile.polygon).is_empty:
			tiles.append(left_tile)
			current_tile = left_tile
			left_tile = Tile(coord.SkyCoord(current_tile.coord.ra.degree + ra_offset.degree, 
											current_tile.coord.dec.degree, unit=(u.deg,u.deg)), 
							 tile_width, tile_height, healpix_obj.nside)


		right_tile = Tile(coord.SkyCoord(current_max_tile.coord.ra - ra_offset, 
										 current_max_tile.coord.dec), 
						  tile_width, tile_height, healpix_obj.nside)

		while not contour_polygon.intersection(right_tile.polygon).is_empty:
			tiles.append(right_tile)
			current_tile = right_tile
			right_tile = Tile(coord.SkyCoord(current_tile.coord.ra.degree - ra_offset.degree, 
											 current_tile.coord.dec.degree, unit=(u.deg,u.deg)), 
							  tile_width, tile_height, healpix_obj.nside)

		# Go up to get new max...
		dec_slice = [(min(ra_deg), (current_max_tile.coord.dec - dec_offset_half).degree),
					 (min(ra_deg), (current_max_tile.coord.dec - dec_offset_threehalf).degree),
					 (max(ra_deg), (current_max_tile.coord.dec - dec_offset_threehalf).degree),
					 (max(ra_deg), (current_max_tile.coord.dec - dec_offset_half).degree)]

		dec_slice_xyz = []
		for c in dec_slice:
			theta = 0.5 * np.pi - np.deg2rad(c[1])
			phi = np.deg2rad(c[0])

			dec_slice_xyz.append(hp.ang2vec(theta, phi))

		internal_pix = hp.query_polygon(healpix_obj.nside, 
										np.asarray(dec_slice_xyz), inclusive=False)

		dec_slice_pix = healpix_obj.prob[internal_pix]
		ipix_max = np.argmax(dec_slice_pix)
		theta, phi = hp.pix2ang(healpix_obj.nside, internal_pix[ipix_max])

		# Convert to highest prob tile
		new_ra_max = coord.Angle(phi*u.rad).wrap_at(180*u.deg).degree
		current_max_tile = Tile(coord.SkyCoord(new_ra_max,current_max_tile.coord.dec - dec_offset, unit=(u.deg,u.deg)),
										   tile_width, tile_height, healpix_obj.nside)
		
	return tiles