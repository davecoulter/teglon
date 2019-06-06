import numpy as np
from astropy import units as u
import astropy.coordinates as coord
from shapely import geometry
import healpy as hp
from matplotlib.patches import Polygon
import statistics
from shapely.ops import linemerge, unary_union, polygonize, split
from abc import ABCMeta, abstractmethod, abstractproperty
import math


class Telgon_Shape(metaclass=ABCMeta):

	def get_coord_lists(poly, convert_radian=True):
		ra_deg,dec_deg = [],[]

		if convert_radian:
			ra_deg,dec_deg = zip(*[(np.degrees(c[0]), np.degrees(c[1])) for c in poly.exterior.coords])
		else:
			ra_deg,dec_deg = zip(*[(c[0], c[1]) for c in poly.exterior.coords])
			
		return ra_deg,dec_deg

	def get_coord_lists_transform(poly):
		ra_deg,dec_deg = zip(*[(c[0]-360.0, c[1]) if c[0] > 180.0 else (c[0], c[1]) for c in poly.exterior.coords])
		return ra_deg,dec_deg

	def get_coord_lists_unshift(poly, convert_radian=True):
		
		ra_deg,dec_deg = [],[]
		
		if convert_radian:
			ra_deg,dec_deg = zip(*[(np.degrees(c[0])+360.0, np.degrees(c[1])) for c in poly.exterior.coords])
		else:
			ra_deg,dec_deg = zip(*[(c[0]-360.0, c[1]) for c in poly.exterior.coords])

		return ra_deg,dec_deg
		
	def get_coord_tuple(ra_list, dec_list):
		return [[r,d] for r,d in zip(ra_list,dec_list)]

	def get_coord_tuple_shift(ra_list, dec_list):
		return [[r+360.0,d] for r,d in zip(ra_list,dec_list)]

	def get_coord_tuple_untransform(ra_list, dec_list):
		return [[r+360.0,d] if r < 0.0 else [r,d] for r,d in zip(ra_list,dec_list)]


	@property
	@abstractmethod
	def radius_proxy(self):
		pass

	@property
	@abstractmethod
	def polygon(self):
		pass

	def create_query_polygon(self, initial_poly_in_radian=True):

		query_polygon = []
		tolerance = self.radius_proxy/360.0
		
		for poly in self.polygon:

			ra_deg,dec_deg = Telgon_Shape.get_coord_lists(poly, convert_radian=initial_poly_in_radian)

			# Sanity - nothing can be > 360.0, and anything smaller than 1" from 0 is 0.
			bounded_ra = []
			for i in range(len(ra_deg)):
				if ra_deg[i] > 360.0:
					bounded_ra.append(360.0)
				elif ra_deg[i] <= 1.0/3600.0: # arcsecond
					bounded_ra.append(0.0)
				else:
					bounded_ra.append(ra_deg[i])

			# COSMETICS
			# If there are consecutive coordinates like (0, -50) and (0, -40), this is a constant line of RA
			# the plotting of Basemap will not plot this smoothly, even if it is correct in coordinate-space.
			# Try to inject "filler" points, e.g. (0, -49) .. (0, -41), which don't change the logical boundary,
			# but may help the plotter not plot something that appears discontinuous across a line of constant RA.
			prev_ra = bounded_ra[0]
			prev_dec = dec_deg[0]

			smoothed_ra = [prev_ra]
			smoothed_dec = [prev_dec]
			for i in range(len(ra_deg)):
				next_ra = bounded_ra[i]
				next_dec = dec_deg[i]

				jump_in_dec = next_dec-prev_dec
				sign = np.sign(jump_in_dec)
				
				# i.e., if greater than 1 degree on a constant line of RA
				if prev_ra == next_ra and np.abs(jump_in_dec) >= 1.0: 
					# Number of half-degree steps between Prev and Next dec
					steps = math.floor(np.abs(jump_in_dec)/0.5)
					for j in range(steps):
						interstitial_dec = prev_dec + sign*0.5*(j+1)
						smoothed_ra.append(next_ra)
						smoothed_dec.append(interstitial_dec)

				# Insert original next dec
				smoothed_ra.append(next_ra)
				smoothed_dec.append(next_dec)

				prev_ra = next_ra
				prev_dec = next_dec
			
			# We now have a smoothed and bounded polygon to work with...
			polygon_degree = geometry.Polygon(Telgon_Shape.get_coord_tuple(smoothed_ra,smoothed_dec))

			# Is this polygon close to the coordinate singularity?
			close_to_the_edge = np.any(np.asarray(ra_deg) < 90.0) and np.any(np.asarray(ra_deg) > 270.0)
			
			if not close_to_the_edge:
				query_polygon += [polygon_degree] # if not, it's fine as it is
			else:
				# If median is close to 360, it's an easteren polygon and convert 0s to 360s.
				if statistics.median(ra_deg) > 300.0:
					eastern_bounded_ra = []
					for i in range(len(ra_deg)):
						if np.isclose(ra_deg[i], 0.0, rtol=tolerance):
							eastern_bounded_ra.append(360.0)
						else:
							eastern_bounded_ra.append(ra_deg[i])
					
					query_polygon += [geometry.Polygon(Telgon_Shape.get_coord_tuple(eastern_bounded_ra,dec_deg))]
				else:
					# If this polygon straddles the singularity, we need to transform it to [-180, 180] space
					# and then shift the polygon away from the singularity to split the polygon into an east and 
					# west component
					shifted_ra, shifted_dec = Telgon_Shape.get_coord_lists_transform(polygon_degree)
					shifted_poly = geometry.Polygon(Telgon_Shape.get_coord_tuple_shift(shifted_ra, shifted_dec))
					meridian = geometry.LineString([(360.0,90),(360.0, -90)])
					split_poly_collection = split(shifted_poly, meridian)
					
					if len(split_poly_collection) > 1:
					
						east = None
						west = None

						for sp in split_poly_collection:
							# Once split, shift the polygon back to the origin of [-180, 180] space, and then
							# transform back to [0,360] space
							unshifted_ra, unshifted_dec = Telgon_Shape.get_coord_lists_unshift(sp, convert_radian=False)
							split_poly = geometry.Polygon(Telgon_Shape.get_coord_tuple_untransform(unshifted_ra, unshifted_dec))
							split_poly_ra, split_poly_dec = Telgon_Shape.get_coord_lists(split_poly, convert_radian=False)

							split_eastern_ra = []
							if np.max(split_poly_ra) > 300.0:
								for i in range(len(split_poly_ra)):
									if np.isclose(split_poly_ra[i], 0.0, rtol=tolerance):
										split_eastern_ra.append(360.0)
									else:
										split_eastern_ra.append(split_poly_ra[i])

								east = geometry.Polygon(Telgon_Shape.get_coord_tuple(split_eastern_ra,dec_deg))
							else:
								west = geometry.Polygon(Telgon_Shape.get_coord_tuple(split_poly_ra,dec_deg))

						query_polygon += [east,west]
					else:
						query_polygon += split_poly_collection

		return query_polygon

