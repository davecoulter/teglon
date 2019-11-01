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
    # def __init__(self, coord, width, height, nside, net_prob=0.0):
    def __init__(self, central_ra_deg, central_dec_deg, width, height, nside, net_prob=0.0, tile_id=None):

        self.id = tile_id

        # self.coord = coord

        # Test
        self.dec_deg = central_dec_deg
        self.ra_deg = central_ra_deg
        self.dec_rad = np.radians(central_dec_deg)
        self.ra_rad = np.radians(central_ra_deg)

        self.width = width
        self.height = height

        self.nside = nside
        self.net_prob = net_prob
        self.mwe = 0.0
        self.mjd = None
        self.mag_lim = None
        self.exp_time = None
        self.band_id = None

        # Gets set after construction
        self.field_name = None

        self.contained_galaxies = []
        self.mean_pixel_distance = None
        self.tile_glade_completeness = -999

        # Properties
        self.__corner_coords = np.array([])
        self.__corner_xyz = np.array([])

        # Used to check if this object is within this "radius proxy" of the coordinate signularity
        self.__radius_proxy = 5.0 * np.sqrt(self.width * self.height / np.pi)
        self.__polygon = None
        self.__query_polygon = None
        self.__query_polygon_string = None

        self.__enclosed_pixel_indices = np.array([])
        self.N128_pixel_index = None
        self.N128_pixel_id = None

    def __str__(self):
        return str(self.__dict__)

    @property
    def corner_coords(self):

        if len(self.__corner_coords) == 0:
            # # Get correction factor:
            # d = self.coord.dec.radian
            # width_corr = self.width/np.abs(np.cos(d))

            # # Define the tile offsets:
            # ra_offset = coord.Angle(width_corr/2., unit=u.deg)
            # dec_offset = coord.Angle(self.height/2., unit=u.deg)

            # southern_dec = (self.coord.dec - dec_offset) if (self.coord.dec - dec_offset).degree > -90 else coord.Angle((self.coord.dec - dec_offset).degree + 0.00001, unit=u.degree)
            # northern_dec = (self.coord.dec + dec_offset) if (self.coord.dec + dec_offset).degree < 90 else coord.Angle((self.coord.dec + dec_offset).degree - 0.00001, unit=u.degree)

            # SW = coord.SkyCoord(self.coord.ra - ra_offset, southern_dec)
            # NW = coord.SkyCoord(self.coord.ra - ra_offset, northern_dec)

            # SE = coord.SkyCoord(self.coord.ra + ra_offset, southern_dec)
            # NE = coord.SkyCoord(self.coord.ra + ra_offset, northern_dec)

            # self.__corner_coords = np.asarray([SW,NW,NE,SE])

            # Get correction factor:
            width_corr = self.width / np.abs(np.cos(self.dec_rad))

            # Define the tile offsets:
            ra_offset = (width_corr / 2.)
            dec_offset = (self.height / 2.)

            southern_dec = (self.dec_deg - dec_offset) if (self.dec_deg - dec_offset) > -90 else (
                        self.dec_deg - dec_offset + 0.00001)
            northern_dec = (self.dec_deg + dec_offset) if (self.dec_deg + dec_offset) < 90 else (
                        self.dec_deg + dec_offset - 0.00001)

            SW = ((self.ra_deg - ra_offset), southern_dec)
            NW = ((self.ra_deg - ra_offset), northern_dec)

            SE = ((self.ra_deg + ra_offset), southern_dec)
            NE = ((self.ra_deg + ra_offset), northern_dec)

            self.__corner_coords = np.asarray([SW, NW, NE, SE])

        return self.__corner_coords

    @property
    def corner_xyz(self):

        if len(self.__corner_xyz) == 0:

            # coord_vertices = self.corner_coords
            # xyz_vertices = []

            # for c in coord_vertices:
            # 	theta = 0.5 * np.pi - np.deg2rad(c.dec.degree)
            # 	phi = np.deg2rad(c.ra.degree)

            # 	xyz_vertices.append(hp.ang2vec(theta, phi))

            # self.__corner_xyz = np.asarray(xyz_vertices)

            coord_vertices = self.corner_coords
            xyz_vertices = []

            for c in coord_vertices:
                theta = 0.5 * np.pi - np.deg2rad(c[1])
                phi = np.deg2rad(c[0])

                xyz_vertices.append(hp.ang2vec(theta, phi))

            self.__corner_xyz = np.asarray(xyz_vertices)

        return self.__corner_xyz

    # Telgon_Shape properties
    @property
    def radius_proxy(self):
        return self.__radius_proxy

    @property  # Returns list of polygon
    def polygon(self):

        # if not self.__polygon:
        # 	tile_vertices = [[coord.ra.radian, coord.dec.radian] for coord in self.corner_coords]
        # 	self.__polygon = geometry.Polygon(tile_vertices)

        if not self.__polygon:
            tile_vertices = [[np.radians(coord_tup[0]), np.radians(coord_tup[1])] for coord_tup in self.corner_coords]
            self.__polygon = geometry.Polygon(tile_vertices)

        return [self.__polygon]

    # NOTE -- always assuming RA coordinate of the form [0, 360]
    @property
    def query_polygon(self):

        if not self.__query_polygon:
            self.__query_polygon = self.create_query_polygon(initial_poly_in_radian=True)
        return self.__query_polygon

    @property
    def query_polygon_string(self):

        if not self.__query_polygon_string:

            mp_str = "MULTIPOLYGON("
            multipolygon = []

            for p in self.query_polygon:

                mp = "(("
                ra_deg, dec_deg = zip(*[(coord_deg[0], coord_deg[1]) for coord_deg in p.exterior.coords])

                # For the SRS in the DB, we need to emulate lat,lon
                for i in range(len(ra_deg)):
                    mp += "%s %s," % (dec_deg[i], ra_deg[i] - 180.0)

                mp = mp[:-1]  # trim the last ","
                mp += ")),"
                multipolygon.append(mp)

            # Use the multipolygon string to create the WHERE clause
            multipolygon[-1] = multipolygon[-1][:-1]  # trim the last "," from the last object

            for mp in multipolygon:
                mp_str += mp
            mp_str += ")"

            self.__query_polygon_string = mp_str;

        return self.__query_polygon_string

    @property
    def enclosed_pixel_indices(self):

        if len(self.__enclosed_pixel_indices) == 0:

            # Start with the central pixel, in case the size of the FOV is <= the pixel size
            self.__enclosed_pixel_indices = np.asarray(
                [hp.ang2pix(self.nside, 0.5 * np.pi - self.dec_rad, self.ra_rad)])
            internal_pix = hp.query_polygon(self.nside, self.corner_xyz, inclusive=False)

            # However, if there is at least one pixel returned from query_polygon, use that array
            if len(internal_pix) > 0:
                self.__enclosed_pixel_indices = internal_pix

        return self.__enclosed_pixel_indices

    def plot(self, bmap, ax_to_plot, **kwargs):  # bmap,

        query_polygon = self.query_polygon
        for p in query_polygon:
            ra_deg, dec_deg = zip(*[(coord_deg[0], coord_deg[1])
                                    for coord_deg in p.exterior.coords])

            x2, y2 = bmap(ra_deg, dec_deg)
            lat_lons = np.vstack([x2, y2]).transpose()
            ax_to_plot.add_patch(Polygon(lat_lons, **kwargs))
