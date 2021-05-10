import numpy as np
from astropy import units as u
import astropy.coordinates as coord
from shapely import geometry
import healpy as hp
from matplotlib.patches import Polygon
import statistics
from shapely.ops import linemerge, unary_union, polygonize, split
from shapely.ops import transform as shapely_transform
from shapely.geometry import Point
from Teglon_Shape import *


class Tile(Teglon_Shape):
    def __init__(self, central_ra_deg, central_dec_deg, detector, nside, position_angle_deg=0.0, net_prob=0.0,
                 tile_id=None, tile_mwe=0.0):

        self.id = tile_id

        # Test
        self.dec_deg = central_dec_deg
        self.ra_deg = central_ra_deg
        self.dec_rad = np.radians(central_dec_deg)
        self.ra_rad = np.radians(central_ra_deg)
        self.position_angle_deg = position_angle_deg
        self.detector = detector

        self.nside = nside
        self.net_prob = net_prob
        self.mwe = tile_mwe
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
        # Used to check if this object is within this "radius proxy" of the coordinate signularity
        self.__radius_proxy = self.detector.radius_proxy
        self.__detector_multipolygon = self.detector.multipolygon
        self.__projected_multipolygon = None

        self.__query_polygon = None
        self.__query_polygon_string = None
        self.__enclosed_pixel_indices = np.array([])

        self.N128_pixel_index = None
        self.N128_pixel_id = None

    def __str__(self):
        return str(self.__dict__)

    # Telgon_Shape properties
    @property
    def radius_proxy(self):
        return self.__radius_proxy

    @property  # Returns list of polygon
    def multipolygon(self):
        return self.__detector_multipolygon

    @property  # Returns list of polygon
    def projected_multipolygon(self):

        if not self.__projected_multipolygon:
            self.__projected_multipolygon = self.project_vertices(self.ra_deg, self.dec_deg, self.position_angle_deg)
        return self.__projected_multipolygon

    # NOTE -- always assuming RA coordinate of the form [0, 360]
    @property
    def query_polygon(self):

        if not self.__query_polygon:
            self.__query_polygon = self.create_query_polygon(initial_poly_in_radian=False)
        return self.__query_polygon

    @property
    def query_polygon_string(self):

        if not self.__query_polygon_string:
            self.__query_polygon_string = self.create_query_polygon_string(initial_poly_in_radian=False)
        return self.__query_polygon_string

    @property
    def enclosed_pixel_indices(self):

        if len(self.__enclosed_pixel_indices) == 0:

            # Start with the central pixel, in case the size of the FOV is <= the pixel size
            self.__enclosed_pixel_indices = np.asarray(
                [hp.ang2pix(self.nside, 0.5 * np.pi - self.dec_rad, self.ra_rad)])

            xyz_polygons = []
            for i, geom in enumerate(self.projected_multipolygon):

                _xyz_vertices = []
                for coord_deg in geom.exterior.coords[:-1]: #-1
                    ra_deg = coord_deg[0]
                    dec_deg = coord_deg[1]
                    theta = 0.5 * np.pi - np.deg2rad(dec_deg)
                    phi = np.deg2rad(ra_deg)
                    _xyz_vertices.append(hp.ang2vec(theta, phi))

                xyz_polygons.append(np.asarray(_xyz_vertices))
            xyz_polygons = np.asarray(xyz_polygons)

            internal_pix = []
            for xyz in xyz_polygons:
                qp_pix = hp.query_polygon(self.nside, xyz, inclusive=False)
                for p in qp_pix:
                    if p not in internal_pix:
                        internal_pix.append(p)

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
