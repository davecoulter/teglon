import numpy as np
from astropy import units as u
import astropy.coordinates as coord
from shapely import geometry
import healpy as hp
from matplotlib.patches import Polygon
import statistics
from shapely.ops import linemerge, unary_union, polygonize, split
from abc import ABCMeta, abstractmethod, abstractproperty
from Teglon_Shape import *


class SQL_Polygon(Teglon_Shape):

    def __init__(self, polygon):

        self.__polygon = polygon
        self.__radius_proxy = None
        self.__query_polygon = None
        self.__query_polygon_string = None

    # Telgon_Shape properties
    @property
    def radius_proxy(self):
        return self.__radius_proxy

    @property
    def multipolygon(self):
        return self.__polygon

    @property
    def projected_multipolygon(self):
        return self.__polygon

    @property
    def query_polygon(self):

        if not self.__query_polygon:
            self.__query_polygon = self.create_query_polygon(initial_poly_in_radian=False)
        return self.__query_polygon

    # Formatted for MySQL WGS84 spatial reference system
    @property
    def query_polygon_string(self):

        if not self.__query_polygon_string:
            self.self.__query_polygon_string = self.create_query_polygon_string(initial_poly_in_radian=False)
        return self.__query_polygon_string

    def plot(self, bmap, ax_to_plot, **kwargs):  # bmap,

        query_polygon = self.query_polygon
        for p in query_polygon:
            ra_deg, dec_deg = zip(*[(coord_deg[0], coord_deg[1])
                                    for coord_deg in p.exterior.coords])

            x2, y2 = bmap(ra_deg, dec_deg)
            lat_lons = np.vstack([x2, y2]).transpose()
            ax_to_plot.add_patch(Polygon(lat_lons, **kwargs))
