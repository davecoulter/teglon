import numpy as np
import healpy as hp
from astropy import units as u
import astropy.coordinates as coord
from shapely import geometry
from matplotlib.patches import Polygon
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors

class Pixel_Element:
    def __init__(self,index,nside,prob):
        
        self.tile_references = []
        
        # Properties
        self.__index = index
        self.__nside = nside
        self.__coord = None
        self.__polygon = None
        self.__prob = prob

        self.__epi = np.array([])
        
    
    @property
    def index(self):
        return self.__index
    
    @property
    def nside(self):
        return self.__nside
    
    @property
    def prob(self):
        return self.__prob
        
    @property
    def coord(self):
        if not self.__coord:
            theta, phi = hp.pix2ang(self.nside, self.index)
            pix_coord = coord.SkyCoord(ra=np.rad2deg(phi), 
                                       dec=np.rad2deg(0.5 * np.pi - theta), 
                                       unit=(u.deg,u.deg))
            self.__coord = pix_coord
        
        return self.__coord
    
    @property
    def polygon(self):
        
        if not self.__polygon:
            pixel_xyz_vertices = hp.boundaries(self.nside, pix=self.index)
            theta, phi = hp.vec2ang(np.transpose(pixel_xyz_vertices))

            dec_rad = (np.pi/2. - theta)
            ra_rad = phi*np.cos(dec_rad)

            pixel_radian_vertices = [[ra,dec] for ra,dec in zip(ra_rad, dec_rad)]
            self.__polygon = geometry.Polygon(pixel_radian_vertices)
            
        return self.__polygon


    def enclosed_pixel_indices(self, nside_out):

        # Sanity
        if nside_out < self.nside:
            raise("Can't get enclosed pixel indices for lower resolution pixels!")
        
        if len(self.__epi) == 0:
            pixel_xyz_vertices = hp.boundaries(self.nside, pix=self.index)
            internal_pix = hp.query_polygon(nside_out, pixel_xyz_vertices, inclusive=False)
            self.__epi = internal_pix
            
        return self.__epi


    
    def update_polygon(self, polygon):
        self.__polygon = polygon
    
    
    def get_patch(self, bmap):
        ra_deg,dec_deg = zip(*[(np.degrees(coord_rad[0]/np.cos(coord_rad[1])), np.degrees(coord_rad[1])) 
                               for coord_rad in self.polygon.exterior.coords])

        # ra_deg_shifted = bmap.shiftdata(ra_deg,lon_0=180.0)

        x2,y2 = bmap(ra_deg,dec_deg)

        lat_lons = np.vstack([x2,y2]).transpose()
        patch = Polygon(lat_lons)

        return patch


    def plot(self, bmap, ax_to_plot, **kwargs): #value, 
        
        # Plot central point
#         c = self.coord
#         x1,y1 = bmap(c.ra.degree, c.dec.degree)
#         bmap.plot(x1, y1, 'b.',markersize=5)

        
        ra_deg,dec_deg = zip(*[(np.degrees(coord_rad[0]/np.cos(coord_rad[1])), np.degrees(coord_rad[1])) 
                               for coord_rad in self.polygon.exterior.coords])

        x2,y2 = bmap(ra_deg,dec_deg)
        lat_lons = np.vstack([x2,y2]).transpose()

        # patches = [Polygon(lat_lons)]
        # patch_collection = mpl.collections.PatchCollection(patches, cmap=plt.cm.viridis, 
        #     norm=colors.LogNorm(), 
        #     **kwargs)
        # patch_collection.set_array([value])
        # ax_to_plot.add_collection(patch_collection)
        ax_to_plot.add_patch(Polygon(lat_lons, **kwargs))

