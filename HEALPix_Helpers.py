import numpy as np
import healpy as hp
from astropy import units as u
import astropy.coordinates as coord
from Contour import distance_metric_squared, rbf_interpolate
import math
from Tile import *
from astropy.coordinates import Distance
import healpy as hp
from astropy.cosmology import WMAP5, WMAP7, LambdaCDM
from shapely.geometry import Point
from matplotlib.patches import CirclePolygon
from shapely.ops import transform as shapely_transform
from ligo.skymap import distance
from scipy.special import erf
import copy
from Tile import *
from Pixel_Element import *
import pprint
import time
import math

import multiprocessing as mp
from scipy import spatial


class Detector:
    def __init__(self, detector_name, detector_width_deg, detector_height_deg, detector_radius_deg=None,
                 detector_id=None):
        self.id = detector_id
        self.name = detector_name
        self.deg_width = detector_width_deg
        self.deg_height = detector_height_deg
        self.deg_radius = detector_radius_deg
        self.area = None

    def __str__(self):
        return str(self.__dict__)


class Unpacked_Healpix:
    def __init__(self, file_name, prob, distmu, distsigma, distnorm, header, nside,
                 npix, area_per_px, linestyle, compute_contours=True, custom_percentile=None):

        self.file_name = file_name
        self.prob = prob
        self.distmu = distmu
        self.distsigma = distsigma
        self.distnorm = distnorm
        self.header = header
        self.nside = nside
        self.npix = npix
        self.area_per_px = area_per_px
        self.linestyle = linestyle

        self.indices_of_50 = None
        self.indices_of_70 = None
        self.indices_of_90 = None
        self.indices_of_95 = None

        self.levels = None
        self.X = None
        self.Y = None
        self.Z = None

        self.pixels_90 = None

        # This must be < 0.9
        self.custom_percentile = custom_percentile
        self.indices_of_custom = None
        self.pixels_custom = None

        print("Initializing '%s'...\n" % self.file_name)
        self.initialize(compute_contours)

    def initialize(self, compute_contours):

        t1 = time.time()

        sorted_prob = np.sort(self.prob)
        sorted_prob = sorted_prob[::-1]  # Reverse sort (highest first)
        max_prob = np.max(sorted_prob)

        # 99th percentile
        _99_cut, index_99 = self.get_prob_val_and_index(sorted_prob, 0.99)
        # 95th percentile
        _95_cut, index_95 = self.get_prob_val_and_index(sorted_prob, 0.95)
        # 90th percentile
        _90_cut, index_90 = self.get_prob_val_and_index(sorted_prob, 0.90)
        # 70th percentile
        _70_cut, index_70 = self.get_prob_val_and_index(sorted_prob, 0.70)
        # 50th percentile
        _50_cut, index_50 = self.get_prob_val_and_index(sorted_prob, 0.50)

        lvls = [_90_cut, _70_cut, _50_cut, max_prob]

        indices_of_50 = np.where(self.prob >= _50_cut)[0]
        indices_of_70 = np.where(self.prob >= _70_cut)[0]
        indices_of_90 = np.where(self.prob >= _90_cut)[0]
        indices_of_95 = np.where(self.prob >= _95_cut)[0]
        indices_of_99 = np.where(self.prob >= _99_cut)[0]

        self.indices_of_50 = indices_of_50
        self.indices_of_70 = indices_of_70
        self.indices_of_90 = indices_of_90
        self.indices_of_95 = indices_of_95
        self.indices_of_99 = indices_of_99

        self.levels = lvls
        self.pixels_90 = [Pixel_Element(i90, self.nside, self.prob[i90]) for i90 in self.indices_of_90]

        self.indices_of_custom = None

        # Custom percentile
        _custom_cut = None
        index_custom = None
        indices_of_custom = None
        if self.custom_percentile is not None:
            _custom_cut, index_custom = self.get_prob_val_and_index(sorted_prob, self.custom_percentile)
            indices_of_custom = np.where(self.prob >= _custom_cut)[0]
            self.indices_of_custom = indices_of_custom
            self.pixels_custom = [Pixel_Element(ic, self.nside, self.prob[ic]) for ic in self.indices_of_custom]

        if compute_contours:
            print("Computing contours for '%s'...\n" % self.file_name)
            self.compute_contours()

        t2 = time.time()

        print("\n********* start DEBUG ***********")
        print("`Unpacked_Healpix.initialize` execution time: %s" % (t2 - t1))
        print("********* end DEBUG ***********\n")

    def get_prob_val_and_index(self, sorted_prob, cum_cutoff):
        cum_prob = 0.0
        cutoff_value = 0.0  # prob at cum_cutoff
        index_of_cutoff = 0

        for i in range(len(sorted_prob)):
            cum_prob += sorted_prob[i]

            if (cutoff_value == 0.0 and cum_prob >= cum_cutoff):
                cutoff_value = sorted_prob[i]
                index_of_cutoff = i

        return (cutoff_value, index_of_cutoff)

    def compute_contours(self):

        # Get coords for plotting...
        theta, phi = hp.pix2ang(self.nside, self.indices_of_95)

        # Convert to RA, Dec.
        ra95 = coord.Angle(phi * u.rad)
        ra95 = ra95.wrap_at(180 * u.deg).degree
        dec95 = coord.Angle((0.5 * np.pi - theta) * u.rad)
        dec95 = dec95.degree

        self.max_ra_deg = np.max(ra95)
        self.min_ra_deg = np.min(ra95)
        self.max_dec_deg = np.max(dec95)
        self.min_dec_deg = np.min(dec95)

        # Compute contours
        # Interpolating 95% because contours themselves will be done at 90, 70, 50.
        prob95 = self.prob[self.indices_of_95]
        # n1 = 150
        # n2 = 150
        n1 = 100
        n2 = 100

        x = np.linspace(np.min(ra95), np.max(ra95), n1)
        y = np.linspace(np.min(dec95), np.max(dec95), n2)
        X, Y = np.meshgrid(x, y)
        # Z = rbf_interpolate(ra95, dec95, prob95, x, y, spacing_x=.07, spacing_y=.07)
        Z = rbf_interpolate(ra95, dec95, prob95, x, y, spacing_x=.03, spacing_y=.03)

        self.X = X
        self.Y = Y
        self.Z = Z


def invoke_enclosed_pix(tile):
    tile.enclosed_pixel_indices
    return tile


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class Cartographer:

    def downsample_map(detector, unpacked_healpix):

        t1 = time.time()

        # We wish to find a map resolution that is similar to the FOV size of a given telescope
        detector_area = detector.deg_width * detector.deg_height
        print("Detector area: %0.5f" % detector_area)

        # How many FOVs to tile the sky?
        _raw_npix = 41253.0 / detector_area
        print("Number of fields to tile the sky: %s" % _raw_npix)

        # Rescale to an acceptable number that plays with the healpix format...
        #     Npix = 12 * Nside**2
        _raw_nside = np.sqrt((_raw_npix / 12.0))

        # What power to raise 2 to get _raw_nside?
        _raw_power_2 = math.ceil(np.log10(_raw_nside) / np.log10(2))

        rescaled_nside = 2 ** _raw_power_2
        print("Rescaled nside for %s: %s" % (detector.name, rescaled_nside))

        # Convert this Nside to a rescaled Npix
        rescaled_npix = hp.nside2npix(rescaled_nside)
        print("New number of pix for %s: %s" % (detector.name, rescaled_npix))

        # area per pix
        rescaled_area_per_pix = 41253.0 / rescaled_npix
        print("New area per pix for %s: %s" % (detector.name, rescaled_area_per_pix))

        # rescale map...
        rescaled_prob, rescaled_distmu, rescaled_distsigma, rescaled_distnorm = hp.ud_grade([unpacked_healpix.prob,
                                                                                             unpacked_healpix.distmu,
                                                                                             unpacked_healpix.distsigma,
                                                                                             unpacked_healpix.distnorm],
                                                                                            nside_out=rescaled_nside,
                                                                                            order_in="RING",
                                                                                            order_out="RING")

        original_pix_per_rescaled_pix = unpacked_healpix.npix / rescaled_npix
        print("Original pix per rescaled pix for %s: %s" % (detector.name, original_pix_per_rescaled_pix))

        print("Renormalizing and initializing rescaled map...")
        rescaled_prob = rescaled_prob * original_pix_per_rescaled_pix

        rescaled_unpacked_healpix = Unpacked_Healpix("Rescaled_%s" % unpacked_healpix.file_name,
                                                     rescaled_prob,
                                                     rescaled_distmu,
                                                     rescaled_distsigma,
                                                     rescaled_distnorm,
                                                     unpacked_healpix.header,
                                                     rescaled_nside,
                                                     rescaled_npix,
                                                     rescaled_area_per_pix,
                                                     linestyle="-",
                                                     compute_contours=False,
                                                     custom_percentile=unpacked_healpix.custom_percentile)

        t2 = time.time()

        print("\n********* start DEBUG ***********")
        print("`downsample_map` execution time: %s" % (t2 - t1))
        print("********* end DEBUG ***********\n")

        return rescaled_unpacked_healpix

    def generate_all_sky_coords(detector):

        t1 = time.time()

        northern_limit = 89.99999
        southern_limit = -89.99999
        eastern_limit = 359.99999
        western_limit = 0.0

        # northern_limit = 89.99999
        # southern_limit = 60.0
        # eastern_limit = 359.99999
        # western_limit = 180.0

        dec_range = 179.99999
        ra_range = 359.99999

        frac_dec_tile, num_dec_tiles = math.modf(dec_range / detector.deg_height)

        total_dec_tiles = int(num_dec_tiles) + 1
        if num_dec_tiles == 0:
            num_dec_tiles = 1

        dec_differential = (detector.deg_height - (frac_dec_tile * detector.deg_height)) / num_dec_tiles

        dec_delta = detector.deg_height - dec_differential
        starting_dec = southern_limit + detector.deg_height / 2.0

        decs = []
        for i in range(total_dec_tiles):
            d = starting_dec + i * dec_delta
            decs.append(d)

        ras_over_decs = []
        for d in decs:

            adjusted_tile_width = detector.deg_width / np.abs(np.cos(np.radians(d)))

            frac_ra_tile, num_ra_tiles = math.modf(ra_range / adjusted_tile_width)
            total_ra_tiles = int(num_ra_tiles) + 1

            if num_ra_tiles == 0:
                num_ra_tiles = 1

            ra_differential = (adjusted_tile_width - (frac_ra_tile * adjusted_tile_width)) / num_ra_tiles
            ra_delta = adjusted_tile_width - ra_differential

            ras = []
            starting_ra = adjusted_tile_width / 2.0
            for i in range(total_ra_tiles):
                r = starting_ra + i * ra_delta
                ras.append(r)

            ras_over_decs.append(ras)

        print("All Sky statistics for %s..." % detector.name)
        print("\tNum of dec strips: %s" % len(decs))

        north_index = find_nearest(decs, northern_limit)
        south_index = find_nearest(decs, southern_limit)
        equator_index = find_nearest(decs, 0.0)

        print("\tNorthern most dec: %s" % decs[north_index])
        print("\tSouthern most dec: %s" % decs[south_index])

        east_index = find_nearest(ras_over_decs[equator_index], eastern_limit)
        west_index = find_nearest(ras_over_decs[equator_index], western_limit)

        print("\tEastern most ra: %s" % ras_over_decs[equator_index][east_index])
        print("\tWestern most ra: %s" % ras_over_decs[equator_index][west_index])

        print("\tNum of ra tiles in northern most dec slice: %s" % len(ras_over_decs[north_index]))
        print("\tNum of ra tiles at celestial equator: %s" % len(ras_over_decs[equator_index]))

        print("Constructing grid of coordinates...")
        RA = []
        DEC = []
        for i, d in enumerate(decs):
            for ra in ras_over_decs[i]:
                RA.append(ra)
                DEC.append(d)

        all_sky_coords = list(zip(RA, DEC))
        print("Total coords for %s: %s" % (detector.name, len(all_sky_coords)))

        print("\n******\n")

        t2 = time.time()

        print("\n********* start DEBUG ***********")
        print("`generate_all_sky_coords` execution time: %s" % (t2 - t1))
        print("********* end DEBUG ***********\n")

        return all_sky_coords

    def assign_galaxy_relative_prob(unpacked_healpix, galaxies, cum_prob_in_tiles, completeness, distance_override=0.0,
                                    stddev_override=0.0):

        t1 = time.time()

        # Get probability to distribute to galaxies
        galaxy_cum_prob = completeness * cum_prob_in_tiles

        # Get normalization for galaxy luminosity proxy
        total_galaxy_lum = np.sum([g.B_lum_proxy for g in galaxies])

        galaxy_pixel_indices = [g.pixel_index for g in galaxies]
        galaxy_distances = [g.z_dist for g in galaxies]
        galaxy_distance_errs = [g.z_dist_err for g in galaxies]
        galaxy_lums = [g.B_lum_proxy for g in galaxies]

        _prob = unpacked_healpix.prob[galaxy_pixel_indices]
        _distmu = unpacked_healpix.distmu[galaxy_pixel_indices]
        _distsigma = unpacked_healpix.distsigma[galaxy_pixel_indices]
        _distnorm = unpacked_healpix.distnorm[galaxy_pixel_indices]
        _mean, _stddev, _norm = distance.parameters_to_moments(_distmu, _distsigma)

        numerator = np.subtract(galaxy_distances, _mean)
        denominator = np.add(np.power(_stddev, 2), np.power(galaxy_distance_errs, 2))

        sigmaTotal = np.abs(numerator) / np.sqrt(denominator)
        z_prob = 1.0 - erf(sigmaTotal)
        three_d_prob = z_prob * _prob

        lum_prob = galaxy_lums / total_galaxy_lum
        four_d_prob = lum_prob * three_d_prob
        four_d_prob = (four_d_prob / np.sum(four_d_prob)) * galaxy_cum_prob  # normalize

        for i, g in enumerate(galaxies):
            g.relative_prob = four_d_prob[i]

        t2 = time.time()

        print("\n********* start DEBUG ***********")
        print("`assign_galaxy_relative_prob` execution time: %s" % (t2 - t1))
        print("********* end DEBUG ***********\n")

    def redistribute_probability(unpacked_healpix, galaxies, tiles, completeness):

        t1 = time.time()

        # Copy probability
        redistributed_prob = copy.deepcopy(unpacked_healpix.prob)

        # Compliment of completeness stays in the pixels
        rescale_factor = 1.0 - completeness
        print("Rescale factor: %s" % rescale_factor)

        # Get unique list of pixels enclosed pix in tiles...
        all_pix = []
        for t in tiles:
            all_pix += list(t.enclosed_pixel_indices)
        unique_pix = np.asarray(list(set(all_pix)))

        # Rescale those pixels...
        redistributed_prob[unique_pix] = redistributed_prob[unique_pix] * rescale_factor

        # Distribute each galaxy's probability over the number of pixels bounded by the galaxy radius
        # These fractional probabilities get added on top of the field probability in the pixels
        for g in galaxies:
            redistributed_prob[np.asarray(g.enclosed_pix)] += g.relative_prob / len(g.enclosed_pix)

        t2 = time.time()

        print("\n********* start DEBUG ***********")
        print("`redistribute_probability` execution time: %s" % (t2 - t1))
        print("********* end DEBUG ***********\n")

        return redistributed_prob, unique_pix

    def generate_tiles(unpacked_healpix, rescaled_pixels_90, rescale_detector, all_sky_coords, fudge_factor=0.75):

        t1 = time.time()

        good_tiles = []

        # Extract pixel coords as tuples, then sort coords by RA, then Dec
        pix_coords = [(p.coord.ra.degree, p.coord.dec.degree) for p in rescaled_pixels_90]
        sorted_pix_coords = sorted(pix_coords, key=lambda p: (p[0], p[1]))

        # Construct 2D binary tree and define nearest-neighbor threshold between pixel/tile
        tree = spatial.cKDTree(sorted_pix_coords)
        threshold = np.sqrt(rescale_detector.deg_height ** 2 + rescale_detector.deg_width ** 2)

        # query tree and extract indices of tiles within threshold
        r = tree.query(all_sky_coords)
        good_indices = np.where(r[0] <= threshold)[0]

        # extract coords and create tiles
        all_sky_coords_arr = np.asarray(all_sky_coords)
        good_coords = all_sky_coords_arr[good_indices]

        # coord.SkyCoord(gc[0], gc[1], unit=(u.deg,u.deg))
        for gc in good_coords:
            t = Tile(gc[0], gc[1], rescale_detector.deg_width, rescale_detector.deg_height, unpacked_healpix.nside)
            good_tiles.append(t)

        t2 = time.time()
        print("\n********* start DEBUG ***********")
        print("`generate_tiles.good_tiles` execution time: %s" % (t2 - t1))
        print("********* end DEBUG ***********\n")

        t1 = time.time()
        # Parallelize initialization of good tiles
        pool = mp.Pool()
        initialized_tiles = pool.map(invoke_enclosed_pix, good_tiles)

        cum_prob = 0.0
        for t in initialized_tiles:
            t.net_prob = np.sum(unpacked_healpix.prob[t.enclosed_pixel_indices])
            cum_prob += t.net_prob

        t2 = time.time()
        print("\n********* start DEBUG ***********")
        print("`generate_tiles.enclosed_pixel_indices` execution time: %s" % (t2 - t1))
        print("********* end DEBUG ***********\n")

        # return cum_prob, good_tiles
        return cum_prob, initialized_tiles

    def assign_tiles(self, input_tiles):

        t1 = time.time()

        for t in input_tiles:
            t.net_prob = np.sum(self.unpacked_healpix.prob[t.enclosed_pixel_indices])

        sorted_tiles = sorted(input_tiles, key=lambda x: x.net_prob, reverse=True)

        cum_prob = 0.0
        tile_sub_set = []
        i = 0

        prob_threshold = self.custom_percentile if self.is_custom else 0.90

        while cum_prob <= prob_threshold and i < len(sorted_tiles):
            tile_sub_set.append(sorted_tiles[i])
            cum_prob += sorted_tiles[i].net_prob
            i += 1

        self.tiles = tile_sub_set
        self.cumlative_prob_in_tiles = cum_prob
        print("Total tiles for %s in `%s`: %s" % (
        self.rescale_detector.name, self.unpacked_healpix.file_name, len(self.tiles)))
        print(
            "Sanity check: Cumulative prob in tiles based on original pixel resolution close to %s? Cumulate Prob = %s" % (
            self.custom_percentile,
            self.cumlative_prob_in_tiles))

        t2 = time.time()
        print("\n********* start DEBUG ***********")
        print("`generate_tiles.assign_tiles` execution time: %s" % (t2 - t1))
        print("********* end DEBUG ***********\n")

    def __init__(self, gwid, unpacked_healpix, rescale_detector, all_sky_coords, generate_tiles=True,
                 downsample_map=True):

        self.gwid = gwid
        self.unpacked_healpix = unpacked_healpix
        self.rescale_detector = rescale_detector
        self.all_sky_coords = all_sky_coords
        self.tiles = None
        self.cumlative_prob_in_tiles = 0.0

        self.is_custom = False
        self.custom_percentile = None

        if self.unpacked_healpix.custom_percentile is not None:
            self.is_custom = True
            self.custom_percentile = self.unpacked_healpix.custom_percentile

        rescaled_healpix = None

        if downsample_map:
            print("Rescaling %s for %s" % (unpacked_healpix.file_name, rescale_detector.name))
            rescaled_healpix = Cartographer.downsample_map(self.rescale_detector, self.unpacked_healpix)

            # Debug
            if self.is_custom:
                total_prob_rescaled_pixels_custom = np.sum([p.prob for p in rescaled_healpix.pixels_custom])
                print("Sanity check: Sum of probability in %s percentile, rescaled pix? Sum = %s" % (
                rescaled_healpix.custom_percentile,
                total_prob_rescaled_pixels_custom))
            else:
                total_prob_rescaled_pixels_90 = np.sum([p.prob for p in rescaled_healpix.pixels_90])
                print(
                    "Sanity check: Sum of probability in 90th percentile, rescaled pix? Sum = %s" % total_prob_rescaled_pixels_90)
        else:
            print("Not rescaling map...")

        if generate_tiles:
            print("Using all sky coords + rescaled pixels to generate tile set...")

            pixels_to_compute = None
            if self.is_custom:
                pixels_to_compute = rescaled_healpix.pixels_custom
            else:
                pixels_to_compute = rescaled_healpix.pixels_90

            cumlative_prob_in_tiles, tiles = Cartographer.generate_tiles(self.unpacked_healpix,
                                                                         pixels_to_compute,
                                                                         self.rescale_detector,
                                                                         self.all_sky_coords)

            self.tiles = tiles
            self.cumlative_prob_in_tiles = cumlative_prob_in_tiles

            print("Total tiles for %s in `%s`: %s" % (
            self.rescale_detector.name, self.unpacked_healpix.file_name, len(self.tiles)))

            if self.is_custom:
                print(
                    "Sanity check: Cumulative prob in tiles based on non-scaled pixels close to %s? Cumulate Prob = %s" % (
                    self.unpacked_healpix.custom_percentile,
                    self.cumlative_prob_in_tiles))
            else:
                print(
                    "Sanity check: Cumulative prob in tiles based on non-scaled pixels close to 0.90? Cumulate Prob = %s" % self.cumlative_prob_in_tiles)
        else:
            print("Not processing tiles...")


class glade_galaxy:

    def __init__(self, db_result, unpacked_healpix, cosmo, distance_cutoff, proper_radius_kpc):
        self.ID = int(db_result[0])
        self.Galaxy_id = int(db_result[1])
        self.Distance_id = int(db_result[2])
        self.PGC = db_result[3]
        self.Name_GWGC = db_result[4]
        self.Name_HyperLEDA = db_result[5]
        self.Name_2MASS = db_result[6]
        self.Name_SDSS_DR12 = db_result[7]
        self.RA = float(db_result[8]) if db_result[8] is not None else db_result[8]

        # Hack: convert the RA of the galaxy from [0,360] to [-180,180]
        g_ra = self.RA
        if g_ra > 180:
            g_ra = self.RA - 360.
        self.RA = g_ra

        self.Dec = float(db_result[9]) if db_result[9] is not None else db_result[9]
        # db_result[10] == MySQL POINT object...
        self.dist = float(db_result[11]) if db_result[11] is not None else db_result[11]

        theta = 0.5 * np.pi - np.deg2rad(self.Dec)
        phi = np.deg2rad(self.RA)
        self.pixel_index = hp.ang2pix(unpacked_healpix.nside, theta, phi)

        self.dist_err = float(db_result[12]) if db_result[12] is not None else db_result[12]
        self.z_dist = float(db_result[13]) if db_result[13] is not None else db_result[13]
        self.z_dist_err = float(db_result[14]) if db_result[14] is not None else db_result[14]
        self.z = float(db_result[15]) if db_result[15] is not None else db_result[15]

        self.enclosed_pix = []
        self.polygon_coords = []

        # Compute cosmological parameters:
        # if self.z_dist <= distance_cutoff:

        # 	try:

        # 		angular_radius_deg = proper_radius_kpc*(cosmo.arcsec_per_kpc_proper(self.z)).value/3600.
        # 		c1 = Point(self.RA, self.Dec).buffer(angular_radius_deg)
        # 		c2 = shapely_transform(lambda x,y,z=None: ((self.RA - (self.RA - x)/np.abs(np.cos(np.radians(y)))),
        # 												   y), c1)
        # 		self.polygon_coords = c2.exterior.coords

        # 		# Get all enclosed pix
        # 		xyz_vertices = []
        # 		for c in self.polygon_coords:

        # 			theta = (0.5 * np.pi - np.deg2rad(c[1])) % np.pi
        # 			phi = np.deg2rad(c[0]) % (2.0*np.pi)

        # 			xyz_vertices.append(hp.ang2vec(theta, phi))

        # 		internal_pix = hp.query_polygon(unpacked_healpix.nside, xyz_vertices[0:-2], inclusive=False)
        # 		self.enclosed_pix = list(internal_pix)

        # 		print("z dist: %s; # of enclosed: %s" % (self.z_dist, len(self.enclosed_pix)))
        # 	except:
        # 		print("Theta too large!")
        # 	finally:
        # 		self.enclosed_pix.append(self.pixel_index)

        # else:
        # 	self.enclosed_pix.append(self.pixel_index)
        self.enclosed_pix.append(self.pixel_index)

        self.B = float(db_result[16]) if db_result[16] is not None else db_result[16]
        self.B_err = float(db_result[17]) if db_result[17] is not None else db_result[17]
        self.B_abs = float(db_result[18]) if db_result[18] is not None else db_result[18]
        self.J = float(db_result[19]) if db_result[19] is not None else db_result[19]
        self.J_err = float(db_result[20]) if db_result[20] is not None else db_result[20]
        self.H = float(db_result[21]) if db_result[21] is not None else db_result[21]
        self.H_err = float(db_result[22]) if db_result[22] is not None else db_result[22]
        self.K = float(db_result[23]) if db_result[23] is not None else db_result[23]
        self.K_err = float(db_result[24]) if db_result[24] is not None else db_result[24]
        self.flag1 = db_result[25]
        self.flag2 = db_result[26]
        self.flag3 = db_result[27]
        self.B_lum_proxy = (self.z_dist ** 2) * 10 ** (-0.4 * self.B)
        self.relative_prob = 0.0
        self.galaxy_glade_completeness = -999

    def plot(self, bmap, ax_to_plot, unpacked_healpix, **kwargs):

        if len(self.polygon_coords) > 0:

            cra_deg, cdec_deg = zip(*[(coord_deg[0], coord_deg[1])
                                      for coord_deg in self.polygon_coords])

            cx, cy = bmap(cra_deg, cdec_deg)
            clat_lons = np.vstack([cx, cy]).transpose()
            ax_to_plot.add_patch(Polygon(clat_lons, **kwargs))

        else:

            if self.relative_prob < 1e-8:
                fc = 'k'
                ec = 'None'
                alph = 0.1
            else:
                fc = 'None'
                ec = 'k'
                alph = 1.0

            c = coord.SkyCoord(self.RA, self.Dec, unit=(u.deg, u.deg))
            pe = Pixel_Element(self.pixel_index, unpacked_healpix.nside, unpacked_healpix.prob[self.pixel_index])
            pe.plot(bmap, ax_to_plot, facecolor=fc, edgecolor=ec, linewidth=0.1, alpha=alph, zorder=9500)

    def __repr__(self):
        return pprint.pformat(vars(self), indent=4, width=1)


def average_distance_prior(original_healpix_object):
    _distsigma = original_healpix_object.distsigma[original_healpix_object.indices_of_90]
    _distmu = original_healpix_object.distmu[original_healpix_object.indices_of_90]

    _mean, _stddev, _norm = distance.parameters_to_moments(_distmu, _distsigma)

    # for some reason inf is returned in some cases. Protect against this.
    _mean[np.where(_mean == float("inf"))] = 0.0

    avg_ligo_dist = np.mean(_mean)
    print("Average LIGO distance from 90th percentile pixels: %s [Mpc]" % avg_ligo_dist)

    return avg_ligo_dist


def GLADE_completeness(average_dist):
    gladedist = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0,
                 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0,
                 310.0, 320.0, 330.0, 340.0, 350.0, 360.0, 370.0, 380.0, 390.0, 400.0, 410.0, 420.0, 430.0, 440.0,
                 450.0, 460.0, 470.0, 480.0, 490.0, 500.0, 510.0, 520.0, 530.0, 540.0, 550.0, 560.0, 570.0, 580.0,
                 590.0, 600.0, 610.0, 620.0, 630.0, 640.0, 650.0, 660.0, 670.0, 680.0, 690.0, 700.0, 710.0, 720.0,
                 730.0, 740.0, 750.0, 760.0, 770.0, 780.0, 790.0, 800.0, 810.0, 820.0, 830.0, 840.0, 850.0, 860.0,
                 870.0, 880.0, 890.0, 900.0, 910.0, 920.0, 930.0, 940.0, 950.0, 960.0, 970.0, 980.0, 990.0, 1000.0,
                 1001.0, 100000.0]
    Bcompleteness = [1.0, 1.0, 1.0, 0.7749892841772487, 0.6054639215026809, 0.59173277302711, 0.7266966899020528,
                     0.7732837192608969, 0.6752081862421173, 0.5582284987571347, 0.5072883157932683, 0.507895400971434,
                     0.532564844819251, 0.5346458909474606, 0.49763919176390864, 0.4844803921814651,
                     0.46788211167218285, 0.4790468050336609, 0.47062778069493116, 0.46528164771775443,
                     0.48097548620982633, 0.4770764886036427, 0.4615442814318801, 0.4654411855172319,
                     0.45638735214830123, 0.45896755648348087, 0.44570126474071337, 0.43575958494159683,
                     0.42076921796612926, 0.4096135559209188, 0.4018645285590103, 0.3836391534374383,
                     0.3696779536992611, 0.3531367226887272, 0.34395543462887035, 0.32368095165817073,
                     0.31130299636761943, 0.28456584973708926, 0.2757327838151936, 0.25803597790355887,
                     0.2383740922467115, 0.22277088371369333, 0.21605968917207002, 0.19571909670889368,
                     0.18894471787283798, 0.18585907630049628, 0.17235941185066928, 0.1598199287433832,
                     0.1661227637862408, 0.1472864698839269, 0.13624967135140292, 0.1449490820279948,
                     0.1323241779478645, 0.12442479349307807, 0.11714837383271669, 0.10998141176488553,
                     0.09840759957389901, 0.10096610825633844, 0.09908869020348751, 0.09609979427193585,
                     0.08461436057525955, 0.09070816472849053, 0.08039981615931603, 0.08246321358113154,
                     0.07230814873142391, 0.06870936227258237, 0.07150501861441742, 0.06004126469998274,
                     0.057540983887985515, 0.05722673740665967, 0.05457968955211691, 0.0536624514349561,
                     0.05057311805622857, 0.04695179045463761, 0.0452931997407635, 0.04283259348744712,
                     0.0413562585135292, 0.040197035707962056, 0.03297981378287233, 0.035498974083870115,
                     0.03461049154738816, 0.02819010591325043, 0.033423422715134286, 0.03226151713308256,
                     0.023573785030586655, 0.029575322229746057, 0.023104048125228465, 0.02678826949448462,
                     0.022270778724629483, 0.0250608344480609, 0.018239068476361663, 0.01994100716722564,
                     0.021739495739457312, 0.01628078391183763, 0.01729358411494358, 0.02010845461972772,
                     0.014551753571501758, 0.015017331234211615, 0.014345060987522267, 0.01655436428767106, 0.0, 0.0]

    return (np.interp(average_dist, gladedist, Bcompleteness))
