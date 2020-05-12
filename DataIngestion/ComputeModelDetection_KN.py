# region imports
import os
import sys
sys.path.append('../')
import pickle
from scipy.interpolate import interp1d
from HEALPix_Helpers import *
from Tile import *
from Pixel_Element import *
from collections import OrderedDict
from astropy.table import Table
from astropy.time import Time
import pytz
# endregion

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

# Multiprocessing methods...
def initial_z(pixel_data):
    cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

    pix_index = pixel_data[0]
    mean_dist = pixel_data[1]

    d = Distance(mean_dist, u.Mpc)
    z = d.compute_z(cosmology=cosmo)

    return (pix_index, z)

def integrate_pixel(pixel_data):

    reverse_band_mapping_new = {
        "SDSS g": "sdss_g",
        "SDSS r": "sdss_r",
        "SDSS i": "sdss_i",
        "Clear": "Clear"
    }

    pix_key = pixel_data[0]
    pix_index = pix_key[0]
    model_key = pix_key[1]

    mean_dist = pixel_data[1]
    dist_sigma = pixel_data[2]
    mwe = pixel_data[3]
    prob_2D = pixel_data[4]
    band = pixel_data[5]
    mjd = pixel_data[6]
    lim_mag = pixel_data[7]
    model_func_dict = pixel_data[8]
    delta_mjd = pixel_data[9]

    f_band = model_func_dict[reverse_band_mapping_new[band]]
    abs_in_band = f_band(delta_mjd)

    # compute distance upper bound, given the limiting magnitude:
    pwr = (lim_mag - abs_in_band + 5.0 - mwe) / 5.0 - 6.0
    d_upper = 10 ** pwr
    prob_to_detect = 0.0

    if d_upper > 0.0:
        # cdf = lambda d: norm.cdf(d, mean_dist, dist_sigma)
        # dist_norm = cdf(np.inf) - cdf(0.0) # truncating integral at d = 0 Mpc
        # prob_to_detect = (prob_2D/dist_norm) * (cdf(d_upper) - cdf(0.0))

        prob_to_detect = prob_2D * (0.5 * erf((d_upper - mean_dist) / (np.sqrt(2) * dist_sigma)) - \
                         0.5 * erf(-mean_dist / (np.sqrt(2) * dist_sigma)))

    new_pix_key = (pix_index, model_key, band, mjd)
    return (new_pix_key, prob_to_detect)

class Teglon:

    def add_options(self, parser=None, usage=None, config=None):
        import optparse
        if parser == None:
            parser = optparse.OptionParser(usage=usage, conflict_handler="resolve")

        parser.add_option('--gw_id', default="", type="str", help='LIGO superevent name, e.g. `S190425z`.')

        parser.add_option('--healpix_file', default="", type="str",
                          help='Healpix filename. Used with `gw_id` to identify unique map.')

        parser.add_option('--healpix_dir', default='../Events/{GWID}', type="str",
                          help='Directory for where to look for the healpix file.')

        parser.add_option('--model_output_dir', default="../Events/{GWID}/ModelDetection", type="str",
                          help='Directory for where to output processed models.')

        parser.add_option('--model_base_dir', default="", type="str",
                          help='Base directory for models.')

        parser.add_option('--num_cpu', default="6", type="int",
                          help='Number of CPUs to use for multiprocessing')

        parser.add_option('--sub_dir', default="", type="str",
                          help='GRB Model sub directory (for batching)')

        parser.add_option('--merger_time_MJD', default="58709.882824224536", type="float",
                          help='''Time of the merger in MJD. This is used to compute where on the light curve we are. 
                          Default to GW190814''')

        return (parser)

    def main(self):
        print("Processes to use: %s" % self.options.num_cpu)

        # region Sanity/Parameter checks
        prep_start = time.time()

        MERGER_MJD = 58709.882824224536
        GWID = "S190814bv"
        HEALPIX_FILE = "LALInference.v1.fits.gz,0"

        t = Time(MERGER_MJD, format='mjd')
        print("** Running models for MJD: %s; UTC: %s **" % (MERGER_MJD, t.to_datetime(timezone=pytz.utc)))


        # FORMATTED_HEALPIX_DIR = "../Events/%s" % GWID
        # FORMATTED_MODEL_OUTPUT_DIR = "../Events/%s/ModelDetection" % GWID
        FORMATTED_HEALPIX_DIR = "./"
        FORMATTED_MODEL_OUTPUT_DIR = "./ModelDetection"

        # HPX_PATH = "%s/%s" % (FORMATTED_HEALPIX_DIR, HEALPIX_FILE)
        HPX_PATH =  "./%s" % (HEALPIX_FILE)

        KN_MODEL_PATH = "%s/%s" % (self.options.model_base_dir, self.options.sub_dir)

        kn_model_files = []
        for file_index, file in enumerate(os.listdir(KN_MODEL_PATH)):
            if file.endswith(".dat"):
                kn_model_files.append("%s/%s" % (KN_MODEL_PATH, file))
        # endregion

        # region Convienence Dictionaries
        # Band abbreviation, band_id mapping
        band_mapping_new = {
            "sdss_g": "SDSS g",
            "sdss_r": "SDSS r",
            "sdss_i": "SDSS i",
            "Clear": "Clear"
        }

        reverse_band_mapping_new = {
            "SDSS g": "sdss_g",
            "SDSS r": "sdss_r",
            "SDSS i": "sdss_i",
            "Clear": "Clear"
        }

        detector_mapping = {
            "s": "SWOPE",
            "t": "THACHER",
            "a": "ANDICAM",
            "n": "NICKEL",
            "m": "MOSFIRE",
            "k": "KAIT",
            "si": "SINISTRO"
        }
        # endregion

        # region Load Serialized Sky Pixels, EBV, and Models from Disk
        # LOADING NSIDE 128 SKY PIXELS AND EBV INFORMATION
        print("\nLoading NSIDE 128 pixels...")
        nside128 = 128
        N128_dict = None
        with open('N128_dict.pkl', 'rb') as handle:
            N128_dict = pickle.load(handle)
        del handle

        print("\nLoading existing EBV...")
        ebv = None
        with open('ebv.pkl', 'rb') as handle:
            ebv = pickle.load(handle)

        models = {}
        for index, mf in enumerate(kn_model_files[0:100]):
            model_table = Table.read(mf, format='ascii.ecsv')
            mask = model_table['sdss_g'] != np.nan

            model_time = np.asarray(model_table['time'][mask])
            g = np.asarray(model_table['sdss_g'][mask])
            r = np.asarray(model_table['sdss_r'][mask])
            i = np.asarray(model_table['sdss_i'][mask])
            clear = np.asarray(model_table['Clear'][mask])

            model_props = model_table.meta['comment']
            vej = float(model_props[0].split("=")[1].strip())
            mej = float(model_props[1].split("=")[1].strip())
            ye = float(model_props[2].split("=")[1].strip())

            base_name = os.path.basename(mf)
            print("Loading `%s`" % base_name)

            # Get interpolation function for each Light Curve
            f_g = interp1d(model_time, g, fill_value="extrapolate")
            f_r = interp1d(model_time, r, fill_value="extrapolate")
            f_i = interp1d(model_time, i, fill_value="extrapolate")
            f_clear = interp1d(model_time, clear, fill_value="extrapolate")

            models[(vej, mej, ye)] = {
                'sdss_g': f_g,
                'sdss_r': f_r,
                'sdss_i': f_i,
                'Clear': f_clear
            }
        # endregion

        # region Get Map, Bands and initialize pixels.
        healpix_map_id = 2
        healpix_map_nside = 1024

        band_dict_by_name = {
            "SDSS u": (1, "SDSS u", 4.239),
            "SDSS g": (2, "SDSS g", 3.303),
            "SDSS r": (3, "SDSS r", 2.285),
            "SDSS i": (4, "SDSS i", 1.698),
            "SDSS z": (5, "SDSS z", 1.263),
            "Landolt B": (6, "Landolt B", 3.626),
            "Landolt V": (7, "Landolt V", 2.742),
            "Landolt R": (8, "Landolt R", 2.169),
            "Landolt I": (9, "Landolt I", 1.505),
            "UKIRT J": (10, "UKIRT J", 0.709),
            "UKIRT H": (11, "UKIRT H", 0.449),
            "UKIRT K": (12, "UKIRT K", 0.302),
            "Clear": (13, "Clear", 0.91)
        }
        band_dict_by_id = {
            1: (1, "SDSS u", 4.239),
            2: (2, "SDSS g", 3.303),
            3: (3, "SDSS r", 2.285),
            4: (4, "SDSS i", 1.698),
            5: (5, "SDSS z", 1.263),
            6: (6, "Landolt B", 3.626),
            7: (7, "Landolt V", 2.742),
            8: (8, "Landolt R", 2.169),
            9: (9, "Landolt I", 1.505),
            10: (10, "UKIRT J", 0.709),
            11: (11, "UKIRT H", 0.449),
            12: (12, "UKIRT K", 0.302),
            13: (13, "Clear", 0.91)
        }

        print("\nRetrieving distinct, imaged map pixels")
        # map_pixel_select = '''
        # SELECT
        #     DISTINCT hp.id,
        #     hp.HealpixMap_id,
        #     hp.Pixel_Index,
        #     hp.Prob,
        #     hp.Distmu,
        #     hp.Distsigma,
        #     hp.Distnorm,
        #     hp.Mean,
        #     hp.Stddev,
        #     hp.Norm,
        #     sp.Pixel_Index as N128_Pixel_Index
        # FROM
        #     HealpixPixel hp
        # JOIN ObservedTile_HealpixPixel ot_hp on ot_hp.HealpixPixel_id = hp.id
        # JOIN ObservedTile ot on ot.id = ot_hp.ObservedTile_id
        # JOIN SkyPixel sp on sp.id = hp.N128_SkyPixel_id
        # WHERE
        #     ot.HealpixMap_id = %s and
        #     ot.Mag_Lim IS NOT NULL
        # '''
        # q = map_pixel_select % healpix_map_id
        # map_pixels = query_db([q])[0]
        # with open("%s/%s" % (FORMATTED_HEALPIX_DIR, 'map_pix.pkl'), 'wb') as handle:
        #     pickle.dump(map_pixels, handle, protocol=pickle.HIGHEST_PROTOCOL)

        map_pixels = None
        with open("%s/%s" % (FORMATTED_HEALPIX_DIR, 'map_pix.pkl'), 'rb') as handle:
            map_pixels = pickle.load(handle)
        print("Retrieved %s map pixels..." % len(map_pixels))

        # Initialize map pix dict for later access
        map_pixel_dict = OrderedDict()

        class Pixel_Synopsis():
            def __init__(self, mean_dist, dist_sigma, prob_2D, pixel_index, N128_index, pix_ebv):  # z
                self.mean_dist = mean_dist
                self.dist_sigma = dist_sigma
                self.forced_norm = 0.0

                self.prob_2D = prob_2D
                self.pixel_index = pixel_index
                self.N128_index = N128_index
                self.pix_ebv = pix_ebv
                self.z = 0.0

                # From the tiles that contain this pixel
                # band:value
                self.measured_bands = []
                self.lim_mags = OrderedDict()
                self.delta_mjds = OrderedDict()

                # From the model (only select the bands that have been imaged)
                self.A_lambda = OrderedDict()  # band:value

                # Final calculation
                # model:band:value
                self.best_integrated_probs = OrderedDict()

            def __str__(self):
                return str(self.__dict__)

        count_bad_pixels = 0

        initial_integrands = []
        for p in map_pixels:
            mean_dist = float(p[7])
            dist_sigma = float(p[8])
            prob_2D = float(p[3])
            pixel_index = int(p[2])
            N128_pixel_index = int(p[10])
            pix_ebv = ebv[N128_pixel_index]

            if mean_dist == 0.0:
                # distance did not converge for this pixel. pass...
                print("Bad Index: %s" % pixel_index)
                count_bad_pixels += 1
                continue

            p_new = Pixel_Synopsis(
                mean_dist,
                dist_sigma,
                prob_2D,
                pixel_index,
                N128_pixel_index,
                pix_ebv)

            initial_integrands.append((p_new.pixel_index, p_new.mean_dist, p_new.dist_sigma))
            map_pixel_dict[pixel_index] = p_new

        # print("Starting z-cosmo pool (%s distances)..." % len(initial_integrands))
        # it1 = time.time()
        # pool1 = mp.Pool(processes=self.options.num_cpu, maxtasksperchild=100)
        # resolved_z = list(pool1.imap_unordered(initial_z, initial_integrands, chunksize=1000))
        # with open("%s/%s" % (FORMATTED_HEALPIX_DIR, 'resolved_z.pkl'), 'wb') as handle:
        #     pickle.dump(resolved_z, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # pool1.close()
        # pool1.join()
        # del pool1
        # it2 = time.time()
        # print("... finished z-cosmo pool: %s [seconds]" % (it2 - it1))

        resolved_z = None
        with open("%s/%s" % (FORMATTED_HEALPIX_DIR, 'resolved_z.pkl'), 'rb') as handle:
            resolved_z = pickle.load(handle)

        for rz in resolved_z:
            map_pixel_dict[rz[0]].z = rz[1]
        print("\nMap pixel dict complete. %s bad pixels." % count_bad_pixels)
        # endregion

        SWOPE = Detector("SWOPE", 0.49493333333333334, 0.4968666666666667, detector_id=1)
        THACHER = Detector("THACHER", 0.34645333333333334, 0.34645333333333334, detector_id=3)
        NICKEL = Detector("NICKEL", 0.2093511111111111, 0.2093511111111111, detector_id=4)
        KAIT = Detector("KAIT", 0.1133333333, 0.1133333333, detector_id=6)
        SINISTRO = Detector("SINISTRO", 0.4416666667, 0.4416666667, detector_id=7)

        # region Load Tiles
        # Get and instantiate all observed tiles
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
                Detector_id = %s and 
                Mag_Lim IS NOT NULL 
        '''
        observed_tiles = []

        print("\nLoading Swope's Observed Tiles...")
        # ot_result = query_db([observed_tile_select % (healpix_map_id, SWOPE.id)])[0]
        # with open("%s/%s" % (FORMATTED_HEALPIX_DIR, 'swope_tile.pkl'), 'wb') as handle:
        #     pickle.dump(ot_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        ot_result = None
        with open("%s/%s" % (FORMATTED_HEALPIX_DIR, 'swope_tile.pkl'), 'rb') as handle:
            ot_result = pickle.load(handle)
        for ot in ot_result:
            t = Tile(float(ot[3]), float(ot[4]), SWOPE.deg_width, SWOPE.deg_height, healpix_map_nside,
                     tile_id=int(ot[0]))
            t.field_name = ot[2]
            t.mjd = float(ot[8])
            t.mag_lim = float(ot[10])
            t.band_id = int(ot[7])
            observed_tiles.append(t)
        print("Loaded %s %s tiles..." % (len(ot_result), SWOPE.name))

        print("\nLoading Nickel's Observed Tiles...")
        # ot_result = query_db([observed_tile_select % (healpix_map_id, NICKEL.id)])[0]
        # with open("%s/%s" % (FORMATTED_HEALPIX_DIR, 'nickel_tile.pkl'), 'wb') as handle:
        #     pickle.dump(ot_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        ot_result = None
        with open("%s/%s" % (FORMATTED_HEALPIX_DIR, 'nickel_tile.pkl'), 'rb') as handle:
            ot_result = pickle.load(handle)
        for ot in ot_result:
            t = Tile(float(ot[3]), float(ot[4]), NICKEL.deg_width, NICKEL.deg_height, healpix_map_nside,
                     tile_id=int(ot[0]))
            t.field_name = ot[2]
            t.mjd = float(ot[8])
            t.mag_lim = float(ot[10])
            t.band_id = int(ot[7])
            observed_tiles.append(t)
        print("Loaded %s %s tiles..." % (len(ot_result), NICKEL.name))

        print("\nLoading Thacher's Observed Tiles...")
        # ot_result = query_db([observed_tile_select % (healpix_map_id, THACHER.id)])[0]
        # with open("%s/%s" % (FORMATTED_HEALPIX_DIR, 'thacher_tile.pkl'), 'wb') as handle:
        #     pickle.dump(ot_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        ot_result = None
        with open("%s/%s" % (FORMATTED_HEALPIX_DIR, 'thacher_tile.pkl'), 'rb') as handle:
            ot_result = pickle.load(handle)
        for ot in ot_result:
            t = Tile(float(ot[3]), float(ot[4]), THACHER.deg_width, THACHER.deg_height, healpix_map_nside,
                     tile_id=int(ot[0]))
            t.field_name = ot[2]
            t.mjd = float(ot[8])
            t.mag_lim = float(ot[10])
            t.band_id = int(ot[7])

            observed_tiles.append(t)
        print("Loaded %s %s tiles..." % (len(ot_result), THACHER.name))

        print("\nLoading KAIT's Observed Tiles...")
        # ot_result = query_db([observed_tile_select % (healpix_map_id, KAIT.id)])[0]
        # with open("%s/%s" % (FORMATTED_HEALPIX_DIR, 'kait_tile.pkl'), 'wb') as handle:
        #     pickle.dump(ot_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        ot_result = None
        with open("%s/%s" % (FORMATTED_HEALPIX_DIR, 'kait_tile.pkl'), 'rb') as handle:
            ot_result = pickle.load(handle)
        for ot in ot_result:
            t = Tile(float(ot[3]), float(ot[4]), KAIT.deg_width, KAIT.deg_height, healpix_map_nside, tile_id=int(ot[0]))
            t.field_name = ot[2]
            t.mjd = float(ot[8])
            t.mag_lim = float(ot[10])
            t.band_id = int(ot[7])
            observed_tiles.append(t)
        print("Loaded %s %s tiles..." % (len(ot_result), KAIT.name))

        print("\nLoading SINISTRO's Observed Tiles...")
        # ot_result = query_db([observed_tile_select % (healpix_map_id, SINISTRO.id)])[0]
        # with open("%s/%s" % (FORMATTED_HEALPIX_DIR, 'sinistro_tile.pkl'), 'wb') as handle:
        #     pickle.dump(ot_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        ot_result = None
        with open("%s/%s" % (FORMATTED_HEALPIX_DIR, 'sinistro_tile.pkl'), 'rb') as handle:
            ot_result = pickle.load(handle)
        for ot in ot_result:
            t = Tile(float(ot[3]), float(ot[4]), SINISTRO.deg_width, SINISTRO.deg_height, healpix_map_nside,
                     tile_id=int(ot[0]))
            t.field_name = ot[2]
            t.mjd = float(ot[8])
            t.mag_lim = float(ot[10])
            t.band_id = int(ot[7])
            observed_tiles.append(t)
        print("Loaded %s %s tiles..." % (len(ot_result), SINISTRO.name))
        # endregion

        print("\nUpdating pixel `delta_mjds` and `lim_mags`...")
        # region Initialize models
        # For each tile:
        #   we want the MJD of observation, and add that to the list of a pixels' MJD collection.
        #   we want the limiting mag, add that to the list of a pixel's lim mag collection
        #   we want to correct the delta_mjd for time dilation
        for t in observed_tiles:
            pix_indices = t.enclosed_pixel_indices

            for i in pix_indices:

                # get band from id...
                band = band_dict_by_id[t.band_id]
                band_name = band[1]

                # Some pixels are omitted because their distance information did not converge
                if i not in map_pixel_dict:
                    continue

                pix_synopsis_new = map_pixel_dict[i]

                if band_name not in pix_synopsis_new.measured_bands:
                    pix_synopsis_new.measured_bands.append(band_name)
                    pix_synopsis_new.delta_mjds[band_name] = {}
                    pix_synopsis_new.lim_mags[band_name] = {}

                # DEBUG
                if t.mjd in pix_synopsis_new.lim_mags[band_name]:
                    if pix_synopsis_new.lim_mags[band_name][t.mjd] != t.mag_lim:
                        print("Collision! Tile: %s" % t)
                        print("MJD: %s" % t.mjd)
                        print("Previous mag lim: %s" % pix_synopsis_new.lim_mags[band_name][t.mjd])
                        print("New mag lim: %s" % t.mag_lim)
                # DEBUG

                # time-dilate the delta_mjd, which is used to get the Abs Mag LC point
                pix_synopsis_new.delta_mjds[band_name][t.mjd] = (t.mjd - MERGER_MJD) / (1.0 + pix_synopsis_new.z)
                pix_synopsis_new.lim_mags[band_name][t.mjd] = (t.mag_lim)

        print("\nInitializing %s models..." % len(models))
        # endregion

        print("\nUpdating pixel `A_lambda`...")
        # region Update pixel MWE
        # Set pixel `A_lambda`
        # pixels_to_integrate = []
        # for pix_index, pix_synopsis in map_pixel_dict.items():
        #     for model_param_tuple, model_dict in models.items():
        #
        #         if model_param_tuple not in pix_synopsis.best_integrated_probs:
        #             pix_synopsis.best_integrated_probs[model_param_tuple] = {}
        #
        #         for model_col in model_dict.keys():
        #             if model_col in band_mapping_new:
        #
        #                 band = band_dict_by_name[band_mapping_new[model_col]]
        #                 band_name = band[1]
        #                 band_coeff = band[2]
        #
        #                 if band_name in pix_synopsis.measured_bands:
        #                     if band_name not in pix_synopsis.A_lambda:
        #                         pix_synopsis.A_lambda[band_name] = (pix_synopsis.pix_ebv * band_coeff)
        #
        #         for band in pix_synopsis.measured_bands:
        #             if band not in pix_synopsis.best_integrated_probs[model_param_tuple]:
        #                 pix_synopsis.best_integrated_probs[model_param_tuple][band] = {}
        #
        #             for i, (mjd, delta_mjd) in enumerate(pix_synopsis.delta_mjds[band].items()):
        #                 if mjd not in pix_synopsis.best_integrated_probs[model_param_tuple][band]:
        #                     pix_synopsis.best_integrated_probs[model_param_tuple][band][mjd] = 0.0
        #
        #                 pixels_to_integrate.append((
        #                     (pix_index, model_param_tuple), # pix key for locating result
        #                     pix_synopsis.mean_dist,
        #                     pix_synopsis.dist_sigma,
        #                     pix_synopsis.A_lambda[band],
        #                     pix_synopsis.prob_2D,
        #                     band,
        #                     mjd,
        #                     pix_synopsis.lim_mags[band][mjd],
        #                     model_dict,
        #                     delta_mjd
        #                 ))

        for pix_index, pix_synopsis in map_pixel_dict.items():
            for model_param_tuple, model_dict in models.items():
                for model_col in model_dict.keys():
                    if model_col in band_mapping_new:

                        band = band_dict_by_name[band_mapping_new[model_col]]
                        band_id = band[0]
                        band_name = band[1]
                        band_coeff = band[2]

                        if band_name in pix_synopsis.measured_bands:
                            if band_name not in pix_synopsis.A_lambda:
                                pix_a_lambda = pix_synopsis.pix_ebv * band_coeff
                                pix_synopsis.A_lambda[band_name] = pix_a_lambda




        # endregion

        compute_start = time.time()
        print("\nUpdating `map_pixel_dict `...")
        count = 0
        for pix_index, pix_synopsis in map_pixel_dict.items():
            for band in pix_synopsis.measured_bands:
                for model_param_tuple, model_dict in models.items():
                    pixel_delta_mjd = pix_synopsis.delta_mjds[band]

                    for i, (mjd, delta_mjd) in enumerate(pixel_delta_mjd.items()):
                        if model_param_tuple not in pix_synopsis.best_integrated_probs:
                            pix_synopsis.best_integrated_probs[model_param_tuple] = {}

                        if band not in pix_synopsis.best_integrated_probs[model_param_tuple]:
                            pix_synopsis.best_integrated_probs[model_param_tuple][band] = {mjd: 0.0}

            count += 1
            if count % 1000 == 0:
                print("Processed: %s" % count)
        compute_end = time.time()
        print("Update `map_pixel_dict_new` time: %s [seconds]" % (compute_end - compute_start))



        prep_end = time.time()
        print("Prep time: %s [seconds]" % (prep_end - prep_start))



        # Compose integrands
        print("Building integrands...")
        integrands_start = time.time()
        pixels_to_integrate = []
        for model_key, model_func_dict in models.items():
            for pix_index, pix_synopsis in map_pixel_dict.items():
                for band in pix_synopsis.measured_bands:
                    mwe = pix_synopsis.A_lambda[band]
                    prob_2D = pix_synopsis.prob_2D
                    mean_dist = pix_synopsis.mean_dist
                    dist_sigma = pix_synopsis.dist_sigma

                    for i, (mjd, delta_mjd) in enumerate(pix_synopsis.delta_mjds[band].items()):
                        pix_key = (pix_index, model_key)
                        pixels_to_integrate.append((
                                pix_key,
                                mean_dist,
                                dist_sigma,
                                mwe,
                                prob_2D,
                                band,
                                mjd,
                                pix_synopsis.lim_mags[band][mjd],
                                model_func_dict,
                                delta_mjd
                        ))
        integrands_end = time.time()
        print("Building integrands time: %s [seconds]" % (integrands_end - integrands_start))

        print("Integrating %s total model/pixel combinations..." % len(pixels_to_integrate))


        mp_start = time.time()
        pool3 = mp.Pool(processes=self.options.num_cpu, maxtasksperchild=1)
        integrated_pixels = pool3.imap_unordered(integrate_pixel,
                                                 pixels_to_integrate,
                                                 chunksize=2204050)

        for ip in integrated_pixels:
            pix_key = ip[0]
            pix_index = pix_key[0]
            model_param_tuple = pix_key[1]
            band = pix_key[2]
            mjd = pix_key[3]
            prob = ip[1]
            map_pixel_dict[pix_index].best_integrated_probs[model_param_tuple][band][mjd] = prob
        mp_end = time.time()

        pool3.close()
        pool3.join()
        del pool3
        print("Integration time: %s [seconds]" % (mp_end - mp_start))
        # endregion




        ## NEW (Take binomial product of chance seeing it in each epoch)
        # probability by model by band by pixel, for all epochs per pixel
        probs_by_model_band_pix_index = {}

        # summed pixel probability by model and band
        probs_by_model_band = {}

        # final probability by model
        net_prob_by_model = {}

        for model_param_tuple, model_dict in models.items():

            # Initialize the model key for each dictionary
            if model_param_tuple not in net_prob_by_model:
                net_prob_by_model[model_param_tuple] = {}

            if model_param_tuple not in probs_by_model_band:
                probs_by_model_band[model_param_tuple] = {}

            if model_param_tuple not in probs_by_model_band_pix_index:
                probs_by_model_band_pix_index[model_param_tuple] = {}

            for pix_index, pix_synopsis in map_pixel_dict.items():
                for band in pix_synopsis.measured_bands:

                    # Initialize the band key for each dictionary
                    if band not in probs_by_model_band_pix_index[model_param_tuple]:
                        probs_by_model_band_pix_index[model_param_tuple][band] = {}

                    if band not in probs_by_model_band[model_param_tuple]:
                        probs_by_model_band[model_param_tuple][band] = {}

                    # sanity -- if the pixel prob is itself 0.0, we can just skip
                    if pix_synopsis.prob_2D <= 0:
                        probs_by_model_band_pix_index[model_param_tuple][band][pix_index] = 0.0
                        continue

                    # For each pixel:
                    #   take the compliment from each epoch's integrated probability
                    #   take the product of these compliments (the product of the individual marginal probabilities of a non-detection)
                    #   take the compliment of this join marginal probability to find the prob of at least one detection
                    product_of_compliments = 1.0

                    test_probs = []
                    for mjd, prob in pix_synopsis.best_integrated_probs[model_param_tuple][band].items():
                        product_of_compliments *= (1.0 - prob / pix_synopsis.prob_2D)
                        test_probs.append(prob)

                    compliment_of_product_of_compliments = pix_synopsis.prob_2D * (1 - product_of_compliments)

                    probs_by_model_band_pix_index[model_param_tuple][band][
                        pix_index] = compliment_of_product_of_compliments

            # Sum all pixels for a given band and model, that makes up the per model, per band prob
            for band, pix_prob_dict in probs_by_model_band_pix_index[model_param_tuple].items():
                summed_prob = 0.0
                for pix_index, prob in pix_prob_dict.items():
                    summed_prob += prob

                probs_by_model_band[model_param_tuple][band] = summed_prob

            # Finally, as above, take the compliment of the product of the compliments for each band
            # to compute the probability of at least 1 detection in any band
            total_covered_prob = 0.0
            for pix_index, pix_synopsis in map_pixel_dict.items():
                total_covered_prob += pix_synopsis.prob_2D

            net_product_of_compliments = 1.0
            for band, sum_prob in probs_by_model_band[model_param_tuple].items():
                net_product_of_compliments *= (1.0 - sum_prob / total_covered_prob)

            net_compliment_of_product_of_compliments = total_covered_prob * (1.0 - net_product_of_compliments)
            net_prob_by_model[model_param_tuple] = net_compliment_of_product_of_compliments




        cols = ['vej', 'mej', 'ye', 'Prob']
        dtype = ['f8', 'f8', 'f8', 'f8']
        result_table = Table(dtype=dtype, names=cols)

        for model_param_tuple, prob in net_prob_by_model.items():

            result_table.add_row([model_param_tuple[0], model_param_tuple[1], model_param_tuple[2], prob])

        result_table.write("%s/Detection_Results_KN_%s.prob" % (FORMATTED_MODEL_OUTPUT_DIR,
                                                                self.options.sub_dir),
                           overwrite=True, format='ascii.ecsv')


        # endregion

if __name__ == "__main__":
    useagestring = """python ComputeModelDetection.py [options]

Example with healpix_dir defaulted to 'Events/<gwid>' amd model_dir to 'Events/{GWID}/Models':
python ComputeModelDetection.py --gw_id <gwid> --healpix_file <filename>

Assumes .dat files with Astropy ECSV format: 
https://docs.astropy.org/en/stable/api/astropy.io.ascii.Ecsv.html

"""
    start = time.time()

    teglon = Teglon()
    parser = teglon.add_options(usage=useagestring)
    options, args = parser.parse_args()
    teglon.options = options

    teglon.main()

    end = time.time()
    duration = (end - start)
    print("\n********* start DEBUG ***********")
    print("Teglon `ComputeModelDetection_KN` execution time: %s" % duration)
    print("********* end DEBUG ***********\n")
