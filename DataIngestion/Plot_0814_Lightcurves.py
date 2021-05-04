# region imports
import matplotlib as mpl
print(mpl.__version__)


import sys
sys.path.append('../')

import os
import sys
import pickle
from scipy.interpolate import interp1d, interp2d, spline
from HEALPix_Helpers import *
from Tile import *
from Pixel_Element import *
from collections import OrderedDict
from astropy.table import Table
from astropy.time import Time, TimeDelta
import pytz
import csv
from astroplan import moon_illumination, moon_phase_angle
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.patheffects as path_effects
from configparser import RawConfigParser
import json

# endregion

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

from mysql.connector import Error
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

def query_db(query_list, commit=False):
    # query_string = ";".join(query_list)

    results = []
    try:
        chunk_size = 1e+6

        db = my.connect(host=db_host, user=db_user, passwd=db_pwd, db=db_name, port=3306)
        cursor = db.cursor()

        for q in query_list:
            cursor.execute(q)

            if commit:  # used for updates, etc
                db.commit()

            streamed_results = []
            print("\tfetching results...")
            while True:
                r = cursor.fetchmany(1000000)
                count = len(r)
                streamed_results += r
                size_in_mb = sys.getsizeof(streamed_results) / 1.0e+6

                print("\t\tfetched: %s; current length: %s; running size: %0.3f MB" % (
                count, len(streamed_results), size_in_mb))

                if not r or count < chunk_size:
                    break

        results.append(streamed_results)

    except Error as e:
        print('Error:', e)
    finally:
        cursor.close()
        db.close()

    return results

is_poster_plot = False

class Teglon:

    def add_options(self, parser=None, usage=None, config=None):
        import optparse
        if parser == None:
            parser = optparse.OptionParser(usage=usage, conflict_handler="resolve")

        return (parser)

    def main(self):

        # Get Kilonova.space data
        gw170817_filters = ["g", "r", "i"]
        gw170817_json_file = open("../HST_Proposals/GW170817.json")
        gw170817_json_data = json.load(gw170817_json_file)
        json_KN_merger_mjd = float(gw170817_json_data["GW170817"]["timeofmerger"][0]["value"])
        sss17a_json_phot = gw170817_json_data["GW170817"]["photometry"]

        hst_json_data = OrderedDict()
        # unique_bands = []
        for p in sss17a_json_phot:

            # if ('band' in p):
            #     if p['band'] not in unique_bands:
            #         unique_bands.append(p['band'])

            if ('band' in p) and ('u_time' in p) and ('upperlimit' not in p) and (p['u_time'] == "MJD") and (
                    'model' not in p):
                for fk in gw170817_filters:

                    if (p['band'] == fk):
                        if fk not in hst_json_data:
                            hst_json_data[fk] = [[], []]

                        hst_json_data[fk][0].append(float(p['time']) - json_KN_merger_mjd)
                        hst_json_data[fk][1].append(float(p['magnitude']))


        # print(unique_bands)
        # return 0




        pickle_path = "../Events/S190814bv/Pickles"

        map_pixels = None
        with open("%s/%s" % (pickle_path, 'map_pix.pkl'), 'rb') as handle:
            map_pixels = pickle.load(handle)
        print("Retrieved %s map pixels..." % len(map_pixels))
        map_pix = [Pixel_Element(int(mp[2]), 1024, float(mp[3]), pixel_id=int(mp[0])) for mp in
                   map_pixels]
        map_pix_dict = {}
        for m in map_pixels:
            map_pix_dict[int(m[2])] = float(m[3])

        map_pix_sorted = sorted(map_pix, key=lambda x: x.prob, reverse=True)
        index_90th = 0
        cutoff_90th = 0.9

        print("Find index for 90th...")
        cum_prob = 0.0
        for i in range(len(map_pix_sorted)):
            cum_prob += map_pix_sorted[i].prob
            index_90th = i

            if (cum_prob >= cutoff_90th):
                break
        print("... %s" % index_90th)
        _90th_indices = []
        for p in map_pix_sorted[0:index_90th]:
            _90th_indices.append(p.index)


        merger_distance = 267 # Mpc
        d = Distance(merger_distance, u.Mpc)
        gw0814_dist_mod = 5.0*np.log10(d.value*1e6) - 5
        z = d.compute_z(cosmology=cosmo)
        print("Dist mod: %s" % gw0814_dist_mod)

        MERGER_MJD = 58709.882824224536
        merger_time = Time(MERGER_MJD, format='mjd')
        print("** Running models for MJD: %s; UTC: %s **" % (MERGER_MJD, merger_time.to_datetime(timezone=pytz.utc)))


        # grb_model_files = {
        #     "onaxis":"../GRBModels/NewModels/grb_onaxis/6/grb_0.0000_0.2929_0.2511886.dat",
        #     "offaxis":"../GRBModels/NewModels/grb_offaxis/8/grb_0.2967_0.2929_0.2511886.dat"
        # }

        # OLD models that Charlie likes for now...
        # grb_model_files = {
        #     "onaxis": "../GRBModels/NewModels/20200811_grb_onaxis/6/grb_0.0000_0.2929_0.2511886.dat",
        #     "offaxis": "../GRBModels/NewModels/20200811_grb_offaxis/8/grb_0.2967_0.2929_0.2511886.dat"
        # }

        # 20210225 - more correct models, but tabling for now...
        grb_model_files = {
            "onaxis": "../GRBModels/NewModels/20200811_grb_onaxis/4/grb_0.0000_21.5443_0.2511886.dat",
            "offaxis": "../GRBModels/NewModels/20200811_grb_offaxis/9/grb_0.2967_21.5443_0.2511886.dat"
        }

        # kn_model_files = {"blue": "../Events/S190814bv/Metzger_0.0254_0.2324_0.4500_blueKN.dat",
        #                   "red": "../Events/S190814bv/Metzger_0.0167_0.1084_0.1000_redKN.dat"}
        # kn_model_files = {"blue": "../KNModels/4/Metzger_0.024890_0.256897_0.450000.dat",
        #                   "red": "../KNModels/9/Metzger_0.058653_0.192069_0.100000.dat"}
        kn_model_files = {"blue": "../KN_NEW_Binning/8/Metzger_0.024890_0.256897_0.450000.dat",
                          "red": "../KN_NEW_Binning/4/Metzger_0.038208_0.143448_0.100000.dat"}

        kn_models = {}
        for key, mf in kn_model_files.items():
            model_table = Table.read(mf, format='ascii.ecsv')
            mask = model_table['sdss_g'] != np.nan

            # adjust model time to observer-frame
            model_time = np.asarray(model_table['time'][mask])/(1.0 + z)

            # adjust model luminosity to observer-frame
            g = np.asarray(model_table['sdss_g'][mask]) #+ dist_mod
            r = np.asarray(model_table['sdss_r'][mask]) #+ dist_mod
            i = np.asarray(model_table['sdss_i'][mask]) #+ dist_mod
            clear = np.asarray(model_table['Clear'][mask]) #+ dist_mod


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

            kn_models[key] = {
                'time': model_time,
                'sdss_g': f_g,
                'sdss_r': f_r,
                'sdss_i': f_i,
                'Clear': f_clear
            }

        grb_models = {}
        for key, mf in grb_model_files.items():
            model_table = Table.read(mf, format='ascii.ecsv')
            mask = model_table['sdss_g'] != np.nan

            # adjust model time to observer-frame
            model_time = np.asarray(model_table['time'][mask]) / (1.0 + z)

            # adjust model luminosity to observer-frame
            g = np.asarray(model_table['sdss_g'][mask])  # + dist_mod
            r = np.asarray(model_table['sdss_r'][mask])  # + dist_mod
            i = np.asarray(model_table['sdss_i'][mask])  # + dist_mod
            clear = np.asarray(model_table['Clear'][mask])  # + dist_mod

            base_name = os.path.basename(mf)
            print("Loading `%s`" % base_name)

            # Get interpolation function for each Light Curve
            f_g = interp1d(model_time, g, fill_value="extrapolate")
            f_r = interp1d(model_time, r, fill_value="extrapolate")
            f_i = interp1d(model_time, i, fill_value="extrapolate")
            f_clear = interp1d(model_time, clear, fill_value="extrapolate")

            grb_models[key] = {
                'time': model_time,
                'sdss_g': f_g,
                'sdss_r': f_r,
                'sdss_i': f_i,
                'Clear': f_clear
            }

        # Convert models to observer-frame -- apply time dilation, distance modulus, and MWE extinction toward center of
        # high prob area in 0814 localization

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
            "si": "SINISTRO",
            "w": "WISE",
        }


        # ss17a_B = []
        # ss17a_B_time = []
        #
        # ss17a_V = []
        # ss17a_V_time = []

        ss17a_g = []
        ss17a_g_time = []

        ss17a_r = []
        ss17a_r_time = []

        ss17a_i = []
        ss17a_i_time = []

        KN_merger_mjd = 57982.52851852
        with open("../Events/S190814bv/SSS17a_Swope.phot", "r") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ')
            for row in csvreader:
                delta_mjd = float(row[0]) - 2400000.5 - KN_merger_mjd
                band = row[1]
                mag = float(row[2])

                # if band == "B":
                #     ss17a_B.append(mag)
                #     ss17a_B_time.append(delta_mjd)
                # elif band == "V":
                #     ss17a_V.append(mag)
                #     ss17a_V_time.append(delta_mjd)
                # elif band == "g":
                if band == "g":
                    ss17a_g.append(mag)
                    ss17a_g_time.append(delta_mjd)
                elif band == "r":
                    ss17a_r.append(mag)
                    ss17a_r_time.append(delta_mjd)
                elif band == "i":
                    ss17a_i.append(mag)
                    ss17a_i_time.append(delta_mjd)



        # region Load Serialized Sky Pixels, EBV, and Models from Disk
        # LOADING NSIDE 128 SKY PIXELS AND EBV INFORMATION
        print("\nLoading NSIDE 128 pixels...")
        nside128 = 128
        N128_dict = None
        with open('%s/N128_dict.pkl' % pickle_path, 'rb') as handle:
            N128_dict = pickle.load(handle)
        del handle

        print("\nLoading existing EBV...")
        ebv = None
        with open('%s/ebv.pkl' % pickle_path, 'rb') as handle:
            ebv = pickle.load(handle)


        # endregion

        # region Get Map, Bands and initialize pixels.
        healpix_map_id = 7
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



        SWOPE = Detector("SWOPE", 0.49493333333333334, 0.4968666666666667, detector_id=1)
        THACHER = Detector("THACHER", 0.34645333333333334, 0.34645333333333334, detector_id=3)
        NICKEL = Detector("NICKEL", 0.2093511111111111, 0.2093511111111111, detector_id=4)
        KAIT = Detector("KAIT", 0.1133333333, 0.1133333333, detector_id=6)
        SINISTRO = Detector("SINISTRO", 0.4416666667, 0.4416666667, detector_id=7)
        WISE = Detector("WISE", 0.94833333333, 0.94833333333, detector_id=7)


        swope_tiles = []
        nickel_tiles = []
        thacher_tiles = []
        kait_tiles = []
        sinistro_tiles = []



        print("\nLoading Swope's Observed Tiles...")
        ot_result = None
        with open("%s/%s" % (pickle_path, 'swope_tile.pkl'), 'rb') as handle:
            ot_result = pickle.load(handle)
        for ot in ot_result:
            t = Tile(float(ot[3]), float(ot[4]), SWOPE.deg_width, SWOPE.deg_height, healpix_map_nside,
                     tile_id=int(ot[0]))
            t.field_name = ot[2]
            t.mjd = float(ot[8]) - MERGER_MJD
            t.mag_lim = float(ot[10])
            t.band_id = int(ot[7])
            swope_tiles.append(t)
        print("Loaded %s %s tiles..." % (len(ot_result), SWOPE.name))


        print("\nLoading Nickel's Observed Tiles...")
        ot_result = None
        with open("%s/%s" % (pickle_path, 'nickel_tile.pkl'), 'rb') as handle:
            ot_result = pickle.load(handle)
        for ot in ot_result:
            t = Tile(float(ot[3]), float(ot[4]), NICKEL.deg_width, NICKEL.deg_height, healpix_map_nside,
                     tile_id=int(ot[0]))
            t.field_name = ot[2]
            t.mjd = float(ot[8]) - MERGER_MJD
            t.mag_lim = float(ot[10])
            t.band_id = int(ot[7])
            nickel_tiles.append(t)
        print("Loaded %s %s tiles..." % (len(ot_result), NICKEL.name))


        print("\nLoading Thacher's Observed Tiles...")
        ot_result = None
        with open("%s/%s" % (pickle_path, 'thacher_tile.pkl'), 'rb') as handle:
            ot_result = pickle.load(handle)
        for ot in ot_result:
            t = Tile(float(ot[3]), float(ot[4]), THACHER.deg_width, THACHER.deg_height, healpix_map_nside,
                     tile_id=int(ot[0]))
            t.field_name = ot[2]
            t.mjd = float(ot[8]) - MERGER_MJD
            t.mag_lim = float(ot[10])
            t.band_id = int(ot[7])

            thacher_tiles.append(t)
        print("Loaded %s %s tiles..." % (len(ot_result), THACHER.name))


        print("\nLoading KAIT's Observed Tiles...")
        ot_result = None
        with open("%s/%s" % (pickle_path, 'kait_tile.pkl'), 'rb') as handle:
            ot_result = pickle.load(handle)
        for ot in ot_result:
            t = Tile(float(ot[3]), float(ot[4]), KAIT.deg_width, KAIT.deg_height, healpix_map_nside, tile_id=int(ot[0]))
            t.field_name = ot[2]
            t.mjd = float(ot[8]) - MERGER_MJD
            t.mag_lim = float(ot[10])
            t.band_id = int(ot[7])
            kait_tiles.append(t)
        print("Loaded %s %s tiles..." % (len(ot_result), KAIT.name))


        print("\nLoading SINISTRO's Observed Tiles...")
        ot_result = None
        with open("%s/%s" % (pickle_path, 'sinistro_tile.pkl'), 'rb') as handle:
            ot_result = pickle.load(handle)
        for ot in ot_result:
            t = Tile(float(ot[3]), float(ot[4]), SINISTRO.deg_width, SINISTRO.deg_height, healpix_map_nside,
                     tile_id=int(ot[0]))
            t.field_name = ot[2]
            t.mjd = float(ot[8]) - MERGER_MJD
            t.mag_lim = float(ot[10])
            t.band_id = int(ot[7])
            sinistro_tiles.append(t)
        print("Loaded %s %s tiles..." % (len(ot_result), SINISTRO.name))

        # wise_tiles = []
        # print("\nLoading WISE's Observed Tiles...")
        # wise_detector_id = 9
        # observed_tile_select = '''
        #             SELECT
        #                 id,
        #                 Detector_id,
        #                 FieldName,
        #                 RA,
        #                 _Dec,
        #                 EBV,
        #                 N128_SkyPixel_id,
        #                 Band_id,
        #                 MJD,
        #                 Exp_Time,
        #                 Mag_Lim,
        #                 HealpixMap_id
        #             FROM
        #                 ObservedTile
        #             WHERE
        #                 HealpixMap_id = %s and
        #                 Detector_id = %s
        #         '''
        # ot_result = query_db([observed_tile_select % (healpix_map_id, wise_detector_id)])[0]
        #
        # for ot in ot_result:
        #     t = Tile(float(ot[3]), float(ot[4]), WISE.deg_width, WISE.deg_height, healpix_map_nside,
        #              tile_id=int(ot[0]))
        #     t.field_name = ot[2]
        #     t.mjd = float(ot[8]) - MERGER_MJD
        #     t.mag_lim = float(ot[10])
        #     t.band_id = int(ot[7])
        #     wise_tiles.append(t)
        # print("Loaded %s %s tiles..." % (len(ot_result), WISE.name))

        y_min = -31 # version to capture more accurate SGRB models (20210301)
        # y_min = -25
        # y_min = -23 # old version
        # y_max = - 5
        y_max = - 9

        def moon_moon(x_v):
            moon_illums = []
            for t_d in x_v:
                td = TimeDelta(t_d, format='jd')
                moon_illums.append(moon_illumination(merger_time + td))
            return np.asarray(moon_illums)

        x_vals = np.linspace(0, 12, 50)
        y_vals = np.linspace(y_min, y_max, 50)
        z_vals = moon_moon(x_vals)
        moon_tuples = []
        for i, x in enumerate(x_vals):
            for y in y_vals:
                moon_tuples.append((x, y, z_vals[i]))
        X = []
        Y = []
        Z = []
        for mt in moon_tuples:
            X.append(mt[0])
            Y.append(mt[1])
            Z.append(mt[2])
        _2d_func = interp2d(X, Y, Z, kind="linear")
        xx, yy = np.meshgrid(x_vals, y_vals)
        z_new = _2d_func(x_vals, y_vals)

        # fig = plt.figure(figsize=(10, 10), dpi=600)

        fig = None
        if not is_poster_plot:
            fig = plt.figure(figsize=(10, 12), dpi=600)
        else:
            fig = plt.figure(figsize=(10, 10), dpi=600)

        ax = fig.add_subplot(111)

        ax.contourf(xx, yy, z_new, levels=np.linspace(0.0, 1.0, 200), cmap=plt.cm.bone, alpha=0.5)


        sm = plt.cm.ScalarMappable(norm=colors.Normalize(0.0, 1.0), cmap=plt.cm.bone_r)
        sm.set_array([])  # can be an empty list
        cb = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.06, pad=0.02, alpha=0.5)  # fraction=0.04875

        tks = np.linspace(0.0, 1.0, 5)
        cb.set_ticks(tks)
        tks_strings = ["100%", "75%", "50%", "35%", "0%"]
        cb.ax.set_xticklabels(tks_strings, fontsize=24)
        cb.set_label("Moon Illumination Fraction", fontsize=32, labelpad=9.0)
        # cb.ax.set_ticks_position("top")
        # cax.xaxis.set_ticks_position("top")
        cb.ax.locator_params(nbins=5)
        cb.ax.tick_params(length=8.0, width=2.0)
        cb.outline.set_linewidth(2.0)


        # lc_width = 2.0
        lc_width = 2.5
        pe_width = 4.5
        lc_border = 8.0
        marker_sz = 10.0
        # obs_alpha = 0.5
        # marker_edg_w = 1.0
        # obs_alpha = 0.75
        obs_alpha = 0.50
        marker_edg_w = 0.8
        KN_dist_mod = 5 * np.log10(40 * 1e6) - 5

        g_ls = "-" # solid
        r_ls = (0, (5, 5)) # dashed
        i_ls = (0,(1, 1)) # densely dotted
        clear_ls = (0, (3, 5, 1, 5)) # dashed-dotted





        # 1: (1, "SDSS u", 4.239),
        # 2: (2, "SDSS g", 3.303),
        # 3: (3, "SDSS r", 2.285),
        # 4: (4, "SDSS i", 1.698),
        # 5: (5, "SDSS z", 1.263),
        # 6: (6, "Landolt B", 3.626),
        # 7: (7, "Landolt V", 2.742),
        # 8: (8, "Landolt R", 2.169),
        # 9: (9, "Landolt I", 1.505),
        # 10: (10, "UKIRT J", 0.709),
        # 11: (11, "UKIRT H", 0.449),
        # 12: (12, "UKIRT K", 0.302),
        # 13: (13, "Clear", 0.91)



        band_colors = {
            "SDSS g": 'g',
            "SDSS r": 'r',
            "SDSS i": 'brown',
            "Clear": 'black',
            "Landolt B": 'blue',
            "Landolt V": 'orange',
            "Landolt R": 'magenta',
            "Landolt I": 'purple'
        }

        clrs = {
            "SWOPE": (230.0 / 256.0, 159 / 256.0, 0),
            "NICKEL": (0, 114.0 / 256.0, 178.0 / 256.0),
            "THACHER": (0, 158.0 / 256.0, 115.0 / 256.0),
            "MOSFIRE": (204.0 / 256.0, 121.0 / 256.0, 167.0 / 256.0),
            "SINISTRO_i": (218.0 / 256.0, 35.0 / 256.0, 35.0 / 256.0),
            "SINISTRO_g": "violet",
            "KAIT": (86.0 / 256.0, 180.0 / 256.0, 233.0 / 256.0),
            "WISE": "lime",
        }




        lcogt_i = None
        lcogt_g = None
        has_lcogt_i = False
        has_lcogt_g = False
        import matplotlib.path as mpath

        # triangle = mpath.Path.unit_regular_polygon(3)
        triangle = "v"
        lcogt_i_triangle = mpath.Path.unit_regular_polygon(3)



        lcogt_g_triangle = mpath.Path.unit_regular_polygon(3)

        if not is_poster_plot:
            for i, t in enumerate(sinistro_tiles):

                # num_pix = len(t.enclosed_pixel_indices)
                # num_within = 0.0
                # for t_index in t.enclosed_pixel_indices:
                #     if t_index in _90th_indices:
                #         num_within += 1
                # clr = plt.cm.viridis(num_within/num_pix)
                # was orange
                clr = clrs["SINISTRO_i"]
                pattern = "\\"


                if t.band_id == 4: # SDSS i
                    if not has_lcogt_i:
                        # lcogt_i, = ax.plot(t.mjd, np.asarray(t.mag_lim) - KN_dist_mod, color=clr, marker='p',
                        #                    linestyle='None', label="LCOGT i", mec="black", ms=10, mew=marker_edg_w, alpha=obs_alpha)
                        lcogt_i, = ax.plot(t.mjd, np.asarray(t.mag_lim) - gw0814_dist_mod, color=clr, marker=triangle,
                                           linestyle='None', label="Las Cumbres i", mec="black", ms=marker_sz, mew=marker_edg_w, fillstyle="left", markerfacecoloralt="white",
                                           alpha=obs_alpha)






                        has_lcogt_i = True
                    else:
                        # ax.plot(t.mjd, np.asarray(t.mag_lim) - KN_dist_mod, color=clr, linestyle='None', marker='p',
                        #         mec="black", ms=10, alpha=obs_alpha, mew=marker_edg_w)
                        ax.plot(t.mjd, np.asarray(t.mag_lim) - gw0814_dist_mod, color=clr, linestyle='None', marker=triangle, fillstyle="left", markerfacecoloralt="white",
                                mec="black", ms=11.0, alpha=obs_alpha, mew=marker_edg_w)

            for i, t in enumerate(sinistro_tiles):

                # num_pix = len(t.enclosed_pixel_indices)
                # num_within = 0.0
                # for t_index in t.enclosed_pixel_indices:
                #     if t_index in _90th_indices:
                #         num_within += 1
                # clr = plt.cm.viridis(num_within / num_pix)
                # clr = clrs["SINISTRO_g"]
                clr = clrs["SINISTRO_i"]
                pattern = "/"
                test2 = lcogt_g_triangle.hatch(pattern, density=3)

                if t.band_id == 2: #sdss g

                    if not has_lcogt_g:
                        # lcogt_g, = ax.plot(t.mjd, np.asarray(t.mag_lim) - KN_dist_mod, color=clr, marker='p',
                        #                    linestyle='None', alpha=obs_alpha, label="LCOGT g", mec="black", ms=10, mew=marker_edg_w)
                        lcogt_g, = ax.plot(t.mjd, np.asarray(t.mag_lim) - gw0814_dist_mod, color=clr, marker=triangle, fillstyle="right", markerfacecoloralt="white",
                                           linestyle='None', alpha=obs_alpha, label="Las Cumbres g", mec="black", ms=marker_sz,
                                           mew=marker_edg_w)
                        has_lcogt_g = True
                    else:
                        # ax.plot(t.mjd, np.asarray(t.mag_lim) - KN_dist_mod, color=clr, linestyle='None', marker='p',
                        #         mec="black", ms=10, alpha=obs_alpha, mew=marker_edg_w)
                        ax.plot(t.mjd, np.asarray(t.mag_lim) - gw0814_dist_mod, color=clr, linestyle='None', marker=triangle, fillstyle="right", markerfacecoloralt="white",
                                mec="black", ms=11.0, alpha=obs_alpha, mew=marker_edg_w)

            for i, t in enumerate(kait_tiles):

                # num_pix = len(t.enclosed_pixel_indices)
                # num_within = 0.0
                # for t_index in t.enclosed_pixel_indices:
                #     if t_index in _90th_indices:
                #         num_within += 1
                # clr = plt.cm.viridis(num_within / num_pix)
                clr = clrs["KAIT"]
                # was deepskyblue
                if i == 0:
                    # kait_clear, = ax.plot(t.mjd, np.asarray(t.mag_lim) - KN_dist_mod, color=clr, marker='o', linestyle = 'None',label="KAIT Clear", mec = "black", ms=10,  mew = marker_edg_w)
                    kait_clear, = ax.plot(t.mjd, np.asarray(t.mag_lim) - gw0814_dist_mod, color=clr, marker=triangle,
                                          linestyle='None', label="KAIT Clear", mec="black", ms=marker_sz, mew=marker_edg_w)
                else:
                    # ax.plot(t.mjd, np.asarray(t.mag_lim) - KN_dist_mod, color=clr, linestyle = 'None',marker='o', mec = "black", ms=10, alpha=obs_alpha,  mew = marker_edg_w)
                    ax.plot(t.mjd, np.asarray(t.mag_lim) - gw0814_dist_mod, color=clr, linestyle='None', marker=triangle,
                            mec="black", ms=marker_sz, alpha=obs_alpha, mew=marker_edg_w)

            for i, t in enumerate(nickel_tiles):
                clr = clrs["NICKEL"] # was brown

                # num_pix = len(t.enclosed_pixel_indices)
                # num_within = 0.0
                # for t_index in t.enclosed_pixel_indices:
                #     if t_index in _90th_indices:
                #         num_within += 1
                # clr = plt.cm.viridis(num_within / num_pix)

                if i == 0:
                    # nickel_r, = ax.plot(t.mjd, np.asarray(t.mag_lim) - KN_dist_mod, color=clr, marker='s', linestyle = 'None',label="Nickel r", mec = "black", ms=10, mew = marker_edg_w)
                    nickel_r, = ax.plot(t.mjd, np.asarray(t.mag_lim) - gw0814_dist_mod, color=clr, marker=triangle, linestyle='None',
                                        label="Nickel r", mec="black", ms=marker_sz, mew=marker_edg_w)
                else:
                    # ax.plot(t.mjd, np.asarray(t.mag_lim) - KN_dist_mod, color=clr, linestyle = 'None',marker='s', mec = "black", ms=10, alpha=obs_alpha,  mew =marker_edg_w)
                    ax.plot(t.mjd, np.asarray(t.mag_lim) - gw0814_dist_mod, color=clr, linestyle='None', marker=triangle,
                            mec="black", ms=marker_sz, alpha=obs_alpha, mew=marker_edg_w)

        for i, t in enumerate(swope_tiles):

            # num_pix = len(t.enclosed_pixel_indices)
            # num_within = 0.0
            # for t_index in t.enclosed_pixel_indices:
            #     if t_index in _90th_indices:
            #         num_within += 1
            # clr = plt.cm.viridis(num_within / num_pix)

            clr = clrs["SWOPE"]  # was red
            if i == 0:
                swope_r, = ax.plot(t.mjd, np.asarray(t.mag_lim) - gw0814_dist_mod, color=clr, marker=triangle, linestyle = 'None',label="Swope r", mec = "black", ms=marker_sz, mew =marker_edg_w)
            else:
                ax.plot(t.mjd, np.asarray(t.mag_lim) - gw0814_dist_mod, color=clr, linestyle = 'None',marker=triangle, mec = "black", ms=marker_sz, alpha=obs_alpha,  mew =marker_edg_w)

        if not is_poster_plot:
            for i, t in enumerate(thacher_tiles):
                clr = clrs["THACHER"]  # was violet

                # num_pix = len(t.enclosed_pixel_indices)
                # num_within = 0.0
                # for t_index in t.enclosed_pixel_indices:
                #     if t_index in _90th_indices:
                #         num_within += 1
                # clr = plt.cm.viridis(num_within / num_pix)

                if i == 0:
                    # thacher_r, = ax.plot(t.mjd, np.asarray(t.mag_lim) - KN_dist_mod, color=clr, marker='*', linestyle = 'None',label="Thacher r", mec = "black", ms=14, mew = marker_edg_w)
                    # KN_dist_mod
                    thacher_r, = ax.plot(t.mjd, np.asarray(t.mag_lim) - gw0814_dist_mod, color=clr, marker=triangle,
                                         linestyle='None', label="Thacher r", mec="black", ms=marker_sz, mew=marker_edg_w)
                else:
                    # ax.plot(t.mjd, np.asarray(t.mag_lim) - KN_dist_mod, color=clr, linestyle = 'None',marker='*', mec = "black", ms=14, alpha=obs_alpha,  mew = marker_edg_w)
                    # KN_dist_mod
                    ax.plot(t.mjd, np.asarray(t.mag_lim) - gw0814_dist_mod, color=clr, linestyle='None', marker=triangle,
                            mec="black", ms=marker_sz, alpha=obs_alpha, mew=marker_edg_w)


        # for i, t in enumerate(wise_tiles):
        #     clr = clrs["WISE"]
        #
        #     if i == 0:
        #         ax.plot(t.mjd, np.asarray(t.mag_lim) - gw0814_dist_mod, color=clr, marker=triangle,
        #                              linestyle='None', mec="black", ms=marker_sz, mew=marker_edg_w, zorder=9999)
        #     else:
        #         ax.plot(t.mjd, np.asarray(t.mag_lim) - gw0814_dist_mod, color=clr, linestyle='None', marker=triangle,
        #                 mec="black", ms=marker_sz, mew=marker_edg_w, zorder=9999)





        if not is_poster_plot:
            blue_kn_g, = ax.plot(kn_models["blue"]['time'], kn_models["blue"]['sdss_g'](kn_models["blue"]['time']),
                                 color="blue", linestyle=g_ls, label="SDSS g", linewidth=lc_width)
            plt.setp(blue_kn_g, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(kn_models["blue"]['time'], kn_models["blue"]['sdss_g'](kn_models["blue"]['time']), 'b-',
                    alpha=0.15, linewidth=lc_border)

            blue_kn_r, = ax.plot(kn_models["blue"]['time'], kn_models["blue"]['sdss_r'](kn_models["blue"]['time']),
                                 color="blue", linestyle=r_ls, label="SDSS r", linewidth=lc_width)
            plt.setp(blue_kn_r, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(kn_models["blue"]['time'], kn_models["blue"]['sdss_r'](kn_models["blue"]['time']), 'b-',
                                alpha=0.15, linewidth=lc_border)

            blue_kn_i, = ax.plot(kn_models["blue"]['time'], kn_models["blue"]['sdss_i'](kn_models["blue"]['time']),
                                 color="blue", linestyle=i_ls, label="SDSS i", linewidth=lc_width)
            plt.setp(blue_kn_i, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(kn_models["blue"]['time'], kn_models["blue"]['sdss_i'](kn_models["blue"]['time']), 'b-',
                    alpha=0.15, linewidth=lc_border)

            blue_kn_clear, = ax.plot(kn_models["blue"]['time'], kn_models["blue"]['Clear'](kn_models["blue"]['time']),
                                     color="blue", linestyle=clear_ls, label="Clear", linewidth=lc_width)
            plt.setp(blue_kn_clear, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(kn_models["blue"]['time'], kn_models["blue"]['Clear'](kn_models["blue"]['time']), 'b-',
                    alpha=0.15, linewidth=lc_border)
        else:
            blue_kn_r, = ax.plot(kn_models["blue"]['time'], kn_models["blue"]['sdss_r'](kn_models["blue"]['time']),
                                 color="blue", linestyle=r_ls, label="SDSS r", linewidth=lc_width)
            plt.setp(blue_kn_r, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(kn_models["blue"]['time'], kn_models["blue"]['sdss_r'](kn_models["blue"]['time']), 'b-',
                    alpha=0.15, linewidth=lc_border)




        if not is_poster_plot:
            red_kn_g, = ax.plot(kn_models["red"]['time'], kn_models["red"]['sdss_g'](kn_models["blue"]['time']),
                                color="red", linestyle=g_ls,
                                label="SDSS g", linewidth=lc_width)
            plt.setp(red_kn_g, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(kn_models["red"]['time'], kn_models["red"]['sdss_g'](kn_models["blue"]['time']), 'r-',
                    alpha=0.15, linewidth=lc_border)

            red_kn_r, = ax.plot(kn_models["red"]['time'], kn_models["red"]['sdss_r'](kn_models["blue"]['time']),
                                color="red", linestyle=r_ls,
                                label="SDSS r", linewidth=lc_width)
            plt.setp(red_kn_r, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(kn_models["red"]['time'], kn_models["red"]['sdss_r'](kn_models["blue"]['time']), 'r-',
                    alpha=0.15, linewidth=lc_border)

            red_kn_i, = ax.plot(kn_models["red"]['time'], kn_models["red"]['sdss_i'](kn_models["blue"]['time']),
                                color="red", linestyle=i_ls,
                                label="SDSS i", linewidth=lc_width)
            plt.setp(red_kn_i, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(kn_models["red"]['time'], kn_models["red"]['sdss_i'](kn_models["blue"]['time']), 'r-',
                    alpha=0.15, linewidth=lc_border)

            red_kn_clear, = ax.plot(kn_models["red"]['time'], kn_models["red"]['Clear'](kn_models["blue"]['time']),
                                    color="red", linestyle=clear_ls,
                                    label="Clear", linewidth=lc_width)
            plt.setp(red_kn_clear, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(kn_models["red"]['time'], kn_models["red"]['Clear'](kn_models["blue"]['time']), 'r-',
                    alpha=0.15, linewidth=lc_border)








        if not is_poster_plot:
            swope_sss17a_g, = ax.plot(ss17a_g_time, np.asarray(ss17a_g) - KN_dist_mod, color="black", linestyle=g_ls,
                                      label='SSS17a g', linewidth=lc_width)
            plt.setp(swope_sss17a_g, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(ss17a_g_time, np.asarray(ss17a_g) - KN_dist_mod, 'k-', alpha=0.15, linewidth=lc_border)

            swope_sss17a_r, = ax.plot(ss17a_r_time, np.asarray(ss17a_r) - KN_dist_mod, color="black", linestyle=r_ls,
                                      label='SSS17a r', linewidth=lc_width)
            plt.setp(swope_sss17a_r, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(ss17a_r_time, np.asarray(ss17a_r) - KN_dist_mod, 'k-', alpha=0.15, linewidth=lc_border)

            swope_sss17a_i, = ax.plot(ss17a_i_time, np.asarray(ss17a_i) - KN_dist_mod, color="black", linestyle=i_ls,
                                      label='SSS17a i', linewidth=lc_width)
            plt.setp(swope_sss17a_i, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(ss17a_i_time, np.asarray(ss17a_i) - KN_dist_mod, 'k-', alpha=0.15, linewidth=lc_border)







        if not is_poster_plot:
            off_axis_grb_g, = ax.plot(grb_models["offaxis"]['time'],
                                      grb_models["offaxis"]['sdss_g'](grb_models["offaxis"]['time']), color="purple",
                                      linestyle=g_ls, label="SDSS g", linewidth=lc_width)
            plt.setp(off_axis_grb_g, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(grb_models["offaxis"]['time'], grb_models["offaxis"]['sdss_g'](grb_models["offaxis"]['time']),
                    color="purple", linestyle='-', alpha=0.15, linewidth=lc_border)

            off_axis_grb_r, = ax.plot(grb_models["offaxis"]['time'],
                                      grb_models["offaxis"]['sdss_r'](grb_models["offaxis"]['time']), color="purple",
                                      linestyle=r_ls, label="SDSS r", linewidth=lc_width)
            plt.setp(off_axis_grb_r, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(grb_models["offaxis"]['time'],
                    grb_models["offaxis"]['sdss_r'](grb_models["offaxis"]['time']), color="purple",
                    linestyle=r_ls, alpha=0.15, linewidth=lc_border)

            off_axis_grb_i, = ax.plot(grb_models["offaxis"]['time'],
                                      grb_models["offaxis"]['sdss_i'](grb_models["offaxis"]['time']), color="purple",
                                      linestyle=i_ls, label="SDSS i", linewidth=lc_width)
            plt.setp(off_axis_grb_i, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(grb_models["offaxis"]['time'], grb_models["offaxis"]['sdss_i'](grb_models["offaxis"]['time']),
                    color="purple", linestyle='-', alpha=0.15, linewidth=lc_border)

            off_axis_grb_clear, = ax.plot(grb_models["offaxis"]['time'],
                                          grb_models["offaxis"]['Clear'](grb_models["offaxis"]['time']), color="purple",
                                          linestyle=clear_ls, label="Clear", linewidth=lc_width)
            plt.setp(off_axis_grb_clear, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(grb_models["offaxis"]['time'], grb_models["offaxis"]['Clear'](grb_models["offaxis"]['time']),
                    color="purple", linestyle='-', alpha=0.15, linewidth=lc_border)










        if not is_poster_plot:
            on_axis_grb_g, = ax.plot(grb_models["onaxis"]['time'],
                                     grb_models["onaxis"]['sdss_g'](grb_models["onaxis"]['time']), color="green",
                                     linestyle=g_ls, label="SDSS g", linewidth=lc_width)
            plt.setp(on_axis_grb_g, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(grb_models["onaxis"]['time'], grb_models["onaxis"]['sdss_g'](grb_models["onaxis"]['time']), 'g-',
                    alpha=0.15, linewidth=lc_border)

            on_axis_grb_r, = ax.plot(grb_models["onaxis"]['time'],
                                     grb_models["onaxis"]['sdss_r'](grb_models["onaxis"]['time']), color="green",
                                     linestyle=r_ls, label="SDSS r", linewidth=lc_width)
            plt.setp(on_axis_grb_r, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(grb_models["onaxis"]['time'],
                    grb_models["onaxis"]['sdss_r'](grb_models["onaxis"]['time']), 'g-', alpha=0.15,
                    linewidth=lc_border)

            on_axis_grb_i, = ax.plot(grb_models["onaxis"]['time'],
                                     grb_models["onaxis"]['sdss_i'](grb_models["onaxis"]['time']), color="green",
                                     linestyle=i_ls, label="SDSS i", linewidth=lc_width)
            plt.setp(on_axis_grb_i, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(grb_models["onaxis"]['time'], grb_models["onaxis"]['sdss_i'](grb_models["onaxis"]['time']), 'g-',
                    alpha=0.15, linewidth=lc_border)

            on_axis_grb_clear, = ax.plot(grb_models["onaxis"]['time'],
                                         grb_models["onaxis"]['Clear'](grb_models["onaxis"]['time']), color="green",
                                         linestyle=clear_ls, label="Clear", linewidth=lc_width)
            plt.setp(on_axis_grb_clear, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(grb_models["onaxis"]['time'], grb_models["onaxis"]['Clear'](grb_models["onaxis"]['time']), 'g-',
                    alpha=0.15, linewidth=lc_border)
        else:
            on_axis_grb_r, = ax.plot(grb_models["onaxis"]['time'],
                                     grb_models["onaxis"]['sdss_r'](grb_models["onaxis"]['time']), color="green",
                                     linestyle=r_ls, label="SDSS r", linewidth=lc_width)
            plt.setp(on_axis_grb_r, path_effects=[path_effects.withStroke(linewidth=pe_width, foreground='white')])
            ax.plot(grb_models["onaxis"]['time'],
                    grb_models["onaxis"]['sdss_r'](grb_models["onaxis"]['time']), 'g-', alpha=0.15,
                    linewidth=lc_border)














        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        ha = mpatches.Patch(visible=False)
        hb = mpatches.Patch(visible=False)
        hc = mpatches.Patch(visible=False)
        hd = mpatches.Patch(visible=False)
        he = mpatches.Patch(visible=False)
        hf = mpatches.Patch(visible=False)
        hblank1 = mpatches.Patch(visible=False)
        hblank2 = mpatches.Patch(visible=False)
        hblank3 = mpatches.Patch(visible=False)
        hblank4 = mpatches.Patch(visible=False)
        hblank5 = mpatches.Patch(visible=False)
        hblank6 = mpatches.Patch(visible=False)
        hblank7 = mpatches.Patch(visible=False)
        hblank8 = mpatches.Patch(visible=False)




        if not is_poster_plot:
            l1 = ax.legend(
                # handles=[blue_hdr, blue_kn_g, blue_kn_r, blue_kn_i, blue_kn_clear, hblank1,
                #          onaxis_hdr, on_axis_grb_g, on_axis_grb_r, on_axis_grb_i, on_axis_grb_clear, hblank5,
                #
                #          red_hdr, red_kn_g, red_kn_r, red_kn_i, red_kn_clear,  hblank2,
                #          offaxis_hdr, off_axis_grb_g, off_axis_grb_r, off_axis_grb_i, off_axis_grb_clear, hblank6,
                #
                #          blk_hdr, swope_sss17a_g, swope_sss17a_r, swope_sss17a_i, hblank3, hblank4,
                #          swope_r, nickel_r, thacher_r, kait_clear, lcogt_g, lcogt_i
                #          ],
                handles=[ha, blue_kn_g, blue_kn_r, blue_kn_i, blue_kn_clear, hblank1,
                         hb, on_axis_grb_g, on_axis_grb_r, on_axis_grb_i, on_axis_grb_clear, hblank5, hblank7,

                         hc, red_kn_g, red_kn_r, red_kn_i, red_kn_clear, hblank2,
                         hd, off_axis_grb_g, off_axis_grb_r, off_axis_grb_i, off_axis_grb_clear, hblank6, hblank8,

                         he, swope_sss17a_g, swope_sss17a_r, swope_sss17a_i, hblank3, hblank4,
                         hf, swope_r, nickel_r, kait_clear, thacher_r, lcogt_g, lcogt_i
                         ],

                # "onaxis": "../GRBModels/NewModels/20200811_grb_onaxis/4/grb_0.0000_21.5443_0.2511886.dat",
                # "offaxis": "../GRBModels/NewModels/20200811_grb_offaxis/9/grb_0.2967_21.5443_0.2511886.dat"


                labels=[r'$\mathrm{\mathbf{Blue\:KN}}$', 'SDSS g', 'SDSS r', 'SDSS i', 'Clear', ' ',
                        # r'$\mathrm{\mathbf{GRB\:170817A}}$' + '\n' + r'$\mathrm{\mathbf{on-axis}}$',
                        r'$\mathrm{\mathbf{Fiducial\:SGRB}}$' + '\n' + r'$\mathrm{\mathbf{on-axis}}$',
                        # r'$\mathrm{\mathbf{E_{k,iso}\:2.15e51\:ergs}}$' + '\n' + r'$\mathrm{\mathbf{n=2.5e-1\:cm^{-3}}}$' + '\n' + r'$\mathrm{\mathbf{on-axis}}$',
                        'SDSS g', 'SDSS r', 'SDSS i', 'Clear', ' ', ' ',

                        r'$\mathrm{\mathbf{Red\:KN}}$', 'SDSS g', 'SDSS r', 'SDSS i', 'Clear', ' ',
                        # r'$\mathrm{\mathbf{GRB\:170817A}}$' + '\n' + r'$\mathrm{\mathbf{off-axis}}$',
                        r'$\mathrm{\mathbf{Fiducial\:SGRB}}$' + '\n' + r'$\mathrm{\mathbf{off-axis}}$',
                        # r'$\mathrm{\mathbf{E_{k,iso}\:2.15e51\:ergs}}$' + '\n' + r'$\mathrm{\mathbf{n=2.5e-1\:cm^{-3}}}$' + '\n' + r'$\mathrm{\mathbf{on-axis}}$',
                        'SDSS g', 'SDSS r', 'SDSS i', 'Clear', ' ', ' ',

                        r'$\mathrm{\mathbf{AT\ 2017gfo}}$', 'SDSS g', 'SDSS r', 'SDSS i', ' ', ' ', # Instead of SSS17a
                        r'$\mathrm{\mathbf{Observational}}$' + '\n' + r'$\mathrm{\mathbf{Limits}}$',
                        # r'$\mathrm{\mathbf{Obs\:Limits}}$',
                        'Swope r', "Nickel r'", 'KAIT Clear', 'Thacher r', 'Las Cumbres g', 'Las Cumbres i'],
                ncol=3, loc=1, fontsize=10, handletextpad=0.2,
                # borderpad=0.35,
                # labelspacing=0.35,
                handleheight=1.5,
                borderpad=1.0,
                labelspacing=0.2,
                columnspacing=-0.5,
                framealpha=0.8, edgecolor='k',
                # bbox_to_anchor=(0.95, 1.0),
                bbox_to_anchor=(1.0, 1.0),
                handlelength=4)
        else:
            l1 = ax.legend(

                handles=[ha, blue_kn_r,

                         hc, on_axis_grb_r,

                         hf, swope_r
                         ],

                labels=[r'$\mathrm{\mathbf{Neutron\:Star}}$' + '\n' + r'$\mathrm{\mathbf{Merger}}$', 'SDSS r',

                        r'$\mathrm{\mathbf{Short}}$' + '\n' + r'$\mathrm{\mathbf{GRB}}$', 'SDSS r',

                        r'$\mathrm{\mathbf{Observational}}$' + '\n' + r'$\mathrm{\mathbf{Limits}}$', 'Swope r'],

                ncol=3, loc=1, fontsize=10, handletextpad=0.2,
                handleheight=1.5,
                borderpad=1.0,
                labelspacing=0.2,
                columnspacing=1.0,

                framealpha=0.8, edgecolor='k',
                bbox_to_anchor=(1.0, 1.0),
                handlelength=4)

            for ii, txt in enumerate(l1.get_texts()):
                # if 'KN' in txt.get_text() or 'GRB' in txt.get_text() or 'SSS' in txt.get_text():
                if 'Short' in txt.get_text() or 'Neutron' in txt.get_text():
                    txt.set_ha("center")  # horizontal alignment of text item
                    txt.set_x(80)  # x-position
                    txt.set_y(10)   # y-position
                elif 'Limits' in txt.get_text() or 'Merger' in txt.get_text() or 'GRB' in txt.get_text():
                    txt.set_ha("center")  # horizontal alignment of text item
                    txt.set_x(150)  # x-position
                    txt.set_y(10)  # y-position
                # else:
                #     txt.set_x(40)  # x-position

            for legend_handle in l1.legendHandles:
                legend_handle.set_lw(2.0)

            l1.get_frame().set_linewidth(2.0)

        if not is_poster_plot:
            for ii, txt in enumerate(l1.get_texts()):
                # if 'KN' in txt.get_text() or 'GRB' in txt.get_text() or 'SSS' in txt.get_text():
                if 'KN' in txt.get_text() or 'GRB' in txt.get_text() or 'gfo' in txt.get_text():
                    txt.set_ha("center")  # horizontal alignment of text item
                    txt.set_x(80)  # x-position
                    txt.set_y(10)   # y-position
                elif 'Limits' in txt.get_text():
                    txt.set_ha("center")  # horizontal alignment of text item
                    txt.set_x(150)  # x-position
                    txt.set_y(10)  # y-position
                else:
                    txt.set_x(40)  # x-position

            for legend_handle in l1.legendHandles:
                legend_handle.set_lw(2.0)

            l1.get_frame().set_linewidth(2.0)

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2.0)
            ax.spines[axis].set_zorder(9999)

        ytcks = np.asarray([-31, -29, -27, -25, -23, -21, -19, -17, -15, -13, -11, -9])
        if not is_poster_plot:
            ax.set_yticks(ytcks)
            ax.set_ylim([y_min, y_max])
        else:
            y_min = -29
            ytcks = np.asarray([-29, -27, -25, -23, -21, -19, -17, -15, -13, -11, -9])
            ax.set_yticks(ytcks)
            ax.set_ylim([y_min, y_max])

        ax.set_xlim([0, 12])
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.invert_yaxis()


        # axy2.set_yticklabels([14, 16, 18, 20, 22, 24, 26, 28])

        if not is_poster_plot:
            print("dist mod: %s" % gw0814_dist_mod)
            axy2 = ax.twinx()

            axy2.set_yticks(ytcks)
            axy2.set_ylim([y_min, y_max])
            axy2.set_yticklabels(["6", "8", "10", "12", "14", "16", "18", "20", "22", "24", "26", "28"])
            axy2.tick_params(axis='both', which='major', labelsize=24, length=12.0, width=2)
            axy2.invert_yaxis()
            axy2.set_ylabel("Apparent Magnitude (AB)", fontsize=32, labelpad=9.0)
            # ytcks2 = ytcks+dist_mod
            # ytcks2 = ytcks+dist_mod
            # axy2.set_yticks(ytcks2)
            # axy2.yaxis.set_ticks_position("right")
            # axy2.yaxis.set_label_position("right")

        if not is_poster_plot:
            ax.set_ylabel("Absolute Magnitude (AB)", fontsize=32, labelpad=9.0)
        else:
            ax.set_ylabel("Brightness ➞ (Abs Mag)", fontsize=32, labelpad=9.0)
        ax.set_xlabel("Days From Merger", fontsize=32, labelpad=10.0)
        ax.grid(linestyle=':', color="gray")

        ax.tick_params(axis='both', which='major', labelsize=24, length=12.0, width=2)
        ax.tick_params(axis='both', which='minor', labelsize=24, length=8.0, width=2)


        # ax.add_artist(l1)
        if not is_poster_plot:
            fig.savefig('GW190814_Lightcurves.png', bbox_inches='tight')
        else:
            fig.savefig('GW190814_Lightcurves_poster.png', bbox_inches='tight', transparent=True)
        plt.close('all')
        print("... Done.")


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
    print("Teglon `ComputeModelDetection_GRB` execution time: %s" % duration)
    print("********* end DEBUG ***********\n")
