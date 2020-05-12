from collections import OrderedDict
import csv
from scipy.interpolate import interp1d

from Tile import *
from Pixel_Element import *


class AAOmega_Galaxy():
    def __init__(self, galaxy_id, ra, dec, prob_galaxy, z, kron_r, pix_index, required_exps, efficiency_func,
                 prob_fraction = 0.0, z_photErr = 0.0, synth_B=0.0):
        self.galaxy_id = galaxy_id
        self.ra = ra
        self.dec = dec
        self.prob_fraction = prob_fraction
        self.prob_galaxy = prob_galaxy
        self.z = z
        self.kron_r = kron_r
        self.synth_B = synth_B
        self.pix_index = pix_index
        self.efficiency_func = efficiency_func
        self.num_exps = 0
        self.available = True
        self.required_exps = required_exps
        self.z_photErr = z_photErr

    def compute_weight(self, num_exps):
        MIN_EXP = 40  # minutes
        total_exp_time = num_exps * MIN_EXP

        efficiency = 1.0
        if self.kron_r > 20.0:
            efficiency = self.efficiency_func(total_exp_time)

        metric = efficiency * self.prob_galaxy * self.prob_fraction
        return metric

    def increment_exps(self, num_exps):
        self.num_exps += num_exps
        self.available = self.num_exps < self.required_exps


class AAOmega_Pixel(Pixel_Element):
    def __init__(self, index, nside, prob, contained_galaxies, pixel_id=None, mean_dist=None, stddev_dist=None):
        Pixel_Element.__init__(self, index, nside, prob, pixel_id, mean_dist, stddev_dist)
        self.contained_galaxies = contained_galaxies

    def get_available_galaxies_by_multiplicity(self, N):
        for galaxy_id, galaxy in self.contained_galaxies.items():
            if galaxy.available and galaxy.required_exps == N:
                yield galaxy


class AAOmega_Tile(Tile):
    def __init__(self, central_ra_deg, central_dec_deg, nside, radius, all_pixels, num_exposures, tile_num):
        Tile.__init__(self, central_ra_deg, central_dec_deg, width=None, height=None, nside=nside, radius=radius)
        self.num_exposures = num_exposures
        self.tile_num = tile_num

        # Initializes galaxies and pixels for tile
        self.contained_pixels_dict = {}
        self.galaxy_pixel_map = {}
        for epi in self.enclosed_pixel_indices:
            if epi in all_pixels:
                pixel = all_pixels[epi]
                self.contained_pixels_dict[epi] = pixel

                for g_id, gal in pixel.contained_galaxies.items():
                    self.galaxy_pixel_map[g_id] = epi

    def calculate_efficiency(self):
        total_prob = 0.0
        total_num_galxies = 0
        gal_ids = []
        total_fibers = 370 * self.num_exposures

        if self.num_exposures == 1:
            # get N=1 list
            n1_galaxies = []
            n1_weights = []

            # Flatten galaxies into list and compute weights
            for pix_index, pixel in self.contained_pixels_dict.items():
                gals = list(pixel.get_available_galaxies_by_multiplicity(1))
                n1_galaxies += gals
                for g in gals:
                    n1_weights.append(g.compute_weight(1))

            # Sort by weight descending
            ordered_indices_n1 = (-np.asarray(n1_weights)).argsort()
            top_galaxies_n1 = list(np.asarray(n1_galaxies)[ordered_indices_n1])

            # Create dictionary of galaxy id and multiplicity of observation
            ordered_galaxies = OrderedDict()
            for g in top_galaxies_n1:
                if g.galaxy_id not in ordered_galaxies:
                    ordered_galaxies[g.galaxy_id] = 0
                ordered_galaxies[g.galaxy_id] += 1

            # Only take as many as we have fibers for
            final_sample = []
            final_count = 0
            for gal_id, multiplicity in ordered_galaxies.items():
                if (final_count + multiplicity) <= total_fibers:
                    final_sample.append((gal_id, multiplicity))
                    final_count += multiplicity
                else:
                    continue

            # Compute final statistics
            for s in final_sample:
                total_num_galxies += 1

                gal_id = s[0]
                num_exposures = s[1]

                pixel_index = self.galaxy_pixel_map[gal_id]
                galaxy = self.contained_pixels_dict[pixel_index].contained_galaxies[gal_id]
                galaxy.increment_exps(num_exposures)

                total_prob += galaxy.prob_fraction

                gal_ids.append(gal_id)

        elif self.num_exposures == 2:

            n1_galaxies = []
            n1_weights = []
            for pix_index, pixel in self.contained_pixels_dict.items():
                gals = list(pixel.get_available_galaxies_by_multiplicity(1))
                n1_galaxies += gals
                for g in gals:
                    n1_weights.append(g.compute_weight(1))

            ordered_indices_n1 = (-np.asarray(n1_weights)).argsort()
            top_galaxies_n1 = list(np.asarray(n1_galaxies)[ordered_indices_n1])
            top_weights_n1 = list(np.asarray(n1_weights)[ordered_indices_n1])


            n2_galaxies = []
            n2_weights = []
            for pix_index, pixel in self.contained_pixels_dict.items():
                gals = list(pixel.get_available_galaxies_by_multiplicity(2))
                n2_galaxies += gals
                for g in gals:
                    n2_weights.append(g.compute_weight(2))

            ordered_indices_n2 = (-np.asarray(n2_weights)).argsort()
            top_galaxies_n2 = list(np.asarray(n2_galaxies)[ordered_indices_n2])
            top_weights_n2 = list(np.asarray(n2_weights)[ordered_indices_n2])

            for i, n2 in enumerate(top_weights_n2):

                found = False

                for j, (w1, w2) in enumerate(zip(top_weights_n1[:-1], top_weights_n1[1:])):
                    combined_weight = w1 + w2

                    if w1 == 0.0:
                        continue
                    elif n2 > combined_weight:
                        top_weights_n1.insert(j, top_weights_n2[i])
                        top_weights_n1.insert(j + 1, 0.0) # place holder

                        top_galaxies_n1.insert(j, top_galaxies_n2[i])
                        top_galaxies_n1.insert(j + 1, top_galaxies_n2[i])

                        found = True
                        break

                if not found:
                    top_galaxies_n1.append(top_galaxies_n2[i])
                    top_galaxies_n1.append(top_galaxies_n2[i])

                    top_weights_n1.append(top_weights_n2[i])
                    top_weights_n1.append(0.0)

            ordered_galaxies = OrderedDict()
            for g in top_galaxies_n1:
                if g.galaxy_id not in ordered_galaxies:
                    ordered_galaxies[g.galaxy_id] = 0
                ordered_galaxies[g.galaxy_id] += 1

            final_sample = []
            final_count = 0
            for gal_id, multiplicity in ordered_galaxies.items():
                if (final_count + multiplicity) <= total_fibers:
                    final_sample.append((gal_id, multiplicity))
                    final_count += multiplicity
                else:
                    continue

            for s in final_sample:
                total_num_galxies += 1

                gal_id = s[0]
                num_exposures = s[1]

                pixel_index = self.galaxy_pixel_map[gal_id]
                galaxy = self.contained_pixels_dict[pixel_index].contained_galaxies[gal_id]
                total_prob += galaxy.prob_fraction
                galaxy.increment_exps(num_exposures)

                gal_ids.append(gal_id)

        elif self.num_exposures == 3:

            n1_galaxies = []
            n1_weights = []
            for pix_index, pixel in self.contained_pixels_dict.items():
                gals = list(pixel.get_available_galaxies_by_multiplicity(1))
                n1_galaxies += gals
                for g in gals:
                    n1_weights.append(g.compute_weight(1))

            ordered_indices_n1 = (-np.asarray(n1_weights)).argsort()
            top_galaxies_n1 = list(np.asarray(n1_galaxies)[ordered_indices_n1])
            top_weights_n1 = list(np.asarray(n1_weights)[ordered_indices_n1])

            # get entire N=2 list -- assuming fewer than 3x370 galaxies in this bin...
            n2_galaxies = []
            n2_weights = []
            for pix_index, pixel in self.contained_pixels_dict.items():
                gals = list(pixel.get_available_galaxies_by_multiplicity(2))
                n2_galaxies += gals
                for g in gals:
                    n2_weights.append(g.compute_weight(2))

            ordered_indices_n2 = (-np.asarray(n2_weights)).argsort()
            top_galaxies_n2 = list(np.asarray(n2_galaxies)[ordered_indices_n2])
            top_weights_n2 = list(np.asarray(n2_weights)[ordered_indices_n2])

            # get entire N=3 list -- assuming fewer than 3x370 galaxies in this bin...
            n3_galaxies = []
            n3_weights = []
            for pix_index, pixel in self.contained_pixels_dict.items():
                gals = list(pixel.get_available_galaxies_by_multiplicity(3))
                n3_galaxies += gals
                for g in gals:
                    n3_weights.append(g.compute_weight(3))

            ordered_indices_n3 = (-np.asarray(n3_weights)).argsort()
            top_galaxies_n3 = list(np.asarray(n3_galaxies)[ordered_indices_n3])
            top_weights_n3 = list(np.asarray(n3_weights)[ordered_indices_n3])

            # bin running_top_galaxies by twos and check if any N=2 galaxy is > any two N=1 galaxies
            for i, n2 in enumerate(top_weights_n2):

                found = False

                for j, (w1, w2) in enumerate(zip(top_weights_n1[:-1], top_weights_n1[1:])):
                    combined_weight = w1 + w2

                    if w1 == 0.0:
                        continue
                    elif n2 > combined_weight:
                        top_weights_n1.insert(j, top_weights_n2[i])
                        top_weights_n1.insert(j + 1, 0.0) # place holder

                        top_galaxies_n1.insert(j, top_galaxies_n2[i])
                        top_galaxies_n1.insert(j + 1, top_galaxies_n2[i])

                        found = True
                        break

                if not found:
                    top_galaxies_n1.append(top_galaxies_n2[i])
                    top_galaxies_n1.append(top_galaxies_n2[i])

                    top_weights_n1.append(top_weights_n2[i])
                    top_weights_n1.append(0.0)

            # bin running_top_galaxies by twos and check if any N=2 galaxy is > any two N=1 galaxies
            for i, n3 in enumerate(top_weights_n3):

                found = False

                for j, (w1, w2, w3) in enumerate(zip(top_weights_n1[:-2], top_weights_n1[1:-1], top_weights_n1[2:])):
                    combined_weight = w1 + w2 + w3
                    if w1 == 0.0:
                        # Don't check vs the w2 placeholder
                        continue
                    elif n3 > combined_weight:

                        top_weights_n1.insert(j, top_weights_n3[i])
                        top_weights_n1.insert(j + 1, 0.0) # place holder
                        top_weights_n1.insert(j + 2, 0.0)  # place holder

                        top_galaxies_n1.insert(j, top_galaxies_n3[i])
                        top_galaxies_n1.insert(j + 1, top_galaxies_n3[i])
                        top_galaxies_n1.insert(j + 2, top_galaxies_n3[i])

                        found = True
                        break

                if not found:
                    top_galaxies_n1.append(top_galaxies_n3[i])
                    top_galaxies_n1.append(top_galaxies_n3[i])
                    top_galaxies_n1.append(top_galaxies_n3[i])

                    top_weights_n1.append(top_weights_n3[i])
                    top_weights_n1.append(0.0)
                    top_weights_n1.append(0.0)

            # Get final list.
            ordered_galaxies = OrderedDict()
            for g in top_galaxies_n1:
                if g.galaxy_id not in ordered_galaxies:
                    ordered_galaxies[g.galaxy_id] = 0
                ordered_galaxies[g.galaxy_id] += 1

            # pprint.pprint(ordered_galaxies)
            final_sample = []
            final_count = 0
            for gal_id, multiplicity in ordered_galaxies.items():
                if (final_count + multiplicity) <= total_fibers:
                    final_sample.append((gal_id, multiplicity))
                    final_count += multiplicity
                else:
                    continue

            for s in final_sample:
                total_num_galxies += 1

                gal_id = s[0]
                num_exposures = s[1]

                pixel_index = self.galaxy_pixel_map[gal_id]
                galaxy = self.contained_pixels_dict[pixel_index].contained_galaxies[gal_id]
                total_prob += galaxy.prob_fraction
                galaxy.increment_exps(num_exposures)

                gal_ids.append(gal_id)

        else:
            raise Exception("Too many exposures!")

        return total_num_galxies, total_prob, gal_ids