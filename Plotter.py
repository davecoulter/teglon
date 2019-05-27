import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from shapely import geometry
from matplotlib.patches import Polygon



def plot_probability_map(output_filename, 
						 colormap=plt.cm.viridis, 
						 healpix_obj_for_contours=None, 
						 pixels_filled=None,
						 pixels_empty=None,
						 tiles=None, 
						 galaxies=None, 
						 completeness=None,
						 net_prob=None,
						 distance=None,
						 # empty_pixel=False,
						 tile_set=None,
						 linear_rings=None):
	
	print("Plotting `%s`" % output_filename)
	
	fig = plt.figure(figsize=(30,30), dpi=1000)
	ax = fig.add_subplot(111)
	
#     # GW190425 - specific
#     m = Basemap(projection='stere',
#                  lon_0=45.0,
#                  lat_0=20.0,
#                  llcrnrlat=10.0,
#                  urcrnrlat=40.0,
#                  llcrnrlon=30.0,
#                  urcrnrlon=60.0)

	# GW190425_2 - specific
	m = Basemap(projection='moll',lon_0=180.0)

	
	# # Plot contours
	# if healpix_obj_for_contours:
	# 	print("Plotting Contours...")
	# 	cs_outline = m.contour(healpix_obj_for_contours.X,
	# 						   healpix_obj_for_contours.Y,
	# 						   healpix_obj_for_contours.Z,
	# 						   levels=healpix_obj_for_contours.levels,
	# 						   colors='orange',
	# 						   latlon=True,
	# 						   alpha=1.0,
	# 						   linestyles=healpix_obj_for_contours.linestyle,
	# 						   linewidths=1.0)
	

	# Plot pixels
	if pixels_filled:
		print("Plotting `pixels_filled`...")
		
		# Scale colormap
		pixels_probs = [p.prob for p in pixels_filled]
		min_prob = np.min(pixels_probs)
		max_prob = np.max(pixels_probs)  

		fracs = pixels_probs/max_prob
		norm = colors.Normalize(fracs.min(), fracs.max())

		patches = []
		values = []
		for p in pixels_filled:
			patches.append(p.get_patch(m))

			pprob = p.prob
			if p.prob <= 0.0:
				pprob = 1e-18
			values.append(pprob)


		patch_collection = mpl.collections.PatchCollection(np.asarray(patches), cmap=colormap, norm=colors.LogNorm(), alpha=0.75, zorder=9900)
		patch_collection.set_array(np.asarray(values))
		ax.add_collection(patch_collection)
		
	if pixels_empty:
		for p in pixels_empty:
			p.plot(m, ax, edgecolor='k', facecolor='None', linewidth=0.25, alpha=1.0)



	if linear_rings:
		for lr in linear_rings:
			# lrc = list(lr.coords)

			# ra_deg,dec_deg = zip(*[(np.degrees(coord_rad[0]), np.degrees(coord_rad[1])) 
			# 				   for coord_rad in lrc])

			# x2,y2 = m(ra_deg,dec_deg)
			# lat_lons = np.vstack([x2,y2]).transpose()
			# patch = Polygon(lat_lons, linewidth=0.25,edgecolor='k', facecolor='None',zorder=9900)
			# ax.add_patch(patch)
			ra_deg,dec_deg = zip(*[(np.degrees(coord_rad[0]), np.degrees(coord_rad[1])) 
							   for coord_rad in lr.exterior.coords])
		
			x,y = m(ra_deg,dec_deg)
			lat_lons = np.vstack([x,y]).transpose()
			ax.add_patch(Polygon(lat_lons, edgecolor='k', linewidth=0.35, facecolor='None',zorder=9000))
	
	if tiles:
		print("Plotting Tiles...")
		for t in tiles:
			t.plot(m, ax, edgecolor='r', facecolor='None', linewidth=0.25, alpha=1.0,zorder=9300)
	if tile_set:
		for ts in tile_set:
			print("Plotting Tiles: %s..." % ts[0])
			for t in ts[1]:
				t.plot(m, ax, edgecolor=ts[2][0], facecolor=ts[2][1], linewidth=0.25, alpha=1.0,zorder=9999)
		   
	if galaxies:
		print("Plotting Galaxies...")
		for g in galaxies:
			g.plot(m, ax, healpix_obj_for_contours, facecolor='None', edgecolor='k', linewidth=0.1, zorder=9990) # ms=72.0/fig.dpi,

	
		

#     # draw meridians
#     meridians = np.arange(0.,360.,10.)
#     par = m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=18,zorder=-1,color='gray', linewidth=2.0)

#     # draw parallels
#     parallels = np.arange(-90.,90.,10.)
#     par = m.drawparallels(parallels,labels=[0,1,0,0],fontsize=18,zorder=-1,color='gray', linewidth=2.0, xoffset=230000)


	m.drawparallels(np.arange(-90.,91.,30.),labels=[True,True,False,False],dashes=[2,2],linewidth=0.5)
	m.drawmeridians(np.arange(-180.,181.,60.),labels=[False,False,False,False],dashes=[2,2],linewidth=0.5)

	
	if pixels_filled:  # and not empty_pixel
		top_left = 0.95
		delta_y = 0.04
		
		if distance:
			ax.annotate('%s Mpc' % distance, xy=(0.05, top_left), xycoords='axes fraction', fontsize=24)
		if completeness:
			ax.annotate('Comp: %0.0f%%' % completeness, xy=(0.05, top_left-delta_y), xycoords='axes fraction', fontsize=24)
		if net_prob:
			ax.annotate('Prob: %0.0f%%' % net_prob, xy=(0.05, top_left-2*delta_y), xycoords='axes fraction', fontsize=24)


		# ****************************************************************************************************

		sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
		sm.set_array([]) # can be an empty list
		tks = np.linspace(min_prob, max_prob, 5)
		tks_strings = []
		for t in tks:
			if t*100 < 10:
				tks_strings.append('%0.2f  ' % (t*100))
			else:
				tks_strings.append('%0.2f' % (t*100))

		cb = fig.colorbar(sm, ax=ax, ticks=[0.2,0.4,0.6,0.8,1.0], orientation='horizontal', fraction=0.08951, pad=0.04)
		cb.ax.set_yticklabels(tks_strings, fontsize=18)
		cb.set_label("% per Pixel", fontsize=24, labelpad=15.0)
		cb.outline.set_linewidth(2)
		# ****************************************************************************************************
	
	
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(2.0)

#     ax.set_xlabel("R.A.",fontsize=36, labelpad=40)
#     ax.set_ylabel("Dec",fontsize=36, labelpad=30)

	ax.invert_xaxis()
	
	fig.savefig('%s.png' % output_filename,bbox_inches='tight') #,dpi=840
	plt.close('all')
	print("... Done.")




def plot_probability_map_2(output_filename, 
						 central_lon,
						 central_lat,
						 lower_left_corner_lat,
						 lower_left_corner_lon,
						 upper_right_corner_lat,
						 upper_right_corner_lon,
						 colormap=plt.cm.viridis, 
						 healpix_obj_for_contours=None, 
						 pixels=None,
						 tiles=None, 
						 galaxies=None, 
						 completeness=None,
						 net_prob=None,
						 distance=None,
						 empty_pixel=False,
						 tile_set=None):
	
	print("Plotting `%s`" % output_filename)
	
	fig = plt.figure(figsize=(30,30))
	ax = fig.add_subplot(111)
	
	# GW190425 - specific
	m = Basemap(projection='stere',
				 lon_0=central_lon,
				 lat_0=central_lat,
				 llcrnrlat=lower_left_corner_lat,
				 urcrnrlat=upper_right_corner_lat,
				 llcrnrlon=lower_left_corner_lon,
				 urcrnrlon=upper_right_corner_lon)

	# GW190425_2 - specific
	# m = Basemap(projection='moll',lon_0=180.0)

	
	# Plot contours
	# if healpix_obj_for_contours:
	# 	print("Plotting Contours...")
	# 	cs_outline = m.contour(healpix_obj_for_contours.X,
	# 						   healpix_obj_for_contours.Y,
	# 						   healpix_obj_for_contours.Z,
	# 						   levels=healpix_obj_for_contours.levels,
	# 						   colors='orange',
	# 						   latlon=True,
	# 						   alpha=1.0,
	# 						   linestyles=healpix_obj_for_contours.linestyle,
	# 						   linewidths=1.0)
	

	# Plot pixels
	if pixels:
		print("Plotting Pixels...")
		
		if not empty_pixel:
			# Scale colormap
			pixels_probs = [p.prob for p in pixels]
			min_prob = np.min(pixels_probs)
			max_prob = np.max(pixels_probs)  

			fracs = pixels_probs/max_prob
			norm = colors.Normalize(fracs.min(), fracs.max())

			patches = []
			values = []
			for p in pixels:
				patches.append(p.get_patch(m))

				pprob = p.prob
				if p.prob <= 0.0:
					pprob = 1e-18
				values.append(pprob)


			patch_collection = mpl.collections.PatchCollection(np.asarray(patches), cmap=colormap, norm=colors.LogNorm(), alpha=0.75)
			patch_collection.set_array(np.asarray(values))
			ax.add_collection(patch_collection)
			
			
		else:
			for p in pixels:
				p.plot(m, ax, edgecolor='k', facecolor='None', linewidth=0.25, alpha=0.35)
	
	if tiles:
		print("Plotting Tiles...")
		for t in tiles:
			t.plot(m, ax, edgecolor='r', facecolor='None', linewidth=0.25, alpha=1.0,zorder=9999)
	if tile_set:
		for ts in tile_set:
			print("Plotting Tiles: %s..." % ts[0])
			for t in ts[1]:
				t.plot(m, ax, edgecolor=ts[2][0], facecolor=ts[2][1], linewidth=0.25, alpha=1.0,zorder=9999)
		   
	if galaxies:
		print("Plotting Galaxies...")
		for g in galaxies:
			g.plot(m, ax, healpix_obj_for_contours, facecolor='None', edgecolor='crimson', linewidth=1.0)
		

#     # draw meridians
#     meridians = np.arange(0.,360.,10.)
#     par = m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=18,zorder=-1,color='gray', linewidth=2.0)

#     # draw parallels
#     parallels = np.arange(-90.,90.,10.)
#     par = m.drawparallels(parallels,labels=[0,1,0,0],fontsize=18,zorder=-1,color='gray', linewidth=2.0, xoffset=230000)


	m.drawparallels(np.arange(-90.,91.,30.),labels=[True,True,False,False],dashes=[2,2],linewidth=0.5)
	m.drawmeridians(np.arange(-180.,181.,60.),labels=[False,False,False,False],dashes=[2,2],linewidth=0.5)

	
	if pixels and not empty_pixel:
		top_left = 0.95
		delta_y = 0.04
		
		if distance:
			ax.annotate('%s Mpc' % distance, xy=(0.05, top_left), xycoords='axes fraction', fontsize=24)
		if completeness:
			ax.annotate('Comp: %0.0f%%' % completeness, xy=(0.05, top_left-delta_y), xycoords='axes fraction', fontsize=24)
		if net_prob:
			ax.annotate('Prob: %0.0f%%' % net_prob, xy=(0.05, top_left-2*delta_y), xycoords='axes fraction', fontsize=24)


		# ****************************************************************************************************

		sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
		sm.set_array([]) # can be an empty list
		tks = np.linspace(min_prob, max_prob, 5)
		tks_strings = []
		for t in tks:
			if t*100 < 10:
				tks_strings.append('%0.2f  ' % (t*100))
			else:
				tks_strings.append('%0.2f' % (t*100))

		cb = fig.colorbar(sm, ax=ax, ticks=[0.2,0.4,0.6,0.8,1.0], orientation='horizontal', fraction=0.08951, pad=0.04)
		cb.ax.set_yticklabels(tks_strings, fontsize=18)
		cb.set_label("% per Pixel", fontsize=24, labelpad=15.0)
		cb.outline.set_linewidth(2)
		# ****************************************************************************************************
	
	
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(2.0)

#     ax.set_xlabel("R.A.",fontsize=36, labelpad=40)
#     ax.set_ylabel("Dec",fontsize=36, labelpad=30)

	ax.invert_xaxis()
	
	fig.savefig('%s.png' % output_filename,bbox_inches='tight',dpi=280)
	plt.close('all')
	print("... Done.")