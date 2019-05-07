import numpy as np

def distance_metric_squared(x_a, y_a, x_b, y_b, spacing_x, spacing_y):
	"""Find the squared distance between two points, using a cartesian norm, 
	which each dimension scaled by a `spacing` variable"""
	return ((x_a - x_b)/spacing_x)**2 \
		  + ((y_a - y_b)/spacing_y)**2

def rbf_interpolate(x_in, y_in, z_in, x_out, y_out, spacing_x, spacing_y):
	"""Interpolate z_in(x_in, y_in) to the points x_out, y_out, 
	using a radial basis function and a cartesian metric
	
	Inputs
	------
		x_in : 1D numpy array
			x coordinates of input sample points
		y_in : 1D numpy array
			y coordinates of input sample points
		z_in : 1D numpy array
			value of target function at sampled points
			
		x_out : 1D numpy array
			x coordinates of points you would like to interpolate at
		y_out : 1D numpy array
			y coordinates of points you would like to interpolate at
			
		spacing_x : float
			The desired gaussian kernel size in the y direction
		spacing_y : float
			The desired gaussian kernel size in the y direction

	
	Returns
	-------
		zz_out : 2d numpy array
			the interpolated value at every combination of x_out and y_out
			
	Notes
	-----
		You are passing 1D arrays for x_out and y_out; zz_out is then interpolate
		to a 2D grid of all combinations of x_out[i] and y_out[j] for all i,j
		 - this effectively evaluates the interpolation at the 
		 coordinates given by a np.meshgrid of x_out and y_out
	"""



	zz_out = np.empty((y_out.size, x_out.size))
	for j in range(x_out.size):
		for i in range(y_out.size):
			weights = np.exp(-.5*distance_metric_squared(x_in, y_in,
														 x_out[j], y_out[i],
														 spacing_x, spacing_y))
			weights /= weights.sum()
			zz_out[i,j] = (weights * z_in).sum()
	
	return zz_out