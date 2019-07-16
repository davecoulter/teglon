

class Sky_Distance:

	mean_B_lum_density = 1.98e-2 # in units of (L10 = 10^10 L_B, solar)

	def __init__(self, D1, D2, dCoV, NSIDE):
		self.D1 = D1.value
		self.D2 = D2.value
		self.dCoV = dCoV.value
		self.Sch_L10 = self.dCoV*Sky_Distance.mean_B_lum_density
		self.NSIDE = NSIDE
		
	def __str__(self):
		return "(NSIDE %s) Dist: [%0.4f, %0.4f]; Vol: %0.4f; L10: %0.4f" % (self.NSIDE, self.D1, self.D2, self.dCoV, self.Sch_L10)