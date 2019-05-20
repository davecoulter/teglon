import mysql.connector
from shapely.geometry import Point
import pprint

def QueryDB(query, port='3306', query_data=None):
	results = []
	
	# cnx = mysql.connector.connect(user='prospecktor', password='4Pr0spi3!', host='127.0.0.1', port=port, database='Prospecktor')
	cnx = mysql.connector.connect(user='prospecktor', password='4Pr0spi3!', host='host.docker.internal', port=port, database='Prospecktor')

	cursor = cnx.cursor()
	
	if query_data is not None:
		cursor.execute(query, query_data)
	else:
		cursor.execute(query)
	
	results = [result for result in cursor]
	
	cursor.close()
	cnx.close()
	
	return results


# class glade_galaxy:
# 	def __init__(self,db_result):
# 		self.ID = db_result[0]
# 		self.PGC = db_result[1]
# 		self.Name_GWGC = db_result[2]
# 		self.Name_HyperLEDA = db_result[3]
# 		self.Name_2MASS = db_result[4]
# 		self.Name_SDSS_DR12 = db_result[5]
# 		self.RA = float(db_result[6]) if db_result[6] is not None else db_result[6]
# 		self.Dec = float(db_result[7]) if db_result[7] is not None else db_result[7]
# 		self.dist = float(db_result[8]) if db_result[8] is not None else db_result[8]
# 		self.dist_err = float(db_result[9]) if db_result[9] is not None else db_result[9]
# 		self.z = db_result[10]
# 		self.B = float(db_result[11]) if db_result[11] is not None else db_result[11]
# 		self.B_err = db_result[12]
# 		self.B_abs = db_result[13]
# 		self.J = db_result[14]
# 		self.J_err = db_result[15]
# 		self.H = db_result[16]
# 		self.H_err = db_result[17]
# 		self.K = db_result[18]
# 		self.K_error = db_result[19]
# 		self.flag1 = db_result[20]
# 		self.flag2 = db_result[21]
# 		self.flag3 = db_result[22]
# 		self.B_lum_proxy = (self.dist**2)*10**(-0.4*self.B)
		
# 		# Hack: convert the RA of the galaxy from [0,360] to [-180,180]
# 		g_ra = self.RA
# 		if g_ra > 180:
# 			g_ra = self.RA - 360.
		
# 		self.polygon_point = Point(g_ra, self.Dec)
		
# 	def __repr__(self):
# 		return pprint.pformat(vars(self), indent=4, width=1)