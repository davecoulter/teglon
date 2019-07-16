import mysql.connector
from shapely.geometry import Point
import pprint
import os
from configparser import RawConfigParser

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
configFile = os.path.join(__location__, 'Settings.ini')

config = RawConfigParser()
config.read(configFile)


db_name = config.get('database', 'DATABASE_NAME')
db_user = config.get('database', 'DATABASE_USER')
db_pwd = config.get('database', 'DATABASE_PASSWORD')
db_host = config.get('database', 'DATABASE_HOST')
db_port = config.get('database', 'DATABASE_PORT')

def QueryDB(query, port='3306', query_data=None):
	results = []

	cnx = mysql.connector.connect(user=db_user, password=db_pwd, host=db_host, port=db_port, database=db_name)

	cursor = cnx.cursor()
	
	if query_data is not None:
		print("this")
		cursor.execute(query, query_data)
	else:
		print("that")
		cursor.execute(query)
	
	results = cursor.fetchall()
	
	cursor.close()
	cnx.close()
	
	return results
