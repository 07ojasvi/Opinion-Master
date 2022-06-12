import sqlite3

connection = sqlite3.connect('./sql/database.db')

with open('./sql/schema.sql') as f:
	connection.executescript(f.read())

connection.commit()
connection.close()

