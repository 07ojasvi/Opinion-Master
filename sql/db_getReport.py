#!/bin/python

import sqlite3
DATABASE_PATH='sql/database.db'

try:
        conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM reports WHERE email = ?", ('abhishek@ver.ma',))
        rows = cursor.fetchall()
        conn.commit()
except:
        print("Could not retreive data")

    
for row in rows:
    print(row)
    number= row[1]
    query= row[2]
    created = row[3]