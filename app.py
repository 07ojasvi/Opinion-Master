import json
import os
import re
from flask import Flask, render_template, redirect, request, session, jsonify
from flask_session import Session
from scripts.sentiments import Sentiments
import nltk
import sqlite3


app=Flask(__name__)
app.secret_key=os.urandom(24)
#app.register_blueprint(second)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

DATABASE_PATH='sql/database.db'

try:
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    nltk.download('stopwords')

    cursor = conn.cursor()
    print("------------------------")
    print("Connected to Database")
    print("------------------------")

except:
    print("Could not connect to database.")

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/home')
def home():
    if 'user_id' in session:
        try:
            cursor.execute("SELECT * FROM reports WHERE email = ?", (session['user_id'],))
            rows = cursor.fetchall()
            conn.commit()
        except:
            print("Could not retreive data")


        return render_template('home.html', len = len(rows), rows=rows)

    else:
        return redirect('/')

@app.route("/login", methods=['GET'])
def login2():
    return render_template('login.html')

@app.route('/login_validation', methods=['POST'])
def login_validation():
    email=request.form.get('email')
    password=request.form.get('password')

    #cursor.execute("""SELECT * from `users` WHERE `email` LIKE '{}' AND `password` LIKE '{}'""".format(email, password))
    try:
        cursor.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
        conn.commit()

    except:
        print("Error. Could not log in.")
    users = cursor.fetchall()
    if len(users)>0:
        session['user_id']=users[0][0]
        print("SessionID---------------------------")
        print(session['user_id'])
        return redirect('/home')
    else:
        return redirect('/login')


@app.route('/add_user', methods=['POST'])
def add_user():
    name=request.form.get('uname')
    email = request.form.get('uemail')
    password = request.form.get('upassword')
    #cursor.execute("""INSERT INTO `users` (`name`,`email`,`password`) VALUES ('{}','{}','{}')""".format(name,email, password))
    cursor.execute("INSERT INTO users(name, email, password) VALUES(?, ?, ?)", (name, email, password))
    conn.commit()
    #cursor.execute("""SELECT * from `users` WHERE `email` LIKE '{}'""".format(email))
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    myuser=cursor.fetchall()
    print(myuser)
    session['user_id']=myuser[0][0]
    return redirect('/home')


@app.route('/getReportDetails', methods=['POST'])
def getReportDetails():
    reportData = []

    cursor.execute("SELECT * FROM reports WHERE email = ?", (session['user_id'],))
    conn.commit()

    rows = cursor.fetchall()
    for row in rows:
        reportData[row[0]]={
            "id": row[0],
            "number": row[1],
            "query": row[2],
            "created": row[3],
            "email": row[4],
        }

    return jsonify(reportData)

@app.route('/logout')
def logout():
    session.pop('user_id')
    return redirect('/')

@app.route('/generateReport', methods=['POST'])
def generateReport():
 
    query = str(request.form.get('query'))
    print(query)

    number = int(request.form.get('number'))
    print(number)

    report = Sentiments(query, number)  
    reportID = report.generateReport()

    redirectURL = '/report/'+reportID

    cursor.execute("INSERT INTO reports(id, number, query, email) VALUES(?, ?, ?, ?)", (reportID, number, query, session['user_id']))
    conn.commit()

    return redirect(redirectURL)


@app.route('/report/<id>', methods=['GET'])
def person(id):

    reportData = []

    try:
        cursor.execute("SELECT * FROM reports WHERE id = ?", (id,))
        rows = cursor.fetchall()
        conn.commit()
    except:
        print("Could not retreive data")

    
    for row in rows:
        print(row)
        number= row[1]
        query= row[2]
        created = row[3]

    return render_template('report.html', reportID=id, queryName=query, createdDate=created, numberOfTweets=number)


if __name__=="__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))




