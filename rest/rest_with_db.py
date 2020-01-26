from flask import Flask, request, abort, json, Response, render_template
from flask_restful import Resource, Api
import mysql.connector

app = Flask(__name__, template_folder='template')
api = Api(app)

dbConfig = {
    "user": "rest",
    "password": "rest433",
    "host": "localhost",
    "database": "restdb",
    "raise_on_warnings": True
}


def db_insert(user, age, address):
    conn = mysql.connector.connect(**dbConfig)
    sql = "insert into users (name, age, address) values(%s, %s, %s)"
    params = (user, age, address,)
    cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit()
    cur.close()
    conn.close()


def db_update(user, age, address):
    conn = mysql.connector.connect(**dbConfig)
    sql = "update users set age = %s, address = %s where name = %s"
    params = (age, address, user)
    cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit()
    cur.close()
    conn.close()


def db_delete(user):
    conn = mysql.connector.connect(**dbConfig)
    sql = "delete from users where name = '" + user + "'"
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    cur.close()
    conn.close()


def db_get(user):
    conn = mysql.connector.connect(**dbConfig)
    sql = "select * from users where name = '" + user + "'"
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchone()
    cur.close()
    conn.close()
    return rows


def db_all():
    conn = mysql.connector.connect(**dbConfig)
    sql = "select * from users"
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


class User(Resource):

    def put(self, userName):
        data = request.json
        user = data['user']
        age = data['age']
        address = data['address']
        db_update(user, age, address)
        return Response(json.dumps({"StatusMessage": "done"}), status=200, mimetype='application/json')

    def post(self, userName):
        data = request.json
        user = data['user']
        age = data['age']
        address = data['address']
        db_insert(user, age, address)
        return Response(json.dumps({"StatusMessage": "done"}), status=200, mimetype='application/json')


    def get(self, userName):
        tup = db_get(userName)
        return Response(json.dumps({"user": tup[0], "age": tup[1], "address": tup[2]}), status=200,
                        mimetype='application/json')

    def delete(self, userName):
        db_delete(userName)


class All(Resource):

    def get(self):
        rows = db_all()
        return Response(json.dumps(rows), status=200, mimetype='application/json')


api.add_resource(User, '/user')
api.add_resource(User, '/user/<userName>')
api.add_resource(All, '/all')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
