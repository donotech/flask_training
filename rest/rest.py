from flask import Flask,request,abort,json,Response,render_template
from flask_restful import Resource,Api

app = Flask(__name__,template_folder='template')
api = Api(app)
gMap = {}


def fake_db_insert(user, age, address):
    gMap[user] = (user,age,address)


def fake_db_delete(user) :
    del gMap[user]


def fake_db_get(user):
    return gMap[user]


def fake_db_exists(user):
    return user in gMap


class User(Resource):

    def put(self, userName):
        data = request.json
        user = data['user']
        age = data['age']
        address = data['address']
        fake_db_insert(user, age, address)
        return Response(json.dumps({"StatusMessage": "done"}), status=200, mimetype='application/json')

    def post(self, userName):
        data = request.json
        user = data['user']
        age = data['age']
        address = data['address']
        fake_db_insert(user, age, address)
        return Response(json.dumps({"StatusMessage": "done"}), status=200, mimetype='application/json')

    def get(self, userName):
        tup = fake_db_get(userName)
        return Response(json.dumps({"user": tup[0], "age": tup[1], "address": tup[2]}), status=200, mimetype='application/json')

    def delete(self, userName):
        fake_db_delete(userName)


class All(Resource):

    def get(self):
        return Response(json.dumps(gMap), status=200, mimetype='application/json')


api.add_resource(User, '/user/<userName>')
api.add_resource(All, '/all')

if __name__ =="__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)