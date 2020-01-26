from flask import Flask, render_template, request, make_response
import random
app = Flask(__name__)

flaskDomain = "FlaskDomain1"
myCookie = str(random.random() * 1000)

@app.route("/")
def main_page():
    print(myCookie)
    if request.cookies.get(flaskDomain):
        if request.cookies.get(flaskDomain) == myCookie:
            return render_template("first.html")

    return app.send_static_file("login.html")


@app.route("/login", methods=["GET"])
def login_page():
    return app.send_static_file("login.html")


@app.route("/login", methods=["POST"])
def login_success():
    name = request.form.get('inputName')
    password = request.form.get('inputPassword')

    if name == password:
        resp = make_response(app.send_static_file("first.html"), 200)
        resp.set_cookie(flaskDomain, myCookie)
        return resp
    else:
        return make_response(app.send_static_file("login.html"), 404)


@app.route("/product")
def product():
    if request.cookies.get(flaskDomain):
        cookie = request.cookies.get(flaskDomain)
        print(cookie)
        if cookie == myCookie:
            item = request.args['name']
            return "came with cookie " + str(cookie) + " to buy " + str(item)

    return app.send_static_file("login.html")


if __name__ == "__main__":
    app.run()
