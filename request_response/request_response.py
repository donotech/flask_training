from flask import Flask, render_template, request
app = Flask(__name__)


@app.route("/")
def hello():
    request.args.g
    return app.send_static_file("first.html")


@app.route("/login")
def login():
    return app.send_static_file("login.html")


@app.route("/buy/<flask_no>")
def buy(flask_no):
    print(flask_no)
    return render_template("buy.html", flaskNumber=flask_no)


if __name__ == "__main__":
    app.run()
