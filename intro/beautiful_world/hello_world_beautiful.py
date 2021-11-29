from flask import Flask, render_template
from flask import request
app = Flask(__name__)


@app.route("/")
def hello():
    return app.send_static_file("first.html")


@app.route("/login")
def login():
    return app.send_static_file("login.html")


@app.route("/dologin", methods=['POST'])
def do_login():
    user = request.form['username']
    password = request.form['password']
    if user == password:
        return "User validated"
    else:
        return "Invalid user"


@app.route("/search")
def search():
    term1 = request.args.get("term1")
    all_param = str(request.args)
    return "Your search term 1 = " + term1 + " all terms = " + all_param + " request method " + request.method

@app.route("/show")
def show():

    flaskList = ["Flask1", "Flask2", "Flask3", "Flask4 ..."]
    # flaskList = "some sql to read flask list from db"
    # return "The input parameter was " + str(flask_no)
    return render_template("buy2.html", my_collection=flaskList)


@app.route("/buy/<flask_no>")
def buy(flask_no):
    print(flask_no)
    # return "You wanted to buy " + str(flask_no)
    return render_template("buy.html", flaskNumber=flask_no)


if __name__ == "__main__":
    app.run()
