from flask import Flask, request
app = Flask(__name__)


@app.route("/login", methods=["GET"])
def login_page():
    return app.send_static_file("login.html")


@app.route("/login", methods=["POST"])
def login_success():
    name = request.form.get('inputName')
    password = request.form.get('inputPassword')

    if name == password:
        return "login success"
    else:
        return "login failed"


if __name__ == "__main__":
    app.run()
