from flask import Flask
app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/more")
def more_hello():
    return "More Hello!"


@app.route("/personalized")
def hello_personalized():
    return " Hello please provide your name after url"


@app.route("/personalized/<name>")
def hello_me(name):
    return " Hello " + str(name) + " !!! "


def dangling1():
    return "I will route later"


app.add_url_rule("/dangling", "dangling", dangling1)


@app.route("/beautiful")
def beautify():
    return "<html><head>Flask Training</head><body><h1>Hello World!</h1></body></html>"


if __name__ == "__main__":
    app.run()
