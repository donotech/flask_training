from flask import Flask
import pickle
model = pickle.load(open("model.pkl"))
app = Flask(__name__)


@app.route("/")
def index():
    return "<html><head>My Flask Page</head><body><p>Loren Ipsum<br><h2> My first HTML Flask Page</h2></body></html>"


@app.route("/predict")
def predict():
    # get the form parameters
    # create np array
    # predicted_value model.predict(array)
    # return predicted_value
    pass

@app.route("/hello")
def hello():
    return "Hello World!"


@app.route("/greetings")
@app.route("/greetings/")
@app.route("/greetings/<name>")
def greetings(name="Anonymous"):
    return "Hello " + name + " ! "


@app.route("/maps")
def hello_map():
    return "This is my maps page"


@app.route("/myhtml")
def hello_google_page():
    return "<html><body><h1>Hello HTML! </h1>" \
           "<a href=\"https://google.com\">click to google</a></body></html>"


if __name__ == "__main__":
    app.run(port=8081)
