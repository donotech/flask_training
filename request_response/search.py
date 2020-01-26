from flask import Flask, render_template, request
app = Flask(__name__)


@app.route("/search", methods=["GET"])
def search_page():
    if len(request.args) > 0:
        print(request.args)
        searchTerm = request.args['searchText']
        return "you have searched with the following term " + str(searchTerm)
    else:
        return app.send_static_file("search.html")


if __name__ == "__main__":
    app.run()
