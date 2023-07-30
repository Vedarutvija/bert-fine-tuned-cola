from elasticsearch import Elasticsearch
from chat import process_input


from flask import Flask, render_template, request, jsonify
from chat import process_input
app = Flask(__name__)
# Create an Elasticsearch client



@app.get("/")
def index_get():
    return render_template("base.html")
@app.route("/course")
def course():
    return render_template('course.html')
@app.route("/about")
def about():
    return render_template('about.html')
@app.route("/gallery")
def gallery():
    return render_template('gallery.html')
@app.route("/contact")
def contact():
    return render_template('contact.html')
@app.post("/predict")
def predict():
    text = request.get_json()["message"]
    response = process_input(text)
    message = {"answer": response}
    return message
if __name__ == "__main__":
    app.run(debug=True)

