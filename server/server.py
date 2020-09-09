from flask import Flask, request, jsonify
import util
app= Flask(__name__)

@app.route("/classify_image", methods=["GET","POST"])
def classify_image():
    image_data = request.form['image_data']
    response = jsonify(util.classify_image(image_data,None))

    response.headers.add('Access-Control-Allow-origin','*')

    return response

if __name__ == "__main__":
    print("starting python flask server.....")
    util.load_artifacts()
    app.run(port=5000)