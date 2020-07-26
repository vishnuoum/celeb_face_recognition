from flask import Flask, render_template,request,jsonify
import json
import util
import base64


app = Flask(__name__)
util.load()

@app.route('/')
def sessions():
    
    return render_template('client.html')




@app.route('/file-upload', methods=['POST'])
def predict():
    image=request.files["file"]
    img_b64=base64.b64encode(image.read())
    return json.dumps(util.img_prediction(img_b64))





if __name__ == '__main__':
    # socketio.run(app, debug=True,host="192.168.42.229")
    app.run()
