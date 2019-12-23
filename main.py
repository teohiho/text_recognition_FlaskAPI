from flask import Flask, request, jsonify
import base64
from PIL import Image
import io
import cv2
import numpy as np
from pageSegmentation import pageSegmentation
from pageSegmentation import pageSegmentationFourPoint
from spellchecker import spellchecker 
from io import BytesIO
from flask import send_file

app = Flask(__name__)


@app.route("/")
def hello():
    return "Welcome to machine learning model APIs!"

@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    # print("json_ nè: " + str(json_))
    stringToRGB(json_[0]["uri"])

    imgSegmentation = pageSegmentation()
    # imgSegmentation = Image.fromarray(imgSegmentation) #convert numpy image to image
    # processed_string = base64.b64encode(imgSegmentation) #convert image to base64

    if(imgSegmentation != 'cut'):
        pil_img = Image.fromarray(imgSegmentation)
        buff = BytesIO()
        pil_img.save(buff, format="JPEG")
        processed_string = base64.b64encode(buff.getvalue()).decode("utf-8")

        spellchecker()
        textRecognition = readListToTextFile("result-spell-checked.txt", mode='r')

        print("textRecognition:"+ str(textRecognition))
        prediction={
            "imgScan" : str(processed_string),
            "textRecognition" : str(textRecognition)
        }

        #########
        foReChe = open("result-spell-checked.txt", "r+")
        deleteContentFile(foReChe)
        foRe = open("result.txt", "r+")
        deleteContentFile(foRe)
        #########
    elif(imgSegmentation == 'cut'):
        prediction={
            "imgScan" : '',
            "textRecognition" : ''
        }
    
    return jsonify({'prediction': prediction}), 201

@app.route('/predictfourpoints', methods=['POST'])
def predictfourpoints():
    json_ = request.json
    stringToRGB(json_[0]["uri"])
    print("json_[0][x1]: " + str(json_[0]["x1"]))
    print("json_[0][x2]: " + str(type(json_[0]["x2"])))
    imgSegmentation = pageSegmentationFourPoint(
        json_[0]["x1"], 
        json_[0]["x2"],
        json_[0]["x3"],
        json_[0]["x4"],
        json_[0]["y1"],
        json_[0]["y2"],
        json_[0]["y3"],
        json_[0]["y4"],       
    )

    pil_img = Image.fromarray(imgSegmentation)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    processed_string = base64.b64encode(buff.getvalue()).decode("utf-8")

    spellchecker()
    textRecognition = readListToTextFile("result-spell-checked.txt", mode='r')

    # print("textRecognition:"+ str(textRecognition))
    prediction={
        "imgScan" : str(processed_string),
        "textRecognition" : str(textRecognition)
    }

    #########
    foReChe = open("result-spell-checked.txt", "r+")
    deleteContentFile(foReChe)
    foRe = open("result.txt", "r+")
    deleteContentFile(foRe)
    #########
   
    
    return jsonify({'prediction': prediction}), 201


def get_request_data():
    return (
        request.args
        or request.form
        or request.get_json(force=True, silent=True)
        or request.data
        or request.files['file']
    )

# Take in base64 string and return cv image
def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    # image = Image.open(io.BytesIO(imgdata))
    # return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    filename = '/home/dell/Documents/TEO/flack/Text-Recognition-Backend/images/some_image.jpg'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(imgdata)


def readListToTextFile(filePath, mode='r'):
    f = open(filePath, mode)
    return f.read()

def deleteContentFile(f): 
    d = f.readlines()
    f.seek(0)
    for i in d:
        f.truncate()
        f.seek(0)
    f.close()

if __name__ == '__main__':
    app.run(debug=True)
    