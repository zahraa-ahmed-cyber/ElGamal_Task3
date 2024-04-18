from ultralytics import YOLO
from flask import request, Flask, jsonify
from waitress import serve
from PIL import Image

app = Flask(__name__)

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    with open("index.html") as file:
        return file.read()


@app.route("/detect", methods=["POST"])
def detect():

    buf = request.files["image_file"]
    boxes = detect_objects_on_image(buf.stream)
    return jsonify(boxes)


def detect_objects_on_image(buf):

    model = YOLO("best.pt")
    results = model.predict(Image.open(buf))
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
            round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
            x1, y1, x2, y2, result.names[class_id], prob
        ])
    return output


serve(app, host='0.0.0.0', port=8080)


# from inference_sdk import InferenceHTTPClient

# CLIENT = InferenceHTTPClient(
#   api_url="https://detect.roboflow.com",
#   api_key="jJmtIG4Zzq6GPJV5b4lM"
# )

# from flask import request, Flask, jsonify
# from waitress import serve
# import base64
# import pprint

# app = Flask(__name__)

# @app.route("/")
# def root():
#   """
#   Site main page handler function.
#   :return: Content of index.html file
#   """
#   with open("index.html") as file:
#     return file.read()


# @app.route("/detect", methods=["POST"])
# def detect():
#   """
#   Handler of /detect POST endpoint
#   Receives uploaded file with a name "image_file", infers objects using Roboflow API,
#   and returns an array of bounding boxes.
#   :return: a JSON array of objects bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
#   """
#   buf = request.files["image_file"]

#   # Read the entire image data into bytes
#   image_bytes = buf.read()
#   image_base64 = base64.b64encode(image_bytes).decode('utf-8')

#   try:
#     # Attempt inference using Roboflow API
#     result = CLIENT.infer(image_base64, model_id="od-cte7x/2")
#     pprint.pprint(result)

#     # Process the inference result based on Roboflow API response format
#     processed_boxes = []
#     for prediction in result["predictions"]:
#       class_id = prediction["class"]
#       score = prediction["confidence"]
#       x = prediction.get('x')
#       y = prediction.get('y')
#       width = prediction.get('width')
#       height = prediction.get('height')
      
#       processed_boxes.append([x, y, width, height, class_id, score])

#     return jsonify(processed_boxes)

#   except Exception as e:
#     # Handle potential errors during inference (optional)
#     print(f"An error occurred during inference: {e}")
#     return jsonify({"error": "An error occurred during object detection."})


# serve(app, host='0.0.0.0', port=8080)

# conda activate pdfchat
# python object_detector.py





