from fastapi import FastAPI, HTTPException, Form, APIRouter
from routers.model import MyHTTPException, \
                        my_exception_handler, \
                        reply_bad_request, \
                        reply_server_error, \
                        reply_success
from ultralytics import YOLO
import os 
import uuid
import configparser
import yaml
from classify import infer, model, device # Example usage

config = configparser.ConfigParser()
config.read('config/config.ini')
current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_directory)
host_ip = config['DEFAULT']['host'] 
port_num = config['DEFAULT']['port'] 
static_folder = "static"

def load_model(custom_checkpoint_path):
    # custom_yaml_path = "1_686\YOLODataset\dataset.yaml"
    model = YOLO(custom_checkpoint_path)  # load a pretrained model (recommended for training)
    return model

with open("config/config.yaml", "r") as file:
    data = yaml.safe_load(file)
    CUSTOM_CHECKPOINT_PATH = data["CUSTOM_CHECKPOINT_PATH"]
MODEL = load_model(CUSTOM_CHECKPOINT_PATH)

with open("config/ship_class.yaml", "r") as file:
    SHIP_CLASS_DICT = yaml.safe_load(file)

def extract_objects(batch_yolo_result):
    obj_dict = {}
    yolo_result = batch_yolo_result[0]
    for single_object_result in yolo_result: # loop through objects in 1 image 
        single_object_result_box = single_object_result.boxes
        xyxy = single_object_result_box.xyxy.cpu().numpy()[0]
        object_cls = single_object_result_box.cls
        obj_class_id = int(object_cls.cpu().tolist()[0])
        new_class = SHIP_CLASS_DICT[obj_class_id]
        obj_dict[str(uuid.uuid4())] = (new_class, xyxy)

    return obj_dict 

import cv2

import cv2

from PIL import Image


def annotate_objects(image_path, obj_dict, res_path):
    image = cv2.imread(image_path)
    pil_image = Image.open(image_path)
    
    # Define the font
    font = cv2.FONT_HERSHEY_SIMPLEX

    for _, value in obj_dict.items():  # loop through bounding box and associated class
        obj_class, xyxy = value
        x1, y1, x2, y2 = xyxy

        # Define the coordinates in xyxy format (xmin, ymin, xmax, ymax)
        coordinates = (x1, y1, x2, y2)

        # Crop the image using the defined coordinates
        cropped_image = pil_image.crop(coordinates)
        # Add text label    
        final_pred = infer(cropped_image, model, device)

        # Draw bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 6)
        
        # Place text label at the top
        text_top = final_pred
        cv2.putText(image, text_top, (int(x1), int(y1) - 10), font, 0.8, (0, 0, 255), 3, cv2.LINE_AA)
        
        # Place text label at the bottom (you can adjust the position below)
        text_bottom = final_pred
        cv2.putText(image, text_bottom, (int(x1), int(y2) + 30), font, 0.8, (0, 0, 255), 3, cv2.LINE_AA)
    
    # Save or display the image with annotations
    cv2.imwrite(res_path, image)  # You can also use cv2.imshow() to display it directly



router = APIRouter()


@router.post("/detect-ship/")
async def detect_ship(
    image_path: str = Form(...),
):
    # Run inference on 'bus.jpg' with arguments
    batch_yolo_result = MODEL.predict(source=image_path, save=True, imgsz=640, conf=0.4)
    obj_dict = extract_objects(batch_yolo_result)
    res_name = str(uuid.uuid4()) + ".jpg"
    res_path = os.path.join(static_folder, res_name)
    annotate_objects(image_path, obj_dict, res_path)
    url = f"http://{host_ip}:{port_num}/static/" + res_name
    return {"result": url}

