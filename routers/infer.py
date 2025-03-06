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
from classify import infer, device # Example usage
from classify import model as classify_model # Example usage

config = configparser.ConfigParser()
config.read('config/config.ini')
current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_directory)
host_ip = config['DEFAULT']['host'] 
port_num = config['DEFAULT']['port'] 
static_folder = "static"
static_video_folder = os.path.join(static_folder, "video")
static_image_folder = os.path.join(static_folder, "image")
os.makedirs(static_video_folder, exist_ok=True)
os.makedirs(static_image_folder, exist_ok=True)
def load_model(custom_checkpoint_path):
    # custom_yaml_path = "1_686\YOLODataset\dataset.yaml"
    model = YOLO(custom_checkpoint_path)  # load a pretrained model (recommended for training)
    return model

with open("config/config.yaml", "r") as file:
    data = yaml.safe_load(file)
    CUSTOM_CHECKPOINT_PATH = data["CUSTOM_CHECKPOINT_PATH"]
MODEL = load_model(CUSTOM_CHECKPOINT_PATH)
print ("----------------- LOAĐED MODEL YOLO -----------------")

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

def cal_distance(focal_pixel, h_real_mm, h_image_pixel):
    distance = focal_pixel * h_real_mm / h_image_pixel
    return distance

def get_distance(focal_pixel, h_real_mm, h_image_pixel):
        # h_image_pixel = abs(y2 - y1)
        distance = cal_distance(focal_pixel, h_real_mm, h_image_pixel)
        distance = int(distance/1000)
        return distance
def get_h_real_mm(final_pred_class):
    print ("final_pred_class", final_pred_class)
    if final_pred_class == "tau_danh_ca":
        return [2500, 4000] 
    elif final_pred_class == "tau_canh_sat":
        return [6000, 9000]
    elif final_pred_class == "tau_van_tai":
        return [2500, 15000]
    elif final_pred_class == "unknown":
        return [1500, 15000] 

def annotate_objects(image_path, obj_dict, res_path):
    image = cv2.imread(image_path)
    pil_image = Image.open(image_path)
    
    # Define the font
    font = cv2.FONT_HERSHEY_SIMPLEX
    focal_pixel = get_focal_pixel(image_path)
    for _, value in obj_dict.items():  # loop through bounding box and associated class
        obj_class, xyxy = value
        x1, y1, x2, y2 = xyxy
        

        # Define the coordinates in xyxy format (xmin, ymin, xmax, ymax)
        coordinates = (x1, y1, x2, y2)

        # Crop the image using the defined coordinates
        cropped_image = pil_image.crop(coordinates)
        # Add text label    
    
        # If the image has an alpha channel (RGBA), convert it to RGB
        if cropped_image.mode != 'RGB':
            cropped_image = cropped_image.convert('RGB')
        final_pred_class = infer(cropped_image, classify_model, device)

        h_image_pixel = abs(y2 - y1)
        
        h_real_mm_range = get_h_real_mm(final_pred_class)
        h_real_mm_min, h_real_mm_max = h_real_mm_range
        distance_min = get_distance(focal_pixel, h_real_mm_min, h_image_pixel)
        distance_max = get_distance(focal_pixel, h_real_mm_max, h_image_pixel)
        # Draw bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        
        # Place text label at the top
        text_top = final_pred_class 
        cv2.putText(image, text_top, (int(x1), int(y1) - 10), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        # Place text label at the bottom (you can adjust the position below)
        text_bottom = str(distance_min) + "-" + str(distance_max) 
        cv2.putText(image, text_bottom, (int(x1), int(y2) + 30), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    print ("annotate into image", res_path)
    # Save or display the image with annotations
    cv2.imwrite(res_path, image)  # You can also use cv2.imshow() to display it directly

def start_video(video_path: str):
    video_capture = cv2.VideoCapture(video_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # Check if video is opened correctly
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        exit()
    return video_capture, total_frames

def estimate_distance(y2, y1, focal_pixel):
    h_image_pixel = abs(y2 - y1)
    h_real_mm = 10000 # assume all 10 m
    distance = cal_distance(focal_pixel, h_real_mm, h_image_pixel)
    distance = int(distance/1000)
    print ("distance", distance)

def get_focal_pixel(image_path):
    import requests

    url = "http://127.0.0.1:4203/calculate-focal-length"

    payload = {'image_path': image_path}
    files=[

    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    response_json = response.json()
    focal_pixel = response_json['result']
    return focal_pixel
def process_image(file_path: str, res_path: str):
    start = time.time()
    batch_yolo_result = MODEL.predict(source=file_path, save=False, imgsz=640, conf=0.30, verbose = False)
    end1 = time.time()
    print ('yolo process time: ', end1 - start)
    obj_dict = extract_objects(batch_yolo_result)
    annotate_objects(file_path, obj_dict, res_path)
    print ("resnet process time: ", time.time() - end1)


import os

def collect_image_paths(folder_path):
    image_paths = []
    # Loop through all files and directories in the specified folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file is an image by its extension
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))  # Add the full path to the list
    return image_paths

# class Image:
#     def __init__(self, image_path: str):
#         self.image_path = image_path
#         self.focal_pixel = get_focal_pixel(self.image_path)
#         self.batch_yolo_result = MODEL.predict(source=self.image_path, save=False, imgsz=640, conf=0.30, verbose = False)
#         obj_dict = extract_objects(self.batch_yolo_result)

# class Item:
#     def __init__(self):
#         self.image_path = image_path
#         self.class_type = class_type
#         self.xyxy = xyxy
#         self.distance = distance

router = APIRouter()
from fastapi import FastAPI, File, UploadFile
import shutil
import time 
def check_image_file(filename):
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    file_extension = os.path.splitext(filename)[1].lower()
    return file_extension in allowed_extensions

static_save_folder_image = os.path.join("static", "image")
static_save_folder_folder = os.path.join("static", "folder")
os.makedirs(static_save_folder_image, exist_ok=True)
os.makedirs(static_save_folder_folder, exist_ok=True)
@router.post("/detect-ship-image/")
async def detect_ship_image(
    file_path: str = Form(...)
):
    res_name = str(uuid.uuid4()) + ".jpg"
    res_path = os.path.join(static_save_folder_image, res_name)

    process_image(file_path, res_path)
    url = f"http://{host_ip}:{port_num}/" + res_path.replace("\\", "/")
    return {"result": url}

@router.post("/detect-ship-image-folder/")
async def detect_ship_image_folder(
    folder_path: str = Form(...)
):
    folder_save = os.path.join(static_save_folder_folder, str(uuid.uuid4()))
    os.makedirs(folder_save, exist_ok=True)
    image_path_ls = collect_image_paths(folder_path)
    # Run inference on 'bus.jpg' with arguments
    for image_path in image_path_ls:
        res_name = str(uuid.uuid4()) + ".jpg"
        res_path = os.path.join(folder_save, res_name)
        process_image(image_path, res_path)
    url = folder_save.replace("\\", "/")
    return {"result": url}

# @router.post("/detect-ship-video/")
# async def detect_ship_video(
#     file_path: str = Form(...)
# ):
    
#     video_capture, total_frames = start_video(video_path = file_path)
#     num_capture_rate = 1
#     frame_count = 0
#     extracted_frame_count = 0
#     video_save_name = str(uuid.uuid4())
#     save_video_folder = os.path.join(static_video_folder, video_save_name)
#     save_process_video_folder = os.path.join(static_video_folder, video_save_name + "_processed")
#     os.makedirs(save_video_folder, exist_ok=True)
#     os.makedirs(save_process_video_folder, exist_ok=True)
#     # start_time = time.time()
#     while True:
        
#         # Read the next frame
#         ret, frame = video_capture.read()
        
#         if not ret:
#             break  # Exit loop when video ends
#         if frame_count % num_capture_rate == 0:  # Capture every 4th frame
#             # Save the frame as an image
#             save_temp_path = os.path.join(save_video_folder, f'{extracted_frame_count}.jpg')
#             cv2.imwrite(save_temp_path, frame)
#             res_path = process_image(save_temp_path, static_save_folder=save_process_video_folder)
#             extracted_frame_count += 1
#         # if time.time() - start_time > 10:
#         #     print ("processed frame: ", extracted_frame_count)
#         #     break 
#         frame_count += 1

#     # Release the video capture object
#     video_capture.release()
#     return {"result": save_process_video_folder}




# @router.post("/detect-ship-image-upload/")
# async def detect_ship_image_upload(
#     file: UploadFile = Form(...)
# ):
#     if not check_image_file(file.filename):
#         raise MyHTTPException(status_code=400, message="Invalid file type, file should be .jpg, .jpeg, .png, .gif")
#     extension = os.path.splitext(file.filename)[1].lower()
#     # Create a unique path to save the file
#     file_path = os.path.join(static_folder,str(uuid.uuid4())) + extension

#     # Save the uploaded file to the static folder
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
#         time.sleep(2)
#     # Run inference on 'bus.jpg' with arguments
#     res_path = process_image(file_path, static_save_folder=static_image_folder)
#     url = f"http://{host_ip}:{port_num}/" + res_path.replace("\\", "/")
#     return {"result": url}