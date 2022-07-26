# ================================================================
#
#   File name   : detection_demo.py
#   Author      : PyLessons
#   Created date: 2020-09-27
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
# ================================================================
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from yolov3.utils import detect_image, Load_Yolo_model
from yolov3.configs import *

image_name = "bike"
image_path = "./IMAGES/{}.jpg".format(image_name)
video_path = "./IMAGES/test.mp4"

yolo = Load_Yolo_model()
detect_image(yolo, image_path, "./IMAGES/{}_pred.jpg".format(image_name), input_size=YOLO_INPUT_SIZE, show=False,
             rectangle_colors=(255, 0, 0))
# detect_video(yolo, video_path, "", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0))
# detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255, 0, 0))

# detect_video_realtime_mp(video_path, "Output.mp4", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0), realtime=False)
