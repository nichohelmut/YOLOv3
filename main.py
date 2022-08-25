import datetime
import os

import uvicorn
from fastapi import FastAPI, UploadFile, File

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from yolov3.utils import detect_image, Load_Yolo_model
from yolov3.configs import *

app = FastAPI()


@app.get('/index')
def get_root(name: str):
    return f"Hello World {name}"


@app.post('/api/predict')
def predict_image(file: UploadFile = File(...)):
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M:%S")
    try:
        contents = file.file.read()
        image_upload_time = "{}_{}".format(time_now, file.filename)
        with open("IMAGES/UPLOAD/{}".format(image_upload_time), 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        image_name = file.filename.split('.')[0]
        image_path = "./IMAGES/UPLOAD/{}.jpg".format(image_name)

        yolo = Load_Yolo_model()
        detect_image(yolo, image_path, "./IMAGES/PREDICTION/{}_{}_pred.jpg".format(time_now, image_name),
                     input_size=YOLO_INPUT_SIZE,
                     show=False,
                     rectangle_colors=(255, 0, 0))

# detect_video(yolo, video_path, "", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0))
# detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255, 0, 0))

# detect_video_realtime_mp(video_path, "Output.mp4", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0), realtime=False)


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')
