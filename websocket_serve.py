import asyncio
import base64
import json

import numpy as np
import websockets
import cv2
from yolo import YOLO
from PIL import Image

print(' ========= websocket is going to run =========')
yolo = YOLO()


async def time_1(websocket, path):
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, img = video_capture.read()
        if not ret:
            break
        try:
            image = Image.fromarray(img)
            result, flag = yolo.detect_image(image)
            result = np.asarray(result)
            image = cv2.imencode('.jpg', result)[1]
            encoded_img = str(base64.b64encode(image))[2:-1]
            result_json = json.dumps({"result": 0, "message": "SUCCESS", "image": encoded_img, "flag": flag})
        except Exception as e:
            print(e)
            msg = str(e)
            result_json = json.dumps({"result": -1, "message": msg})
        print(result_json)
        await websocket.send(result_json)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    yolo.close_session()


start_server = websockets.serve(time_1, "10.20.50.163", 5680)
print(' ========= websocket running =========')
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

