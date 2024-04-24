"""
uses a local `detectserver` instance to separate images 
"""

import requests
import os 
import json
import cv2 
import base64
import numpy as np
import argparse
import pathlib
from tqdm import tqdm
import shutil


def make_request(image_path: str):
    try:
        select_image = cv2.imread(os.path.join(image_path))
        _, im_arr = cv2.imencode(".jpeg", select_image)
        img_bytes = im_arr.tobytes()
        im_b64 = base64.b64encode(img_bytes)
        myobj = {"card_limit":1,"data":f"{im_b64.decode('ascii')}"}
        resp = requests.post(
                ADDRESS,
                json=myobj
                )
        _resp = resp.json()
        extension = os.path.splitext(image_path)[1] 
        fname = pathlib.Path(image_path).stem + extension
        # print(_resp)
        if _resp['message'] == "success":
            shutil.move(image_path, os.path.join(VERIFIED_DIR, fname))
        # else:
        #     shutil.move(image_path, os.path.join(BACK_DIR, fname))
    except Exception as e:
        pass


if __name__ == "__main__":

    ADDRESS = "http://localhost:5005/verify"
    args = argparse.ArgumentParser()
    args.add_argument("--folder", type=str, help="processing folder")
    opt = args.parse_args()
    
    VERIFIED_DIR = os.path.join(opt.folder, "verified")
    # if not os.path.exists(BACK_DIR):
    #     os.makedirs(BACK_DIR)
    if not os.path.exists(VERIFIED_DIR):
        os.makedirs(VERIFIED_DIR)
    
    all_images = [os.path.join(opt.folder, x) for x in os.listdir(opt.folder)]
    # print(all_images)
    [make_request(x) for x in tqdm(all_images)]
