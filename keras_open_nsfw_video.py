#!/usr/bin/env python
import sys
import argparse
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from model import OpenNsfw
#from image_utils import create_yahoo_image_loader

import cv2
import math
import numpy as np


def main(argv):
    model = OpenNsfw()
    #fn_load_image = create_yahoo_image_loader()
    frameNsfw = 0
    frameTotal = 0
    videoFile = sys.argv[1]
    print("Processing...")
    cap = cv2.VideoCapture(videoFile)
    frameRate = cap.get(5) #frame rate
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if(ret != True):
            break
        if(frameId % math.floor(frameRate) == 0):
            cv2.imwrite('temp.jpg', frame)
            image_path = 'temp.jpg'
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            frameTotal= frameTotal+1
            preds = model.predict(x)

            if(preds[0][1]>=0.50):
                frameNsfw= frameNsfw+1

    #print("\tSFW score:\t{}\n\tNSFW score:\t{}".format(*predictions[0]))

    cap.release()
    if(frameNsfw>0):
        print("contain NSFW")
    else:
        print("SFW")
    print("NSFW % = "+str((frameNsfw/frameTotal)*100))


if __name__ == "__main__":
    main(sys.argv)
