#!/usr/bin/env python
import sys
import argparse
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from model import OpenNsfw

import numpy as np


def main(argv):
    model = OpenNsfw()
    
    img_path = sys.argv[1]
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = model.predict(x)
    print('SFW:', preds[0][0])
    print('NSFW:', preds[0][1])

if __name__ == "__main__":
    main(sys.argv)
