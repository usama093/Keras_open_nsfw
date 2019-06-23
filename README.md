# Keras Implementation of Yahoo's Open NSFW Model

This repository contains an implementation of [Yahoo's Open NSFW Classifier](https://github.com/yahoo/open_nsfw) rewritten in Keras.

The original caffe weights have been extracted using [Caffe to TensorFlow](https://github.com/ethereon/caffe-tensorflow). You can find them at `data/open_nsfw-weights.npy`.

This works for both Images and Videos.

## Prerequisites

All code should be compatible with `Python 3.6` and `Keras` . The model implementation can be found in `model.py`.

### Usage
#### Images
```
> python keras_open_nsfw.py test.jpg

Results for 'test.jpg'
	SFW score:	0.9355766177177429
	NSFW score:	0.06442338228225708
```


#### Video
```
> python keras_open_nsfw_video.py video.mp4

Results for 'video.mp4'
Contain NSFW
NSFW % = 57.89473684210527
```

