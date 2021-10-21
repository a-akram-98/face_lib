# Face Library

![](https://img.shields.io/badge/current%20version-v1.1.0-brightgreen)
[![Downloads](https://pepy.tech/badge/face-library)](https://pepy.tech/project/face-library)
![](https://img.shields.io/badge/python-%3E%3D3.6-blue)
![](https://img.shields.io/badge/licence-MIT-red)

<p align="center">
<img src="https://raw.githubusercontent.com/a-akram-98/face_lib/master/logo/FL.jpeg" width="500"> 
</p>
Face Library is a 100% python open source package for accurate and real-time face detection and recognition. The package is built over OpenCV and using famous models and algorithms for face detection and recognition tasks. Make face detection and recognition with only one line of code.
The Library doesn't use heavy frameworks like TensorFlow, Keras and PyTorch so it makes it perfect for production.

## Release 1.1.0
**BlazeFace** model used in face detection now instead of Haar Cascade, decreasing the inference time x10 times and detect frontal and profile face more accurate 
## Patch 1.1.3
Solving little import issue

Please Upgrade to latest version if you already have Face Library.

Table of contents
=================
<!--ts-->
   * [Installation](#installation)
   * [Usage](#usage)
      * [Importing](#importing)
      * [Face detection](#face-detection)
      * [Face verfication](#face-verfication)
      * [Extracting face embeddings](#extracting-face-embeddings)
      * [For PIL images](#for-pil-images)
   * [Contributing](#contributing)
   * [Support](#support)
   * [Licence](#licence)
<!--te-->


## Installation
```bash
pip install face-library
```

## Upgrade
```bash
pip install face-library -U
```
## Usage
### Importing
```python
from face_lib import face_lib
FL = face_lib()
```

The model is built over OpenCV, so it expects cv2 input (i.e. BGR image), it will support *PIL* in the next version for RGB inputs. At the end there is a piece of code to make *PIL* image like cv2 image.

### Face detection
```python
import cv2

img = cv2.imread(path_to_image)
faces = FL.get_faces(img) #return list of RGB faces image
```
If you want to get faces locations (coordinates) instead of the faces from the image you can use
```python
no_of_faces, faces_coors = FL.faces_locations(face_img)
```
You can change the maximum number of faces could be detcted as follows
```python
no_of_faces, faces_coors = FL.faces_locations(face_img, max_no_faces = 10) #default number of max_no_faces is 2
```
You can change face detection thresholds *(score threshold, iou threshold)* -if needed-, by using the following function
```python
FL.set_detection_params(scoreThreshold=0.82, iouThreshold=0.24) # default paramters are scoreThreshold=0.7, iouThreshold=0.3
```

### Face verfication
The verfication process is compossed of two models, a face detection model detect faces in the image and a verfication model verfiy those face.

```python
img_to_verfiy = cv2.imread(path_to_image_to_verify) #image that contain face you want verify
gt_img = cv2.imread(path_to_image_to_compare) #image of the face to compare with

face_exist, no_faces_detected = FL.recognition_pipeline(img_to_verfiy, gt_image)
```

You can change the threshold of verfication with the best for your usage or dataset like this :
```python
face_exist, no_faces_detected = FL.recognition_pipeline(img_to_verfiy, gt_image, threshold = 1.1) #default number is 0.92
```
also if you know that `gt_img` has only one face and the image is zoomed to that face (minimum 65%-75% of image is face) like this :
<p align="center">
<img src="https://raw.githubusercontent.com/a-akram-98/face_lib/master/example%20img/jake.jpg" width="100"> 
</p>

You can save computing time and the make the model more faster by using

```python
face_exist, no_faces_detected = FL.recognition_pipeline(img_to_verfiy, gt_image, only_face_gt = True)
```
**Note**: if you needed to change detection parameters before the recognition pipeline you can call `set_detection_params` function as mentioned in [Face detection](#face-detection) section.
### Extracting face embeddings

I you want represent the face with vector from face only image, you can use
```python
face_embeddings = FL.face_embeddings(face_only_image)
```

### For PIL images
```python
import cv2
import numpy
from PIL import Image

PIL_img = Image.open(path_to_image)

cv2_img = cv2.cvtColor(numpy.array(PIL_img), cv2.COLOR_RGB2BGR) #now you can use this to be input for face_lib functions
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Support

There are many ways to support a project - starring⭐️ the GitHub repo is just one.

## Licence

Face library is licensed under the MIT License



