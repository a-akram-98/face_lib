# Face Library
Face Library is an open source package for accurate and real-time face detection and recognition. The package is built over OpenCV and using famous models and algorithms for face detection and recognition tasks. Make face detection and recognition with only one line of code.
The Library doesn't use heavy frameworks like TensorFlow, Keras and PyTorch so it makes it perfect for production.

## Installation
```bash
pip install face_lib
```

## Usage
### Importing
```python
from face_lib import face_lib
FL = face_lib()
```

The model is built over OpenCV, so it expects cv2 input (i.e. BGR image), it will support PIL in the next version for RGB inputs. At the end there is a piece of code to make PIL image like cv2 image.

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
## Face verfication
```python
img_to_verfiy = cv2.imread(path_to_image_to_verify) #image that contain face you want verify
gt_img = cv2.imread(path_to_image_to_compare) #image of the face to compare with

face_exist, no_faces_detected = FL.recognition_pipeline(img_to_verfiy, gt_image)
```



You can change the threshold of verfication with the best for your usage or dataset like this :
```python
face_exist, no_faces_detected = FL.recognition_pipeline(img_to_verfiy, gt_image, threshold = 1.1) #default number is 0.92
```
also if you know that `gt_img` has only one face and the image is zoomed to that face like this :
<p align="center">
<img src="https://raw.githubusercontent.com/a-akram-98/face_lib/master/example%20img/jake.jpg" width="100"> 
</p>

You can save computing time and the make the model more faster by using

```python
face_exist, no_faces_detected = FL.recognition_pipeline(img_to_verfiy, gt_image, only_face_gt = True)
```
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

cv2_img = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR) #now you can use this to be input for face_lib functions
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Support

There are many ways to support a project - starring⭐️ the GitHub repo is just one.

## Licence

Face_lib is licensed under the MIT License



