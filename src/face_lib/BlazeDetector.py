import time
import cv2
import numpy as np
import onnxruntime
from .blazeFaceUtils import gen_anchors, AnchorsOptions
import json
import time
import os

class BlazeFaceDetector:

    def __init__(self, typ = "front", scoreThreshold = 0.7, iouThreshold = 0.3):
        BASE_DIR = os.path.dirname(__file__)
        self.type = typ
        self.scoreThreshold = scoreThreshold
        self.iouThreshold = iouThreshold
        self.sigmoidScoreThreshold = np.log(self.scoreThreshold/(1-self.scoreThreshold))
        self.initModel(typ,BASE_DIR)

        self.generateAnchorBoxes(typ)
    
    
    def initModel(self, typ, BASE_DIR):
        """
        Read and initialize the model
        """
        front_model_path = os.path.join(BASE_DIR , "face_detection_front_128x128.onnx")
        back_model_path = os.path.join(BASE_DIR , "face_detection_back_256x256.onnx")
        if typ == "front":
            self.session = onnxruntime.InferenceSession(front_model_path, None)
        elif typ =="back":
            self.session = onnxruntime.InferenceSession(back_model_path, None)
       
        self.getModelInputDetails()
        self.getModelOutputDetails()
    
    def detectFaces(self, image, maxFaces = 2):
        """
        This function aims to detect faces in image
        input: [image] a numpy cv2 array (i.e. BGR image)
        returns:
        """
        preprocessed_image = self.detectionPreprocessing(image)
        result = self.inference(preprocessed_image)

        scores, goodDetections = self.filterDetections(result[1])

        faces_locations = self.extractDetections(result[0], goodDetections)
        faces_locations = self.nms(faces_locations, scores, maxFaces)
        return faces_locations
    
    def __call__(self, image, maxFaces=2):
        return self.detectFaces(image, maxFaces)

    
    
    def getModelInputDetails(self):
        """
        This function aims to get inlut node details for onnx inference
        """
        self.input_details = self.session.get_inputs()[0]
        self.input_name = self.input_details.name
        input_shape = self.input_details.shape
        self.inputHeight = input_shape[1]
        self.inputWidth = input_shape[2]
        self.channels = input_shape[3]

    def getModelOutputDetails(self):
        """
        this funtion aims to get the output nodes details (mainly we need the names only) for onnx inference
        """
        self.output_details = self.session.get_outputs()
    
    def generateAnchorBoxes(self, typ):
        """
        This function aims to generate anchor boxes depends on the type of the model (front model or back model)

        """
        if typ == "front":
		# Options to generate anchors for SSD object detection models.
            anchors_options = AnchorsOptions(input_size_width=128, input_size_height=128, min_scale=0.1484375, max_scale=0.75
                , anchor_offset_x=0.5, anchor_offset_y=0.5, num_layers=4
                , feature_map_width=[], feature_map_height=[]
                , strides=[8, 16, 16, 16], aspect_ratios=[1.0]
                , reduce_boxes_in_lowest_layer=False, interpolated_scale_aspect_ratio=1.0
                , fixed_anchor_size=True)

        elif typ == "back":
            # Options to generate anchors for SSD object detection models.
            anchors_options = AnchorsOptions(input_size_width=256, input_size_height=256, min_scale=0.15625, max_scale=0.75
                    , anchor_offset_x=0.5, anchor_offset_y=0.5, num_layers=4
                    , feature_map_width=[], feature_map_height=[]
                    , strides=[16, 32, 32, 32], aspect_ratios=[1.0]
                    , reduce_boxes_in_lowest_layer=False, interpolated_scale_aspect_ratio=1.0
                    , fixed_anchor_size=True)
        self.anchors = gen_anchors(anchors_options)

    def detectionPreprocessing(self, image):
        """
        This function aims to preproccess the image before feeding it to the onnx model
        input: [image] image loaded with cv2.imread (i.e. BGR image)
        returns: [data] processed image for onnx model feeding
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.img_height, self.img_width, self.img_channels = img.shape
        # Input values should be from -1 to 1 with a size of 128 x 128 pixels for the fornt model
        # and 256 x 256 pixels for the back model
        img = img / 255.0
        """ img_resized = tf.image.resize(img, [self.inputHeight,self.inputWidth], 
                                    method='bicubic', preserve_aspect_ratio=False)
         """
        img_resized = cv2.resize(img, (self.inputHeight,self.inputWidth), interpolation=cv2.INTER_CUBIC)
        img_input = (img_resized - 0.5) / 0.5

        # Adjust matrix dimenstions
        reshape_img = img_input.reshape(1,self.inputHeight,self.inputWidth,self.channels)

        data = reshape_img.astype('float32')

        return data
    
    def inference(self, data):
        """
        This function takes the data from detectionPreprocessing function and passes it to BlazeFace model using onnx
        input: [data] data we got from detectionPreprocessing which is image as a numpy array
        return: [boxes] bounding boxes (including face landmarks)
                [scores] confidence of each output [boxes]
        """

        result = self.session.run([self.output_details[0].name,self.output_details[1].name,self.output_details[2].name,self.output_details[3].name], {self.input_name: data})
       
        scores = np.squeeze(np.concatenate((result[0], result[1]), axis = 1), axis =0)
        boxes = np.squeeze(np.concatenate((result[2], result[3]), axis = 1), axis = 0)
        
        return boxes, scores

    def filterDetections(self, scores):
        """
        This function aims to filter the bounding boxes by there confidence level
        input: [scores] numpy array with shape (num_boxes, 1) holds the confidence level of each box
        returns: numpy array scores after being filtered, and indices of those scores
        """
		# Filter based on the score threshold before applying sigmoid function
        goodDetections = np.where(scores > self.sigmoidScoreThreshold)[0]
        
		# Convert scores back from sigmoid values
        scores = 1.0 /(1.0 + np.exp(-scores[goodDetections]))
        
        return scores, goodDetections
    
    def extractDetections(self, predections, selectedIndices):
        """
        This function aims to extract the outputs from the anchor boxes
        inputs: [predections] numpy array holds the predicted outputs (boxes and keypoints), but we will use here the boxes only shape (num_of_boxes, 16)
                [selectedIndices] numpy array holds the indices of selected boxes
        returns: numpy array of boxes extracted from generated anchors shape (num_of_boxes, 4)
        """

        numGoodDetections = selectedIndices.shape[0]

        boxes = np.zeros((numGoodDetections, 4))
        for idx, detectionIdx in enumerate(selectedIndices):
            anchor = self.anchors[detectionIdx]

            sx = predections[detectionIdx, 0]
            sy = predections[detectionIdx, 1]
            w = predections[detectionIdx, 2]
            h = predections[detectionIdx, 3]

            cx = sx + anchor.x_center * self.inputWidth
            cy = sy + anchor.y_center * self.inputHeight

            cx /= self.inputWidth
            cy /= self.inputHeight
            w /= self.inputWidth
            h /= self.inputHeight
            
            # The next section is for facial keypoints detected from BlazeFace model 
            # commented it for speed concerns

            """ for j in range(KEY_POINT_SIZE):
                lx = output0[detectionIdx, 4 + (2 * j) + 0]
                ly = output0[detectionIdx, 4 + (2 * j) + 1]
                lx += anchor.x_center * self.inputWidth
                ly += anchor.y_center * self.inputHeight
                lx /= self.inputWidth
                ly /= self.inputHeight
                keypoints[idx,j,:] = np.array([lx, ly])
 """
            boxes[idx,:] = np.array([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5])

        return boxes

    def nms(self, boxes, probs=None, max_num_outputs = 1, iouThreshold=0.3):
        """
        This a fast non max suppression (NMS) implementation
        inputs: [boxes] numpy array conatins bounding boxes in format of [x1, y1, x2, y2] with shape (num_boxes, 4)
                , [probs] numpy array holds the confidence level of each box from 0-1 with shape (num_boxes, 1)
                , [max_num_outputs] an intger holds the maximum number of ouputs needed
                , [iouThrshold] an integer from 0-1 represents the intersection over union threshold
        returns: numpy array of boxes after applying NMS
        """
	# if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes are integers, convert them to floats -- this
        # is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and grab the indexes to sort
        # (in the case that no probabilities are provided, simply sort on the
        # bottom-left y-coordinate)
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = y2

        # if probabilities are provided, sort on them instead
        if probs is not None:
            idxs = probs.squeeze()

        # sort the indexes
        idxs = np.argsort(idxs)
        face_counter = 0
        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the index value
            # to the list of picked indexes
            if face_counter == max_num_outputs:
                break
            face_counter +=1
            last = len(idxs) - 1
            i = idxs[last]
            
            pick.append(i)

            # find the largest (x, y) coordinates for the start of the bounding
            # box and the smallest (x, y) coordinates for the end of the bounding
            # box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have overlap greater
            # than the provided overlap threshold
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > iouThreshold)[0])))
        # return only the bounding boxes that were picked
        
        boxes = boxes[pick]
        boxes[:,0] = self.img_width * boxes[:,0]
        boxes[:,2] = self.img_width * boxes[:,2]
        boxes[:,1] = self.img_height * boxes[:,1]
        boxes[:,3] = self.img_height * boxes[:,3]

        # convert from X1y1x2y2 to xywh
        boxes[:,2] = boxes[:,2] - boxes[:,0]
        boxes[:,3] = boxes[:,3] - boxes[:,1]
    
        return boxes.astype("int")
