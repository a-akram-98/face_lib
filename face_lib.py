import cv2

class face_lib:

    def __init__(self):
        
        self.frontalClassfier = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
        self.profileClassfier = cv2.CascadeClassifier("haarcascade_profileface.xml")
        self.faceEmbeddingNet = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")
    

    def face_detection(self, frame):
        """
            input: frame given by cv.VideoCapture

            return: [int] no. of faces detected in images, 
                    [list] list of faces coordinates in shape of (x, y, w, h)
        
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.frontalClassfier.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)
        if len(faces) == 0:
            faces = self.profileClassfier.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)
            if len(faces) == 0:
                grayFlip = cv2.flip(gray, 1)
                faces = self.profileClassfier.detectMultiScale(grayFlip, scaleFactor = 1.1, minNeighbors = 3)  #coordinates here are on the flipped image needed to be fliped on the original image 
        
        no_faces = len(faces)

        return no_faces, faces_coors   

    def face_recognition(self, face_coors, gt, frame):
        """
        input: face to verify [output from face_detection],
                ground_truth image
        output: verfied (boolen), True verified, False unverified

        """
        ######### preproces face ##########
        (x, y, w, h) = face_coors
        face = frame[x:x+h,y:y+h]
        face = cv2.resize(face, (96,96))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        ###################################

        ######### preprocess GT ###########
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        gt = cv2.resize(gt, (96,96))
        ###################################

        ####### get faceEmbeddings #######
        face_embeddings = self.face_embeddings(face)
        gt_embeddings = self.face_embeddings(gt)
        ######################################
    

    def face_embeddings(self, face):
        """
        input: image of face with dimensions of 96x96
        return: face embeddings of the image
        """
        ###### get blobs from images ######
        face_blob = cv2.dnn.blobFromImage(face)
        ###################################

        self.faceEmbeddingNet.setInput(face_blob)
        faceEmbeddings = self.faceEmbeddingNet.forward()
        return faceEmbeddings
