import cv2
import numpy as np
import logging
import os
from .download import download

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logging_handle = "[Face Library]"
logger = logging.getLogger(logging_handle)


class face_lib:

    def __init__(self):
        BASE_DIR = os.path.dirname(__file__)

        self.__frontalClassfier = cv2.CascadeClassifier(os.path.join(BASE_DIR , "haarcascade_frontalface_alt2.xml"))
        ##TODO need to detcet the side of face if no frontal face detected
        #self.profileClassfier = cv2.CascadeClassifier("haarcascade_profileface.xml")
        self.__faceEmbeddingNet = None
        file_exists = False
        try:
            file_exists = os.path.exists(os.path.join(BASE_DIR , "graph_final.pb"))
            self.__faceEmbeddingNet = cv2.dnn.readNetFromTensorflow(os.path.join(BASE_DIR , "graph_final.pb"))
        except:
            ############## Download Face Recognition model fom github #############
            if file_exists :
                logger.info("Face recognition model is corrupted downloading it again, please wait ...")
                logger.info("This download will be done once if no errors happened, don't worry ...")
            else:
                logger.info("Downloading face recognition model for the first time ...")
                logger.info("This download will be done once if no errors happened, don't worry ...")
            
            logger.info("If the download didn't done correctly you can create the instance again it will try to download it again automatically for you")
            
            check_error = download("https://github.com/a-akram-98/face_lib/releases/download/v1.0.5/graph_final.pb", os.path.join(BASE_DIR , "graph_final.pb"), quiet=False)
            if check_error == "Error":
                logger.error("It seems github release is too slow, switching to the download from Google Drive")
                check_error = download(id = "1pNU-V31cdgCSRu9XQqoSv1tjbuAfVOSG",output= os.path.join(BASE_DIR , "graph_final.pb"), quiet=False)
                if check_error == "Error":
                    logger.error("Download Failed, It seems your connection is slow ...")
                    logger.info("You can download the model manually from this link: "+ "https://github.com/a-akram-98/face_lib/releases/download/v1.0.5/graph_final.pb")
                    logger.info("then add it to the following path:")
                    logger.info(BASE_DIR)
                    return


            self.__faceEmbeddingNet = cv2.dnn.readNetFromTensorflow(os.path.join(BASE_DIR , "graph_final.pb"))
            

    
    def recognition_pipeline(self, face_img, gt_img, only_face_gt = False, threshold = 0.92):
        """
        input: test image (face_img), and ground truth image (given from cv.imread(), expecting BGR image),
        return: [True] if the person in the test image is the same as frounf truth image,
                [False] if there is no person detected or the preson in test not the one in the ground truth
                [no_of_faces] number of faces detected in face_img (0 in case of no faces detected)

                if multiple faces in the test image (face_img) the function will verfiy if the face provided in g_image appears in the test image or not.
        """
        
        no_of_faces, faces_coors = self.faces_locations(face_img)

        no_of_faces_gt,  gt_coors = None, None
        gt = None
        if not only_face_gt:
            no_of_faces_gt,  gt_coors = self.faces_locations(gt_img)
            if no_of_faces_gt != 1:
                raise Exception("Detected more than one face in the ground truth image ... ")
            gt   = self.verification_preprocess(gt_img,gt_coors[0])
            
        else:
            gt   = self.verification_preprocess(gt_img)



        if no_of_faces == 0:
            return False, no_of_faces
    
        distances = list()        
        
        for face_coor in faces_coors:
            face = self.verification_preprocess(face_img, face_coor)


            distance = self.face_similarity(face,gt)
            distances.append(distance)

        min_distance = min(distances)

        if min_distance < threshold:
            return True, no_of_faces

        return False, no_of_faces


    
    def get_faces(self, image):
        """
        input: BGR image
        return: list of faces as numpy array
        """
        _, faces_locations = self.faces_locations(image)
        faces = list()
        for face_location in faces_locations:
            (x,y,w,h) = face_location
            face = image[y:y+h,x:x+w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            faces.append(face)

        return faces

    def faces_locations(self, image):
        """
            input: frame given by cv.VideoCapture or image given from cv2.imread() (BGR images)

            return: [int] no. of faces detected in images, 
                    [list] list of faces coordinates in shape of (x, y, w, h)
        
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_coors = self.__frontalClassfier.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 10)

        no_faces = len(faces_coors)

        return no_faces, faces_coors   
    
    def verification_preprocess(self,  img, coors = None):
        """
        input: coordinates given by face detection, image to extract the face, None if the image contains only one face
        operation: preprocessing the image to extract the face frome image and do some preprocessing for
        the faceEmbeddings model

        return: preprocessed image
        """
        img = img
        if coors is not None:
            (x,y,w,h) = coors
            img = img[y:y+h,x:x+w]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160,160), interpolation=cv2.INTER_LINEAR)
        img = self.prewhiten(img)
        img = img.transpose([2, 0, 1])
        img = np.expand_dims(img, axis=0)

        return img

    def prewhiten(self, x):
        """
        Normalizing the vector
        """
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y
    
    def face_similarity(self, face, gt):
        """
        input: face to verify [output from face_detection],
                ground_truth image
        output: verfied (boolen), True verified, False unverified

        """

        face_embeddings = self.face_embeddings(face)
        gt_embeddings = self.face_embeddings(gt)

        euclidean_distance = self.findEuclideanDistance(self.l2_normalize(face_embeddings), self.l2_normalize(gt_embeddings))

        
        return euclidean_distance


    

    def face_embeddings(self, face):
        """
        input: image of face with dimensions of 96x96
        return: face embeddings of the image
        """
        self.__faceEmbeddingNet.setInput(face)
        faceEmbeddings = self.__faceEmbeddingNet.forward()[0]
        return faceEmbeddings
    
    
    def findEuclideanDistance(self, source_embedding, test_embedding):
        """
        inputs: emebddings of two faces
        returns: euclidean distance between two embeddings (vectors)
        """
        euclidean_distance = source_embedding - test_embedding
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance
    
    def l2_normalize(self, x):
        return x / np.sqrt(np.sum(np.multiply(x, x)))
