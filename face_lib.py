import cv2
import numpy as np

class face_lib:

    def __init__(self):
        self.frontalClassfier = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
        #self.profileClassfier = cv2.CascadeClassifier("haarcascade_profileface.xml")
        self.faceEmbeddingNet = cv2.dnn.readNetFromTensorflow("graph_final.pb")
    
    def recognition_pipeline(self, face_img, gt_img, only_face_gt = False):
        """
        input: test image (frame), and ground truth image (given from cv.imread()),
        return: [True] if the person in the test image is the same as frounf truth image,
                [False] if there is no person detected or the preson in test not the one in the ground truth
        """
        
        _, face_coors = self.face_detection(face_img)
        _, gt_coors = self.face_detection(gt_img)

        if len(face_coors) == 0:
            return False
        face = self.recognition_preprocess(face_coors[0], face_img)
        gt = None
        if not only_face_gt:
            gt   = self.recognition_preprocess(gt_coors[0], gt_img)
        else:
            gt = gt_img

        distance = self.face_similarity(face,gt)

        if distance < 0.92:
            return True

        return False


    

    def face_detection(self, frame):
        """
            input: frame given by cv.VideoCapture

            return: [int] no. of faces detected in images, 
                    [list] list of faces coordinates in shape of (x, y, w, h)
        
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_coors = self.frontalClassfier.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 10)

        no_faces = len(faces_coors)

        return no_faces, faces_coors   
    
    def recognition_preprocess(self, coors, img):
        """
        input: coordinates given by face detection, image to extract the face
        operation: preprocessing the image to extract the face frome image and do some preprocessing for
        the faceEmbeddings model

        return: preprocessed image
        """
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
        self.faceEmbeddingNet.setInput(face)
        faceEmbeddings = self.faceEmbeddingNet.forward()[0]
        return faceEmbeddings

    @classmethod
    def findCosineDistance(cls, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    
    
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
