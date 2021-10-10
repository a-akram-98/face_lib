import cv2

class face_lib:

    def __init__(self):
        
        self.frontalClassfier = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
        self.profileClassfier = cv2.CascadeClassifier("haarcascade_profileface.xml")
    

    def face_detection(self, frame):
        """
            input: frame given by cv.VideoCapture

            return: [int] no. of faces detected in images, 
                    [list] list of faces in shape of (x, y, w, h)
        
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.frontalClassfier.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)
        if len(faces) == 0:
            faces = self.profileClassfier.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)
            if len(faces) == 0:
                grayFlip = cv2.flip(gray, 1)
                faces = self.profileClassfier.detectMultiScale(grayFlip, scaleFactor = 1.1, minNeighbors = 3)  #coordinates here are on the flipped image needed to be fliped on the original image 
        
        no_faces = len(faces)

        return no_faces, faces
    

    def face_recognition(self):
        raise NotImplementedError(self.__class__.__name__ + '.face_recognition')