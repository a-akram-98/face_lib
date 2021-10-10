import cv2
import time
import numpy as np 

cap  = cv2.VideoCapture(0,cv2.CAP_DSHOW)


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
        faces = frontalClassfier.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)
        no_faces = len(faces)

        return no_faces, faces



        
        

frontalClassfier = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
profileClassfier = cv2.CascadeClassifier("haarcascade_profileface.xml")

while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    start = time.time()
    faces = frontalClassfier.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 10)
    
    """ if len(faces) == 0:
        faces = profileClassfier.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)
        if len(faces) == 0:
            grayFlip = cv2.flip(gray, 1)
            faces = profileClassfier.detectMultiScale(grayFlip, scaleFactor = 1.1, minNeighbors = 3)  """

    

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y-10), (x+w,y+h+10), (0,255,0), thickness = 2)
        z = frame[x:x+h,y:y+h]
        print(z.shape)

    cv2.imshow("frame", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(cv2)