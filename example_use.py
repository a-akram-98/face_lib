import cv2
import time
import numpy as np
from face_lib import face_lib


cap  = cv2.VideoCapture(0,cv2.CAP_DSHOW)
myface = cv2.imread("myface1.jpg")


face_lib = face_lib()

while(True):
    ret, frame = cap.read()

   
    v = face_lib.recognition_pipeline(frame, myface)
    print(v)

    cv2.imshow("frame", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(cv2)