import cv2
import time
import numpy as np
from face_lib import face_lib


cap  = cv2.VideoCapture(0)
myface = cv2.imread("myfile.jpg")


face_lib = face_lib()

while(True):
    time.sleep(3)
    ret, frame = cap.read()

    start = time.time()
    v = face_lib.recognition_pipeline(frame, myface)
    print(time.time()-start)

    cv2.imshow("frame", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
