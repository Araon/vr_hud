import requests
import cv2
import numpy as np


url = "http://192.168.137.100:8080/shot.jpg"

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:

	img_resp = requests.get(url)

	img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)

	img = cv2.imdecode(img_arr, -1)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)

	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

	#imS = cv2.resize(faces, (320, 240))

	cv2.imshow("AndroidCam", img)

	if cv2.waitKey(1) == 27:
		break