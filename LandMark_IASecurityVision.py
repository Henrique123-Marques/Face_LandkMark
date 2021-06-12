import dlib
import cv2
import numpy as np


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:

	_, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #esquema de cor dos pontos na camera

	faces = detector(gray)
	for face in faces:

		#posicoes dos pontos
		x1 = face.left()
		y1 = face.top()
		x2 = face.right()
		y2 = face.bottom()
		
		landmarks = predictor(gray, face)

		for n in range(0, 68):
			x = landmarks.part(n).x
			y = landmarks.part(n).y
			cv2.circle(frame, (x, y), 4, (300,200,200), -1) 
	
	cv2.imshow('Frame', frame)
	key = cv2.waitKey(1)
	if key == 27:
		break 