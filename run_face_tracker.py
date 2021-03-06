import cv2
import numpy as np 
import dlib

# Caputre Webcam
cap = cv2.VideoCapture(0)

# Create face detector and predictor for landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # search boundaries
centerX = 320
centerY = 240
bounds = 30
# top_bounds = 150
# bot_bounds = 800
# left_bounds = 200
# right_bounds = 1600

# Rescale camera
cap.set(3, 640)
cap.set(4,480)

# Main loop running tracking
while(True):
	
	# Capture frame by frame
	ret, frame = cap.read()

	# Convert to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Appy detector to frame
	faces = detector(gray)
	for face in faces:

		# Draw Rectangle around face
		y1 = face.top()
		x1 = face.left()
		y2 = face.bottom()
		x2 = face.right()
		color = (0, 255, 0)
		stroke = 2
		cv2.rectangle(frame, (x1,y1), (x2,y2), color, stroke)

		# print landmark face points
		landmarks = predictor(gray, face)
		for n in range(27, 48):
			x = landmarks.part(n).x
			y = landmarks.part(n).y
			cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

		# print(landmarks.part(30))

		if landmarks.part(30).y < centerY - bounds:
			print("MOVE UP")
		if landmarks.part(30).y > centerY + bounds:
			print("MOVE DOWN")
		if landmarks.part(30).x < centerX - bounds:
			print("MOVE LEFT")
		if landmarks.part(30).x > centerX + bounds:
			print("MOVE RIGHT")


	# Display webcam feed
	cv2.imshow('frame',frame)

	# Exit loop by pressing "esc"
	if cv2.waitKey(20) == 27:
		break

cap.release()
cv2.destroyAllWindows()