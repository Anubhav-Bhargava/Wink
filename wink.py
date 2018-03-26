from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	
eye_threshold = 0.25
consecutive_frames = 5
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
cap=cv2.VideoCapture(0)

count=0
left_count=0
right_count=0
while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:

		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)#converting to NumPy Array

		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]

		left_eye_asp_ratio = eye_aspect_ratio(leftEye)
		right_eye_asp_ratio = eye_aspect_ratio(rightEye)

		eye_asp_ratio = (left_eye_asp_ratio + right_eye_asp_ratio) / 2.0

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		if left_eye_asp_ratio < eye_threshold:
			left_count = left_count + 1

			if left_count >= consecutive_frames:
				cv2.putText(frame, "****************LEFT WINK!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************LEFT WINK!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


		elif right_eye_asp_ratio < eye_threshold:
			right_count = right_count + 1

			if right_count >= consecutive_frames:
				cv2.putText(frame, "****************RIGHT WINK!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************RIGHT WINK!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		else:
			left_count = 0
			right_count=0	
		
	cv2.imshow("Frame", frame)
	k = cv2.waitKey(1) & 0xFF
	if k == 27 or k==ord("q"):
		break
		
cv2.destroyAllWindows()
cap.stop()
