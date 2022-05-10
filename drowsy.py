#importing the required dependencies
import cv2  #video rendering
import dlib # for face and landmark detection
from scipy.spatial import distance # for calculating dist b/w the eye landmarks
import time
import imutils
import os
#to get the landmark ids of the left and right eyes
from imutils import face_utils
def calculate_EAR(eye):
    #Vertical eye landmarks
    #eye[0] = 37 eye[1] = 38 eye[2] = 39 eye[3] = 40 eye[4] = 41 eye[5] = 42
	#38번점과 42번점의 거리 (수직)
    A = distance.euclidean(eye[1], eye[5])
	#39번점과 41번점의 거리 (수직)
    B = distance.euclidean(eye[2], eye[4])
	#Horizontal eye landmarks
    #37번점과 40번점의 거리 (수평)
    C = distance.euclidean(eye[0], eye[3])
	
    #The EAR Equation
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio


def calculate_MAR(mouth):
	A = distance.euclidean(mouth[3], mouth[9])
	B = distance.euclidean(mouth[0], mouth[6])
	mar_aspect_ratio = A/B
	return mar_aspect_ratio

#비디오 불러오기
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Initializing the models for landmark and face detection
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor(
	"shape_predictor_68_face_landmarks.dat")
cnt=0
arr=[]
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps: ",fps)
while True:
    #카메라의 이미지 읽어들이기
    _, frame = cap.read()
    #이미지를 축소해서 출력하기
    frame = imutils.resize(frame, width = 640)

    #frame을 그레이  스케일로 바꿔줌
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #face detecting
    faces = hog_face_detector(gray)
    
    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []
        mouth = []
        #왼쪽 눈
        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            #선 그어주기
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)
        #오른쪽 눈
        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)
        #입
        for n in range(48, 60):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            mouth.append((x, y))
            next_point = n+1
            if n == 59:
                next_point = 48
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)
            
        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)
        MAR = round(calculate_MAR(mouth),2)
        
        EAR = (left_ear+right_ear)/2
        EAR = round(EAR, 2)
        arr.append({EAR,MAR})
        
        #입이 벌어져 있는 정도가 크면 -> 하품
        if MAR >= 0.5:  
            #print(MAR)
            #문자 출력
            print("yawn, eye Drowsy => NO")
            cv2.putText(frame, "******Are you Sleepy?*****", (10,100),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cnt=0
        #입이 벌어져 있는 정도는 작고 눈도 작게 뜨고 있으면 -> 눈 졸림 
        elif MAR <= 0.5 and EAR < 0.2:
            print(MAR)
            print("eye Drowsy")
            cnt += 1
            if cnt >= 50:
                cv2.putText(frame, "******Wakeup*****", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            
            print(cnt)
        #나머지 경우, 입이 작게 벌어져 있고 눈을 크게 뜨고 있으면 -> 졸음 X
        else:
            #카운트 초기화
            cnt=0

    cv2.imshow("Are you Sleepy", frame)
    new_arr = []
    number = 0
    number_arr = 0
    c = 0
    #1초 대기
    key = cv2.waitKey(1)
    if key == 27:
        for i in arr:
            if c % 3 == 0:
                #1초당 30프레임이므로 5초가 지나면 150프레임
                if c > 150:
                    break
                else:
                    #샘플링은 1초당 10프레임으로 잡았으므로 50개가 최대임
                    new_arr.append(arr[c])
                    number = number + 1
            c = c + 1       
        print(arr)
        print("new arr\n")
        print(new_arr)
        break
cap.release()
cv2.destroyAllWindows()
