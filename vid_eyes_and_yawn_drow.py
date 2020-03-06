import argparse
import platform
import subprocess
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from PIL import Image
from PIL import ImageDraw
import imutils
from imutils import face_utils
import numpy as np
import dlib
import cv2
import time
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from picamera import PiCamera
from time import sleep
import platform
import subprocess
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
import itertools
from picamera.array import PiRGBArray
import itertools



def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--model',
      help='Path of the detection model, it must be a SSD model with postprocessing operator.',
      required=True)
    parser.add_argument('--label', help='Path of the labels file.')
    parser.add_argument(
      '--keep_aspect_ratio',
      dest='keep_aspect_ratio',
      action='store_true',
      help=(
          'keep the image aspect ratio when down-sampling the image by adding '
          'black pixel padding (zeros) on bottom or right. '
          'By default the image is resized and reshaped without cropping. This '
          'option should be the same as what is applied on input images during '
          'model training. Otherwise the accuracy may be affected and the '
          'bounding box of detection result may be stretched.'))
    parser.set_defaults(keep_aspect_ratio=False)
    args = parser.parse_args()
    
    #Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
    EYE_ASPECT_RATIO_CONSEC_FRAMES = 30
    
    YAWN_THRESH = 20
    YAWN_CONSEC_FRAMES = 20
    
    #COunts no. of consecutuve frames below threshold value
    COUNTER_EAR = 0
    COUNTER_YAWN = 0
    
    # Initialize engine.
    print("[INFO] loading Coral model...")
    engine = DetectionEngine(args.model)
    labels = dataset_utils.read_label_file(args.label) if args.label else None

    
    
    print("[INFO] loading shape predictor dlib model...")
    predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("[INFO] loaded shape predictor dlib model...")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    blank_img = np.zeros([375,500,3],dtype = np.uint8)
    blank_img.fill(255)
    
    
    ### Taking still pictures for person-specific thresholds
    camera = PiCamera()
    
    time.sleep(0.1)
    
    for i in range (1,3):
        print('Picture ' + str(i)+ ': ' + 'Please keep your eyes open, as you would do normally')
        msg1 = "Keep your eyes open normally."
        
        camera.start_preview()
        camera.annotate_text = msg1
        sleep(5)
        
        camera.capture('pic'+ str(i) + '.jpg')
        
        
        camera.stop_preview()
        sleep(2)
        
        
        
        
    eye_img1 = cv2.imread('/home/pi/FaceDetection/pic1.jpg')
    eye_img2 = cv2.imread('/home/pi/FaceDetection/pic2.jpg')
    
    eye_img12 = cv2.cvtColor(eye_img1, cv2.COLOR_BGR2RGB)
    eye_img22 = cv2.cvtColor(eye_img2, cv2.COLOR_BGR2RGB)
    
    
    eye_img13 = Image.fromarray(eye_img12)
    eye_img23 = Image.fromarray(eye_img22)
    
    results_eye_1 = engine.DetectWithImage(eye_img13, threshold=0.4, keep_aspect_ratio=True, relative_coord=False)
    results_eye_2 = engine.DetectWithImage(eye_img23, threshold=0.4, keep_aspect_ratio=True, relative_coord=False)
    
    for r in results_eye_1:
        
        box = r.bounding_box.flatten().astype("int")
        (startX, startY, endX, endY) = box
        cv2.rectangle(eye_img1, (startX, startY), (endX, endY),
                            (0, 255, 0), 2)
        
        y = startY - 15 if startY - 15 > 15 else startY + 15
        
        text = "{:.2f}%".format(r.score * 100)
        
        cv2.putText(eye_img1, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)        
        
        left = box[0]
        top = box[1]
        right = box[2]
        bottom = box[3]
       
        box1 = dlib.rectangle(int(left),int(top),int(right),int(bottom))

        shape = predict(eye_img1, box1)
        shape = face_utils.shape_to_np(shape)

        distance = lip_distance(shape)

        #Get array of coordinates of leftEye and rightEye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear_1 = (leftEAR + rightEAR) / 2.0
        
        print('EAR = '+ str(ear_1))
        
        #Use hull to remove convex contour discrepencies and draw eye shape around eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        
        lip = shape[48:60]



    
    for r in results_eye_2:
        
        box = r.bounding_box.flatten().astype("int")
        (startX, startY, endX, endY) = box
        cv2.rectangle(eye_img2, (startX, startY), (endX, endY),
                            (0, 255, 0), 2)
        
        y = startY - 15 if startY - 15 > 15 else startY + 15
        
        text = "{:.2f}%".format(r.score * 100)
        
        cv2.putText(eye_img2, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)        
        
        left = box[0]
        top = box[1]
        right = box[2]
        bottom = box[3]
       
        box1 = dlib.rectangle(int(left),int(top),int(right),int(bottom))

        shape = predict(eye_img2, box1)
        shape = face_utils.shape_to_np(shape)

        distance = lip_distance(shape)

        #Get array of coordinates of leftEye and rightEye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear_2 = (leftEAR + rightEAR) / 2.0
        
        print('EAR = '+ str(ear_2))
        
        #Use hull to remove convex contour discrepencies and draw eye shape around eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        
        lip = shape[48:60]
    
    
    camera.close()
    
    avg_ear_nd = (ear_1 + ear_2) * 0.5
    print('Average normal EAR: ' + str(avg_ear_nd))   
    
    
    EYE_ASPECT_RATIO_THRESHOLD = avg_ear_nd * 0.9
    
    
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 500 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        orig = frame.copy()
        print(orig.shape)

        # prepare the frame for object detection by converting (1) it
        # from BGR to RGB channel ordering and then (2) from a NumPy
        # array to PIL image format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = Image.fromarray(frame)

        # make predictions on the input frame
        #start = time.time()
        #print("Face detection started at: ", start)
        results = engine.DetectWithImage(frame, threshold=0.4, keep_aspect_ratio=True, relative_coord=False)
##        
##        if results == []:
##            orig = cv2.addWeighted(orig,0,blank_img,0.6, 0)
##            cv2.putText(orig, "Please position your face correctly.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        for r in results:
            
            if (r.score*100) < 99.20:
                orig = cv2.addWeighted(orig,0,blank_img,0.6, 0)
                cv2.putText(orig, "Cannot detect drowsiness. Please position your face towards the camera.", (25, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    
                    
            else:   
            # extract the bounding box and box and predicted class label
                box = r.bounding_box.flatten().astype("int")

                (startX, startY, endX, endY) = box

                # draw the bounding box on the image
                cv2.rectangle(orig, (startX, startY), (endX, endY),
                        (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                
                
                text = "{:.2f}%".format(r.score * 100)
                
                
                    
                
                cv2.putText(orig, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                left = box[0]
                top = box[1]
                right = box[2]
                bottom = box[3]

                box1 = dlib.rectangle(int(left),int(top),int(right),int(bottom))

                shape = predict(orig, box1)
                shape = face_utils.shape_to_np(shape)

                distance = lip_distance(shape)

                #Get array of coordinates of leftEye and rightEye
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0
                
                frame=np.array(frame)
                
                #Use hull to remove convex contour discrepencies and draw eye shape around eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(orig, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(orig, [rightEyeHull], -1, (0, 255, 0), 1)
                flag = 0

                lip = shape[48:60]
                cv2.drawContours(orig, [lip], -1, (0, 255, 0), 1)

                #Detect if eye aspect ratio is less than threshold
                if(ear < EYE_ASPECT_RATIO_THRESHOLD):
                    COUNTER_EAR += 1
                    #If no. of frames is greater than threshold frames,
                    if COUNTER_EAR >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                        flag = 1
                        #cv2.putText(frame, "The driver is drowsy!", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2) 
                    else:
                        flag = 0
                else:
                    COUNTER_EAR = 0
                    


                if (distance > YAWN_THRESH):
                    #cv2.putText(frame, "The driver is drowsy!", (10, 30),
                                #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    COUNTER_YAWN += 1
                    
                    if COUNTER_YAWN >= YAWN_CONSEC_FRAMES:
                        flag = 1
                    else:
                        flag = 0
                else:
                    COUNTER_YAWN = 0
                    
                if flag == 1:
                    cv2.putText(orig, "The driver is drowsy!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
      

            # show the output frame and wait for a key press
        frame=np.array(frame)
        cv2.imshow('Frame',orig)
        key = cv2.waitKey(1) & 0xFF
    
    
    # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    
if __name__ == '__main__':
    main()

