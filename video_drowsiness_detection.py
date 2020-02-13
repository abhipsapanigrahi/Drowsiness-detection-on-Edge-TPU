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
    
    EYE_ASPECT_RATIO_THRESHOLD = 0.3
    #Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
    EYE_ASPECT_RATIO_CONSEC_FRAMES = 50
    
    #COunts no. of consecutuve frames below threshold value
    COUNTER = 0
    
    # Initialize engine.
    print("[INFO] loading Coral model...")
    engine = DetectionEngine(args.model)
    labels = dataset_utils.read_label_file(args.label) if args.label else None


    # initialize the video stream and allow the camera sensor to warmup
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    #vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)
    
    
    print("[INFO] loading shape predictor dlib model...")
    ### Inserting code for detecting face landmarks here
    predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("[INFO] loaded shape predictor dlib model...")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 500 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        orig = frame.copy()

        # prepare the frame for object detection by converting (1) it
        # from BGR to RGB channel ordering and then (2) from a NumPy
        # array to PIL image format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = Image.fromarray(frame)

        # make predictions on the input frame
        start = time.time()
        print("Face detection started at: ", start)
        results = engine.DetectWithImage(frame, threshold=0.3, keep_aspect_ratio=True, relative_coord=False)
        #print(results)
        end = time.time()
        print("Face detection ended at: ", end)


        # loop over the results
        for r in results:
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
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


            #Detect if eye aspect ratio is less than threshold
            if(ear < EYE_ASPECT_RATIO_THRESHOLD):
                COUNTER += 1
                #If no. of frames is greater than threshold frames,
                if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                    cv2.putText(frame, "You are Drowsy!", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            else:
                COUNTER = 0

      

        # show the output frame and wait for a key press
        frame=np.array(frame)
        cv2.imshow('Frame',frame)
        key = cv2.waitKey(1) & 0xFF
        
    
    # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    
if __name__ == '__main__':
    main()

