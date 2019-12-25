# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 08:32:46 1019
@author: cse.mizan
"""
#import Image, ImageTk
import tkexample as gui
from PIL import Image, ImageTk
gui.update()


import sys # system functions (ie. exiting the program)
import os # operating system functions (ie. path building on Windows vs. MacOs)
import time # for time operations
from pynput.mouse import Button, Controller
from pynput.keyboard import Key, Controller as KeyController
import numpy as np # matrix operations (ie. difference between two matricies)
import cv2 # (OpenCV) computer vision functions (ie. tracking)
import keras # high level api to tensorflow (or theano, CNTK, etc.) and useful image skin_segmenting\n",
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D



mouse = Controller()
keyboard = KeyController()

CURR_POSE = 'five'
DATA = os.path.join('validation_data')
MODEL_PATH = os.path.join('model')
MODEL_FILE = os.path.join(MODEL_PATH, 'hand_5_gesture_10dec2.hdf5') # path to model weights and architechture file
handBackImg = cv2.imread(os.path.join("resources\handBackground.png"), 1) 
#********************************** Timer **********************************
timer_bg = time.time()
timer_bg2 = time.time()
click_time = time.time()
lastMoveTime = time.time()
lastColorExTime = time.time()

#********************************** Boolean **********************************

isTrainingMode = False
isDynamicEnv = False
isHandColorModeDisabled = False
isHandColorModeTrackDisabled = True


isHandColorModeOnly = False
isClickEnabled = False
wasFist = 0
isInitialGestureDetected = False
disableDoubleClick = False
disableSingleClick = False

#********************************** Bounding Box *****************************

# Tracking
# Bounding box -> (TopRightX, TopRightY, Width, Height)
bi = (0, 0, 0, 0) # Starting position for bounding box
biScroll = bi # Starting position for bounding box
b = bi
bm = bi
bmi = bi
bzoom = bi
bzoominit = bi

#********************************** Counter **********************************
subtractRate = 10
track_count = 0
img_count = 108
tracker=0
trackerMed=0
trackerMouse=0
tracking = 0
trackingMed = 0
trackingMouse = 0
click_count = 0
five_count = 0
grab_count = 0
next_count = 0
previous_count = 0
right_count = 0
nonGest_count = 0
notTrack_count = 0
# Capture, process, display loop    
kernel = np.ones((3,3),np.uint8)


curGest = "None"

#********************************** Functions **********************************


hand_model = load_model(MODEL_FILE, compile=False)

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


hand_cascade = cv2.CascadeClassifier('palm3.xml')
face_cascade = cv2.CascadeClassifier('face.xml')


classes = {
    0: 'five',
    1: 'grab',
    2: 'next',
    3: 'previous',
    4: 'right'
}

positions = {
    'hand_pose': (15, 40), # hand pose text
    'hand_gest': (300, 40), # hand pose text
    'fps': (15, 10), # fps counter
    'null_pos': (100, 200) # used as null point for mouse control
}

def get_count(count):
    count_val = 1
    global click_count
    global five_count
    global grab_count
    global next_count
    global previous_count
    global right_count
    global curGest
    curGest = 'No Gesture Detected!'
    if(count == 'click'):
        click_count = click_count + 1
        five_count = 0
        grab_count = 0
        next_count = 0
        previous_count = 0
        right_count = 0
        count_val = click_count
    elif(count == 'five'):
        click_count = 0
        five_count = five_count + 1
        grab_count = 0
        next_count = 0
        previous_count = 0
        right_count = 0
        count_val = five_count
    elif(count == 'grab'):
        click_count = 0
        five_count = 0
        grab_count = grab_count + 1
        next_count = 0
        previous_count = 0
        right_count = 0
        count_val = grab_count
    elif(count == 'next'):
        click_count = 0
        five_count = 0
        grab_count = 0
        next_count = next_count + 1
        previous_count = 0
        right_count = 0
        count_val = next_count
    elif(count == 'previous'):
        click_count = 0
        five_count = 0
        grab_count = 0
        next_count = 0
        previous_count = previous_count + 1
        right_count = 0
        count_val = previous_count
    elif(count == 'right'):
        click_count = 0
        five_count = 0
        grab_count = 0
        next_count = 0
        right_count = right_count + 1
        count_val = right_count
    return count_val               

# Set up tracker.
def setup_tracker(ttype):
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[ttype]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
    return tracker

# Helper function for applying a mask to an array
def mask_array(array, imask):
    if array.shape[:2] != imask.shape:
        raise Exception("Shapes of input and imask are incompatible")
    output = np.zeros_like(array, dtype=np.uint8)
    for i, row in enumerate(imask):
        output[i, row] = array[i, row]
    return output

def skin_segment(action_frame):

    blur = cv2.GaussianBlur(action_frame, (3,3), 0)

    min_YCrCb = np.array([0,133,77],np.uint8)
    max_YCrCb = np.array([235,173,127],np.uint8)

    imageYCrCb = cv2.cvtColor(blur,cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
    blur = cv2.medianBlur(skinRegionYCrCb, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    ycrcb_d = cv2.dilate(blur, kernel)

    return ycrcb_d

def back_subtract(bg,frame):

    # Processing
    # First find the absolute difference between the two images
    bg = cv2.GaussianBlur(bg, (3,3), 0)
    frame = cv2.GaussianBlur(frame, (3,3), 0)
    
    diff = cv2.absdiff(bg, frame)
    #diff = cv2.absdiff(bg_init, diff)

    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Threshold the mask
    th, thresh = cv2.threshold(mask, subtractRate, 255, cv2.THRESH_BINARY)
    # Opening, closing and dilation
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    img_dilation = cv2.dilate(closing, kernel, iterations=1)
    # Get mask indexes
    imask = img_dilation > 0
    # Get foreground from mask
    foreground = mask_array(frame, imask)
    
    return foreground.copy()
    
def getFrame():
    ok, frame = video.read()
    frame = cv2.resize(frame, (720, 480), interpolation=cv2.INTER_LINEAR)
    frame = cv2.flip(frame, 1)
    frame = frame[100:480, 180:720]
    return frame

def capture(num):
    for i in range(1,num+1):
        video.read()
  
# Begin capturing video
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Could not open video")
    sys.exit()      
        
# Read first frame
ok, frame = video.read()
if not ok:
    print("Cannot read video")
    sys.exit()
# Use the first frame as an initial background frame
f1 = getFrame()
f2 = getFrame()
f3 = getFrame()
f4 = getFrame()
f5 = getFrame()
f6 = getFrame()

frame = f1.copy()

bg = frame.copy()
bg_initial = frame.copy()
bg_initial2 = f3.copy()
bg_initial3 = f3.copy()


bg_2 = frame.copy()
projImage = frame.copy()

# +++++++++++++++++++++++++++++++++++++++++++++++ GUI setup +++++++++++++++++++++++++++++++++++++++++++++

dynval = gui.tk.IntVar()
handval = gui.tk.IntVar()
handTrackval = gui.tk.IntVar()

def changeSubRate(val):
    global subtractRate
    subtractRate=float(val)
    
def changeDynEnv():
    global isDynamicEnv
    isDynamicEnv=bool(dynval.get())
    
def changeHandColorMode():
    global isHandColorModeDisabled
    if(handval.get()):
        isHandColorModeDisabled=False
    else:
        isHandColorModeDisabled=True
def changeHandColorTrackMode():
    global isHandColorModeTrackDisabled
    if(handTrackval.get()):
        isHandColorModeTrackDisabled=False
    else:
        isHandColorModeTrackDisabled=True
isActive = True

def on_quit():
    global isActive
    isActive = False
    
def Restart():
    global tracking,tracker
    global bi,b,f1,f2,f3,f4,f5,frame,bg,bg_initial,bg_initial2,bg_initial3,bg_2,projImage
    tracking = 0
    tracker = setup_tracker(2)
    bi = (0, 0, 0, 0)
    b = bi
    
    capture(20) 


gui.background_label.config(image=gui.background_image)
gui.subRate.config(command=changeSubRate)
gui.subRate.set(subtractRate)
gui.dynEnvButton.config(command=changeDynEnv,variable=dynval)
gui.handColorButton.config(command=changeHandColorMode,variable=handval)
gui.handColorTrackButton.config(command=changeHandColorTrackMode,variable=handTrackval)
handTrackval.set(0)

gui.button3.config(command=Restart)

gui.root.protocol("WM_DELETE_WINDOW", on_quit)

# +++++++++++++++++++++++++++++++++++++++++++++++ Start +++++++++++++++++++++++++++++++++++++++++++++

while isActive:
    # Read a new frame
    
    frame = getFrame()
    main_frame = frame.copy()
    display = frame.copy()
    data_display = np.zeros_like(display, dtype=np.uint8) # Black screen to display data
    if not ok:
        break
    
    foreground = back_subtract(bg, frame)
    foreground = back_subtract(bg_initial2,foreground)
    
    mask = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    # Threshold the mask
    th, img_dilation = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)

    foreground_display = foreground.copy()
    
    #cv2.imshow("preproces",img_skin_segment)
            
    # +++++++++++++++++++++++++++++++++++++++++++++++ Find hand +++++++++++++++++++++++++++++++++++++
        
    img = foreground.copy()
    img_skin_segment = skin_segment(img)
    #cv2.imshow("img_skin_segment",img_skin_segment)
    #cv2.imshow("img_ycr",img_ycr)
      
    if(isHandColorModeTrackDisabled==True):
        frame = img_dilation.copy()
    else:
        frame = img_skin_segment.copy()
    
    
    
    if tracking == 0:
        isInitialGestureDetected = False
        hands = hand_cascade.detectMultiScale(frame, 1.1, 5)
        X,Y,W,H=0,0,0,0
        for(x,y,w,h) in hands:
            cv2.rectangle(frame, (x,y), (x+w, int(y+h*1.5)), (255,255,255), 2)
            if X<x: 
                X=x
                W=w
            if Y<y: 
                Y=y
                H=h
        if (not (X==0 and Y==0 and W==0 and H==0) ):
            bi = (X-20,Y-15,W+30,H+45) # Starting position for bounding box
            b = bi            
            tracker = setup_tracker(2)
            tracking = tracker.init(frame, b)
            bInitGesture = (X-20,Y-15,W+30,H+45)
            initGestureTime = time.time()


        if (tracking == 0):
            notTrack_count = notTrack_count + 1
            if(notTrack_count % 10 == 0):
                bg_initial2 = f6.copy()
                f6 = main_frame.copy()
            bg = f1.copy()
            bg_2 = bg.copy()
            f1 = f2.copy()
            f2 = f3.copy()
            f3 = f4.copy()
            f4 = f5.copy()
            f5 = main_frame.copy()

        else:
            notTrack_count = 0
            f6 = bg_initial2.copy()

            bg = bg_initial2.copy()
            f1 = bg_initial2.copy()
            f2 = bg_initial2.copy()
            f3 = bg_initial2.copy()
            f4 = bg_initial2.copy()
            f5 = bg_initial2.copy()

           
    if tracking != 0:
        tracking, b = tracker.update(frame)
        tracking = int(tracking)
        if (tracking != 0):# and (time.time()-timer_bg)>0.2):
            track_count = 1
            timer_bg = time.time()
            if((time.time()-initGestureTime)<0.20): 
               bInitGesture = b
            elif((time.time()-initGestureTime)>0.20 and (time.time()-initGestureTime)<2.00  and isInitialGestureDetected == False):
                maxb = 0
                gest = ""
                if(b[0]-bInitGesture[0]>((b[2]/4))):
                    maxb = b[0]-bInitGesture[0]
                    gest = "left"
                    curGest = "right"
                elif(bInitGesture[0]-b[0]>((b[2]/5)) and maxb < bInitGesture[0]-b[0]):
                    maxb = bInitGesture[0]-b[0]
                    gest = "right"
                    curGest = "left"
                elif(b[1]-bInitGesture[1]>((b[3]/5)) and maxb < b[1]-bInitGesture[1]):
                    maxb = b[1]-bInitGesture[1]
                    gest = "down"
                    curGest = "down"
                elif(bInitGesture[1]-b[1]>((b[3]/3)) and maxb < bInitGesture[1]-b[1]):
                    maxb = bInitGesture[1]-b[1]
                    gest = "up"
                    curGest = "up"    
                if(gest == "right"):
                    keyboard.press(Key.right)
                    keyboard.release(Key.right)
                    isInitialGestureDetected = True
                    print("key.............................left")
                    cv2.putText(foreground_display, "hand gesture: right", positions['hand_gest'], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                
                elif(gest == "left"):
                    keyboard.press(Key.left)
                    keyboard.release(Key.left)
                    isInitialGestureDetected = True
                    print("key.............................right")
                    cv2.putText(foreground_display, "hand gesture: left", positions['hand_gest'], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                
                elif(gest == "down"):
                    if(isHandColorModeTrackDisabled==True):
                        isHandColorModeTrackDisabled = False
                        handTrackval.set(1)
                        curGest = "Color Track enabled!"
                    else:
                        isHandColorModeTrackDisabled = True
                        handTrackval.set(0)
                        curGest = "Color Track disabled!"
                    #keyboard.press(Key.down)
                    #keyboard.release(Key.down)
                    isInitialGestureDetected = True
                    print(curGest)
                    cv2.putText(foreground_display, "hand gesture: down", positions['hand_gest'], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                
                elif(gest == "up"):
                    isInitialGestureDetected = False
                    initGestureTime = time.time()-5.00
                    print("key.............................up")
                    cv2.putText(foreground_display, "hand gesture: up", positions['hand_gest'], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                   
            if(isDynamicEnv):
                if((b[1]-bi[1])>0):
                    main_frame[int(b[1]):int(bi[1]+b[3]), int(b[0]):int(b[0]+b[2])] = bg_initial[int(b[1]):int(bi[1]+b[3]), int(b[0]):int(b[0]+b[2])] 
                    main_frame[int(bi[1]+b[3]):int(b[1]+b[3]), int(b[0]):int(b[0]+b[2])] = bg_2[int(bi[1]+b[3]):int(b[1]+b[3]), int(b[0]):int(b[0]+b[2])] 
                else:#if((b[1]-bi[1])<5 and (bi[1]-b[1]<5) and (b[0]-bi[0])<5 and (bi[0]-b[0]<5)):
                    main_frame[int(b[1]):int(b[1]+b[3]), int(b[0]):int(b[0]+b[2])] = bg_initial[int(b[1]):int(b[1]+b[3]), int(b[0]):int(b[0]+b[2])] 
                bg = main_frame
                bg_initial = bg
            
        
    #cv2.imshow("main_frame",main_frame)
    #cv2.imshow("bg",bg)
    # Draw bounding box
    p1 = (int(b[0]), int(b[1]))
    p2 = (int(b[0] + b[2]), int(b[1] + b[3]))
    cv2.rectangle(foreground_display, p1, p2, (255, 0, 0), 2, 1)
    cv2.rectangle(display, p1, p2, (255, 0, 0), 2, 1)


    # ++++++++++++++++++++++++++++++++++++++++++ Gesture recognize ++++++++++++++++++++++++++++++++++++++++++
        
    # Use numpy array indexing to crop the foreground frame
    hand_crop = img_dilation[int(b[1]):int(b[1]+b[3]), int(b[0]):int(b[0]+b[2])]
    hand_crop_skin = img_skin_segment[int(b[1]):int(b[1]+b[3]), int(b[0]):int(b[0]+b[2])]
    try:
        # Resize cropped hand and make prediction on gesture
        hand_crop_resized = np.expand_dims(cv2.resize(hand_crop, (54, 54)), axis=0).reshape((1, 54, 54, 1))
        hand_crop_resized_skin = np.expand_dims(cv2.resize(hand_crop_skin, (54, 54)), axis=0).reshape((1, 54, 54, 1))
        prediction = hand_model.predict(hand_crop_resized)
        prediction_skin = hand_model.predict(hand_crop_resized_skin)
        if(isHandColorModeDisabled == False and (isHandColorModeOnly == True or max(prediction[0])<max(prediction_skin[0]))):
            prediction = prediction_skin
        #print("max: ",max(prediction[0]))
        if(max(prediction[0])>0.9):
            nonGest_count = 0
            predi = prediction[0].argmax() # Get the index of the greatest confidence
            #print(prediction[0])
            gesture = classes[predi]
            if(gesture=="click" and isClickEnabled == False):
                gesture = "five"
            if(isTrainingMode == False and (time.time()-initGestureTime)>2.5 and isInitialGestureDetected == False):
                #print(gesture)
                if(gesture=="five" and get_count("five")>2 ):
                    #print(b[0]*4,b[1]*4)
                    #print("mouse:",mouse.position)
                    '''
                    if(isColorExtractable or five_count == 60 ):
                        hand_colorImg = img.copy()
                        x = threading.Thread(target=thread_function, args=(hand_colorImg,b))
                        x.start()
                        isColorExtractable = False
                        #lastColorExTime = time.time()
                    '''    
                    curGest = "Mouse"
                    if(b[0]-bi[0]>2 or bi[0]-b[0]>2 or b[1]-bi[1]>2 or bi[1]-b[1]>2):
                        mouse.position = ((b[0]-10)*7,(b[1]-15)*7)
                        #mouse.position = ((bi[0]-10)*7+(b[0]-bi[0])*5,(bi[1]-15)*7+(b[1]-bi[1])*5)
                        bi = b
                        lastMoveTime = time.time()
                        disableDoubleClick = False
                        disableSingleClick = False
                    elif(time.time()-lastMoveTime>1.5):
                        curGest = "Mouse left double clicked!"
                        if(disableDoubleClick == False):
                            mouse.click(Button.left,2)
                            disableDoubleClick = True
                    elif(time.time()-lastMoveTime>0.6):
                        curGest = "Mouse left clicked!"
                        if(disableSingleClick == False):
                            mouse.click(Button.left)
                            disableSingleClick = True
                elif(gesture=="right" and get_count("right")>4):
                    curGest = "Mouse right clicked!"
                    if(right_count==5):
                        print("clicked ........................................right:)")
                        mouse.click(Button.right)  
                elif(gesture=="previous" and get_count("previous")>3):
                    print(" ........................................scrolling:)")
                    curGest = "Mouse scrolling!"
                    if(b[0]-bi[0]>3 or bi[0]-b[0]>3 or b[1]-bi[1]>3 or bi[1]-b[1]>3):
                        mouse.scroll(-(bi[0]-b[0]), (bi[1]-b[1]))
                elif(gesture=="next" and get_count("next")>3):
                    curGest = "Drag and Drop!"
                    if(next_count == 4):
                        mouse.press(Button.left)
                        print("press:...........................................left 1)")
                    if(b[0]-bi[0]>2 or bi[0]-b[0]>2 or b[1]-bi[1]>2 or bi[1]-b[1]>2):
                        #mouse.position = ((bi[0]-10)*7+(b[0]-bi[0])*5,(bi[1]-15)*7+(b[1]-bi[1])*5)
                        mouse.position = ((b[0]-10)*7,(b[1]-15)*7)
                        bi = b
                elif(gesture=="grab" and get_count("grab")>5):
                    curGest = "Zooming!"
                    bz2 = (b[0]+30,b[1]+30,b[2]-50,b[3]-50)
                    if(wasFist==0):
                        trackerMed = setup_tracker(4)
                        trackingMed = trackerMed.init(img, bz2)
                        bzoominit = bz2
                    elif trackingMed != 0:
                        trackingMed, bzoom = trackerMed.update(img)
                        trackingMed = int(tracking)  
                        p1 = (int(bzoom[0]), int(bzoom[1]))
                        p2 = (int(bzoom[0] + bzoom[2]), int(bzoom[1] + bzoom[3]))
                        cv2.rectangle(foreground_display, p1, p2, (255, 0, 0), 2, 1)
                        if(bzoom[0]-bzoominit[0]>3 or bzoominit[0]-bzoom[0]>3 or bzoom[1]-bzoominit[1]>3 or bzoominit[1]-bzoom[1]>3):
                            keyboard.press(Key.ctrl)
                            mouse.scroll((bzoominit[2]-bzoom[2])*2, (bzoominit[3]-bzoom[3])*2)
                            #biScroll = b
                            keyboard.release(Key.ctrl)
                            bzoominit = bzoom
                            
                    wasFist = 1
                else:
                    wasFist = 0
                if(gesture !="next"):
                    mouse.release(Button.left)
        else:
            nonGest_count = nonGest_count + 1
            if(nonGest_count>10):
                Restart()
            
            
        cv2.putText(foreground_display, "hand pose: {}".format(gesture), positions['hand_pose'], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    except Exception as ex:
        #if(tracking):
            #nonGest_count = nonGest_count + 1
            #if(nonGest_count>10):
                #Restart()
        cv2.putText(display, "hand pose: error", positions['hand_pose'], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(foreground_display, "hand pose: error", positions['hand_pose'], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    
     
        
        
    # _______________**************************** Display **************************__________________

    
    #cv2.imshow("display", display)
    #cv2.imshow("data", data_display)
    (x,y,w,h)=b
    btemp=b
    if((x+w)>foreground_display.shape[1]-1):
        w = w - ((x+w)-foreground_display.shape[1])-1
    elif(x<0):
        w = x+w
        x = 0
    if((y+h)>foreground_display.shape[0]-1):
        h = h - ((y+h)-foreground_display.shape[0])-1
    elif(y<0):
        h = (y+h)
        y = 0
    b=(x,y,w,h)
    gui.gestureLabel.config(text= curGest)
    
    backtorgb = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
    #backtorgb = cv2.flip(backtorgb, 1)
    img3 = handBackImg.copy()
    img3[int(b[1]):int(b[1]+b[3]), 300+int(b[0]):300+int(b[0]+b[2])] = cv2.addWeighted(handBackImg[int(b[1]):int(b[1]+b[3]), 300+int(b[0]):300+int(b[0]+b[2])], 1.0, backtorgb[int(b[1]):int(b[1]+b[3]), int(b[0]):int(b[0]+b[2])], 0.7, 0)     
    #img3[0:500,150:850] = cv2.addWeighted(handBackImg[0:500,150:850], 1.0, backtorgb, 0.7, 0)     
    ba,ga,ra = cv2.split(img3)
    imgTk = cv2.merge((ra,ga,ba))
    # Convert the Image object into a TkPhoto object
    im = Image.fromarray(imgTk)
    imgtk = ImageTk.PhotoImage(image=im) 
    # Put it in the display window
    gui.background_label.config(image=imgtk) 
    
    if(gui.isVisiblePreview==True):
        
        imgTk1=cv2.resize(foreground_display, (280, 180), interpolation=cv2.INTER_LINEAR)
        ba,ga,ra = cv2.split(imgTk1)
        imgTk1 = cv2.merge((ra,ga,ba))
        im1 = Image.fromarray(imgTk1)
        imgtk1 = ImageTk.PhotoImage(image=im1) 
        gui.preview1.config(image=imgtk1)
        
        backtorgb2 = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
        imgTk2=cv2.resize(backtorgb2, (280, 180), interpolation=cv2.INTER_LINEAR)
        ba,ga,ra = cv2.split(imgTk2)
        imgTk2 = cv2.merge((ra,ga,ba))
        im2 = Image.fromarray(imgTk2)
        imgtk2 = ImageTk.PhotoImage(image=im2) 
        gui.preview2.config(image=imgtk2)
    gui.update()
    b=btemp
    
    #cv2.imshow("img_skin_segment", img_skin_segment)
    '''
    try:
        # Display hand_crop
        cv2.imshow("hand_crop_resized", hand_crop)
        cv2.imshow("hand_crop_resized_skin", hand_crop_skin)
    except:
        pass
    # Display foreground_display
    '''
   
    # _______________************************** Keyboard ****************************__________________
    
    k = cv2.waitKey(1) & 0xff
    
    if k == 27: break # ESC pressed
    elif k == 115:
        # s pressed

        img_count += 1
        fname = os.path.join(DATA, CURR_POSE, "{}_{}.jpg".format(CURR_POSE, img_count))
        print(fname)
        cv2.imwrite(fname, hand_crop)
        img_count += 1
        fname = os.path.join(DATA, CURR_POSE, "{}_{}.jpg".format(CURR_POSE, img_count))
        print(fname)
        cv2.imwrite(fname, hand_crop_skin)
    elif k != 255: print(k)
        
        
cv2.destroyAllWindows()
video.release()
gui.root.destroy()