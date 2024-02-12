import cv2
import numpy as np
import tensorflow as tf

# Set the GPU device for TensorFlow
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

path = 'haarcascade_frontalface_default.xml'

font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

# setting the recatangle background to white
rectangle_bgr = (255,255,255)

# making a black image
img = np.zeros((500,500))

# set some text
text = " text in box"

# adjusting width and height of the test box
(text_width, text_height) = cv2.getTextSize(text,font, fontScale=font_scale,thickness=1)[0]

# set the text start position
text_offset_x = 10
text_offset_y = img.shape[0] - 25

# make the coodrinates of the box with a small padding of twoo pixels
box_coords = ((text_offset_x,text_offset_y),(text_offset_x+text_width+2,text_offset_y-text_height-2))
cv2.rectangle(img,box_coords[0],box_coords[1],rectangle_bgr,cv2.FILLED)
cv2.putText(img,text,(text_offset_x,text_offset_y), font, fontScale=font_scale,color=(0,0,0),thickness=1)

cap = cv2.VideoCapture(1)
# check if the webcam is opened correctly

if not cap.isOpened():
        cap = cv2.VideoCapture(0)
if not cap.isOpened():
        raise IOError("Cannot open Webcam!!")

# Define the positions of emotion icons at the bottom right corner
icon_x_offset = 20
icon_y_offset = 20
icon_spacing = 10

#Load the emotion icons
emotions = ['angry', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
emotion_icons = {}
for emotion in emotions:
    icon = cv2.imread(f'{emotion}.png', cv2.IMREAD_UNCHANGED)
    if icon is None:
        print(f"Error: Failed to load '{emotion}.png'")
    emotion_icons[emotion] = cv2.resize(icon, (50,50))
icon_y = icon_y_offset
#=======================================================

new_model = tf.keras.models.load_model('emotion_detector.h5') # defining the model
# Define the prediction function with @tf.function
@tf.function
def predict_emotion(image):
    return new_model(image,training=False)

#==============================================

cap.set(3, 640)  # Set the width to 640 pixels
cap.set(4, 480)  # Set the height to 480 pixels
while True:
    ret,frame = cap.read()
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,1.1,4)

    for x,y,w,h in faces:
            roi_gray = gray[y:y+h,x:x+w]
            roi_color= frame[y:y+h,x:x+w]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            facess = faceCascade.detectMultiScale(roi_gray)
            if len(facess) == 0:
                    print("Face not Detected")
            else:
                for(ex,ey,ew,eh) in facess:
                        face_roi = roi_color[ey:ey+eh,ex:ex+ew] # crp the face

    final_image = cv2.resize(face_roi,(224,224))
    final_image = np.expand_dims(final_image,axis=0)
    final_image = final_image/255.0

    font = cv2.FONT_HERSHEY_SIMPLEX

    

    Predictions = predict_emotion(final_image)

    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN

    if (np.argmax(Predictions)==0):
        status = 'Angry'

        x1,y1,w1,h1 = 0,0,175,75
           
        # draw black background rectagle 

        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)

        #add text
        cv2.putText(frame,status,(x1+int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,0,255))
        
        # icon = emotion_icons['angry']
        # icon_x = frame.shape[1] - icon.shape[1] - icon_x_offset
        # frame[icon_y:icon_y + icon.shape[0], icon_x:icon_x + icon.shape[1]] = icon
        # icon_y += icon.shape[0] + icon_spacing

    elif (np.argmax(Predictions)==1):
        status = 'contempt'

        x1,y1,w1,h1 = 0,0,175,75
           
        # draw black background rectagle 

        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)

        #add text
        cv2.putText(frame,status,(x1+int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,0,255))
        
        # icon = emotion_icons['contempt']
        # icon_x = frame.shape[1] - icon.shape[1] - icon_x_offset
        # frame[icon_y:icon_y + icon.shape[0], icon_x:icon_x + icon.shape[1]] = icon
        # icon_y += icon.shape[0] + icon_spacing

    elif (np.argmax(Predictions)==2):
        status = 'disgust'

        x1,y1,w1,h1 = 0,0,175,75
           
        # draw black background rectagle 

        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)

        #add text
        cv2.putText(frame,status,(x1+int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,0,255))
        
        # icon = emotion_icons['disgust']
        # icon_x = frame.shape[1] - icon.shape[1] - icon_x_offset
        # frame[icon_y:icon_y + icon.shape[0], icon_x:icon_x + icon.shape[1]] = icon
        # icon_y += icon.shape[0] + icon_spacing

    elif (np.argmax(Predictions)==3):
        status = 'fear'

        x1,y1,w1,h1 = 0,0,175,75
           
        # draw black background rectagle 

        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)

        #add text
        cv2.putText(frame,status,(x1+int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,0,255))
        
        # icon = emotion_icons['fear']
        # icon_x = frame.shape[1] - icon.shape[1] - icon_x_offset
        # frame[icon_y:icon_y + icon.shape[0], icon_x:icon_x + icon.shape[1]] = icon
        # icon_y += icon.shape[0] + icon_spacing

    elif (np.argmax(Predictions)==4):
        status = 'happiness'

        x1,y1,w1,h1 = 0,0,175,75
           
        # draw black background rectagle 

        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)

        #add text
        cv2.putText(frame,status,(x1+int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,0,255))
        
        # icon = emotion_icons['happiness']
        # icon_x = frame.shape[1] - icon.shape[1] - icon_x_offset
        # frame[icon_y:icon_y + icon.shape[0], icon_x:icon_x + icon.shape[1]] = icon
        # icon_y += icon.shape[0] + icon_spacing

    elif (np.argmax(Predictions)==5):
        status = 'neutral'

        x1,y1,w1,h1 = 0,0,175,75
           
        # draw black background rectagle 

        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)

        #add text
        cv2.putText(frame,status,(x1+int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,0,255))
        
        # icon = emotion_icons['neutral']
        # icon_x = frame.shape[1] - icon.shape[1] - icon_x_offset
        # frame[icon_y:icon_y + icon.shape[0], icon_x:icon_x + icon.shape[1]] = icon
        # icon_y += icon.shape[0] + icon_spacing

    elif (np.argmax(Predictions)==6):
        status = 'Sadness'

        x1,y1,w1,h1 = 0,0,175,75
           
        # draw black background rectagle 

        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)

        #add text
        cv2.putText(frame,status,(x1+int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,0,255))
        
        # icon = emotion_icons['sadness']
        # icon_x = frame.shape[1] - icon.shape[1] - icon_x_offset
        # frame[icon_y:icon_y + icon.shape[0], icon_x:icon_x + icon.shape[1]] = icon
        # icon_y += icon.shape[0] + icon_spacing

    else:
        status = 'suprise'

        x1,y1,w1,h1 = 0,0,175,75
           
        # draw black background rectagle 

        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)

        #add text
        cv2.putText(frame,status,(x1+int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,0,255))
        
        # icon = emotion_icons['suprise']
        # icon_x = frame.shape[1] - icon.shape[1] - icon_x_offset
        # frame[icon_y:icon_y + icon.shape[0], icon_x:icon_x + icon.shape[1]] = icon
        # icon_y += icon.shape[0] + icon_spacing
        
        
    cv2.imshow("Face Emotion Recogniton",frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()