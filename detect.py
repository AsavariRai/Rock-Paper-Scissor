# To process image array
#import numpy as np
import cv2
import tensorflow as tf

# import the tensorflow modules and load the model
import numpy as np
model= tf.keras.models.load_model('keras_model.h5')
  
# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

  
while(True):

    # Reading / Requesting a Frame from the Camera 
   status , frame = camera.read()      
    # if we were sucessfully able to read the frame
   if status: 
    # Flip the frame
    frame = cv2.flip(frame , 1)
    #resize the frame
    img= cv2.resize(frame, (224, 224))
    test_img= np.array(img, dtype= np.float32)
    # expand the dimensions
    test_img= np.expand_dims(test_img, axis= 0)
    # normalize it before feeding to the model
    normalised_img= test_img/255.0
    # get predictions from the model
    prediction= model.predict(normalised_img)
    print("prediction", prediction)
    # displaying the frames captured
    cv2.imshow('frame', frame)
      
    # waiting for 1ms
    key = cv2.waitKey(1)
    
    # if space key is pressed, break the loop
    if key == 32:
        break
  
# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()