import cv2
import os
import numpy as np
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout,Conv2D,MaxPooling2D
import sklearn.model_selection
import pickle
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
import tensorflow
from keras.utils import to_categorical

data_path='C://Users//Adwit//Desktop//Projectdata'
categories=os.listdir(data_path)        #putting the two categories in a list known as categories
labels=[i for i in range(len(categories))]    #putting 0,1 in a list labels to show the position of file and it will be the key in dict

label_dict=dict(zip(categories,labels))         #putting the two lists together with zip and creatiing a key value scenario to make a dict

img_size=100
data=[]
target=[]


for category in categories:
    folder_path = os.path.join(data_path, category)        #mostly goes into the file project date and in one iteration opens the file inside it and next iteration the next one
    img_names=os.listdir(folder_path)
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)                              #this is basically accessing each image in each folder of the main'Projectdata'

        try:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            resized=cv2.resize(gray,(img_size,img_size))
            data.append(resized)
            target.append(label_dict[category])              #appends 0 or 1 according to which image it is with or without (as the value in dict is 0 or 1 and key is with mask or without mask)

        except Exception as e:
            print("Exception occured:",e)                 #we put try and cath sincr sometimes computer is unable to read images


data=np.array(data)/255     #making the list into an array and dividing every pixel by 255 to make computation easier
data=np.reshape(data,(data.shape[0],img_size,img_size,1))   #we make the img into 4 dimensional array since cnn only accepts that,1 is there since the img is gray
target=np.array(target)

#new_target=np_utils.to_categorical(target)          #since the last layer of our cnn will be 2 neurons and wwill choose one category hence our target is categorical

np.save('data',data)                          #saving
#np.save('target',new_target)

lb = LabelBinarizer()
target = lb.fit_transform(target)
target = to_categorical(target)




'''data_train, data_test, target_train, target_test = sklearn.model_selection.train_test_split(data, target,test_size=0.1)      # splits 10% of data away

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(50, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint= ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit(data_train,target_train,epochs=20,callbacks=[checkpoint],validation_split=0.2)

results=model.evaluate(data_test,target_test)
model.save("model.h5")'''
model= keras.models.load_model("model.h5")

video= cv2.VideoCapture("C://Users//Adwit//Desktop//Test4.mp4 ")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

color_dict={0:(0,255,0),1:(0,0,255)}

while True:
    check,frame=video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # to make the frame captured gray
    gray = cv2.GaussianBlur(gray, (21, 21), 0)  # to make the motion detection  more efficient

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7)

    for x,y,w,h in faces:
        region_o_inter=gray[y:y+w,x:x+w]    #now to send our region of interest which is the face into the model cnn
        resized=cv2.resize(region_o_inter,(100,100))
        normalized=resized/255
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)
        print(result)

        label=np.argmax(result,axis=1)[0]
        print(color_dict[label])

        cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],2)   #main rectangle around face
        cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[label],-1)   #for the triangle in which we write mask or without mask
        #cv2.putText(frame,label_dict[label],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        frame2=frame[y:y+w,x:x+w]
        cv2.imshow("tst", frame2)
    cv2.imshow("main",frame)

    key = cv2.waitKey(1)  # every 1 millisecond it switches to a new frame , waitKey(0)is used to close the current frame at the moment user presses a key
    if key == ord('q'):
        break

cv2.destroyAllWindows()



