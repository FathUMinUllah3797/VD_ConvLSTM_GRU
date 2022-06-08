import tensorflow.keras
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GRU, RepeatVector
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# declare some constants
data_dir = "Datasets//Surveillance_Fight_Dataset//" 
img_height , img_width = 64, 64 # dimension of each frame of videos
seq_len = 40 # number of images pass as one sequence
final_seq = int(seq_len/5)
#print("Final Sequence Length: ", final_seq)
classes = ["Non-Violent", "Violent"]

# extraction of frames from videos
#  Creating frames from videos
def frames_extraction(video_path):
	frames_list = []
     
	vidObj = cv2.VideoCapture(video_path)
    # Used as counter variable 
	count = 1
 
	while count <= seq_len: 
         
		success, image = vidObj.read() 
		if success:
			image = cv2.resize(image, (img_height, img_width))
			image = image/255
			if count%5 == 0:
				frames_list.append(image)
			count += 1
		else:
			#print("Defected frame")
			break
 
       
	return frames_list, count

# data creation
def create_data(input_dir):
	X = []
	Y = []
     
	classes_list = os.listdir(input_dir)
     
	for c in classes_list:
		print(c)
		if c in classes:
			if c == "Non-Violent":
				y = int(0)
				#print("*** noFight class ***")
			elif c == "Violent":
				y = int(1)
				#print("*** fight class ***")
			else:
				print()
				#print("*** Other class***")
			files_list = os.listdir(os.path.join(input_dir, c))
			for f in files_list:
				frames, count = frames_extraction(os.path.join(os.path.join(input_dir, c), f))
				#print(len(frames))
	           
				if len(frames) == final_seq:
					X.append(frames)
					Y.append(y)
	     
	X = np.asarray(X)
	Y = np.asarray(Y)
	return X, Y
X, Y = create_data(data_dir)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, shuffle=True, random_state=0)



# ConvLSTM based model design
model = Sequential()
model.add(ConvLSTM2D(filters = 64, kernel_size = (3, 3), return_sequences = True, data_format = "channels_last", input_shape = (final_seq, img_height, img_width, 3)))
model.add(ConvLSTM2D(filters = 128, kernel_size = (3, 3), return_sequences = True))
#model.add(Dropout(0.2))
model.add(ConvLSTM2D(filters = 256, kernel_size = (3, 3), return_sequences = True))
#model.add(Flatten())
#model.add(RepeatVector(1))
model.add(TimeDistributed(Flatten()))
model.add(GRU(200))
#model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# training the model
#opt = keras.optimizers.SGD(lr=0.0001)
opt = tensorflow.keras.optimizers.Adam(0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tensorflow.keras.metrics.BinaryAccuracy()])
earlystop = EarlyStopping(patience=50)
callbacks = [earlystop]
print("[INFO]...Model is training:")
history = model.fit(x = X_train, y = y_train, epochs=50, batch_size = 8 , shuffle=True, validation_split=0.10, callbacks=callbacks)

print("[INFO]...Model is saving:")
model.save("Surveillance_trained_model.h5")


