import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout

model=Sequential()
model.add(Conv2D(10,(5,5),activation="relu",input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(20,(5,5),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100,activation="relu"))
model.add(Dense(10,activation="softmax"))
model.compile(optimizer="rmsprop",loss=tf.keras.losses.categorical_crossentropy,metrics=['accuracy'])


#加载训练数据
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
normalized_x_train=tf.keras.utils.normalize(x_train)
normalized_x_test=tf.keras.utils.normalize(x_test)

one_hot_y_train=tf.one_hot(y_train,10)
one_hot_y_test=tf.one_hot(y_test,10)

reshape_x_train=normalized_x_train.reshape(-1,28,28,1)

reshape_x_test=normalized_x_test.reshape(-1,28,28,1)
train_result=model.fit(reshape_x_train,one_hot_y_train,epochs=20,validation_data=(reshape_x_test,one_hot_y_test))

model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')

#显示训练结果
import matplotlib.pyplot as plt
plt.plot(train_result.history['accuracy'])
plt.plot(train_result.history['val_accuracy'])
plt.legend(["Accuracy","Validation Acc"])
plt.show()


import cv2
img=cv2.imread("E:\\cnn_test\\6.JPG")
img_width=img.shape[1]
img_height=img.shape[0]
# 切割图片为正方形
col_start=int((img_width-img_height)/2)
col_end=int(col_start+img_height)
cropped_img=img[:,col_start:col_end,:]

# 调灰度
gray_img=cv2.cvtColor(cropped_img,cv2.COLOR_BGR2GRAY)
#去灰度，形成黑白照
(thresh,black_white)=cv2.threshold(gray_img,128,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
black_white=cv2.bitwise_not(black_white)
black_white=cv2.resize(black_white,(28,28))#切割


black_white=black_white/255
black_white=black_white.reshape(-1,28,28,1)
prediction=model.predict(black_white)

import numpy as np
print(np.argmax(prediction))


cv2.imshow("a deer",black_white)
cv2.waitKey(0)
cv2.destroyAllWindows()



