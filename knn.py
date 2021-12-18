import tensorflow as tf

# 数据准备
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

# import matplotlib.pyplot as plt
# plt.imshow(x_train[1],cmap=plt.cm.binary)#显示图像并灰度处理
# plt.show()
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
########################################################
# 构建模型
#输入层：748
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())#展平
#第一层128个神经元
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
#第二层
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
#输出层10个
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#训练模型开始训练
model.fit(x_train,y_train,epochs=5)

val_loss,val_acc=model.evaluate(x_test,y_test)
print("loss=",val_loss,"acc",val_acc)

predictions=model.predict([x_test[:10]])
