# %%
from keras.applications import vgg16
from keras.layers import GlobalAveragePooling2D

# %%
model = vgg16.VGG16(weights='imagenet'  , include_top = False ,input_shape=(64,64,3))

# %%
model.layers[:]

# %%
from keras.models import Sequential

# %%
from keras.layers import Dense, Dropout, Flatten

# %%
for  layer in model.layers:
    layer.trainable = False

# %%
last_layer = model.output

# %%
last_layer = GlobalAveragePooling2D()(last_layer)

# %%
last_layer = Flatten()(last_layer)

# %%
last_layer = Dense(512, activation='relu')(last_layer)

# %%
last_layer = Dropout(0.2)(last_layer)

# %%
last_layer = Dense(256, activation='relu')(last_layer)

# %%
last_layer = Dropout(0.2)(last_layer)

# %%
last_layer = Dense(128, activation='relu')(last_layer)

# %%
last_layer = Dropout(0.2)(last_layer)

# %%
last_layer = Dense(64, activation='relu')(last_layer)

# %%
last_layer = Dropout(0.2)(last_layer)

# %%
last_layer = Dense(32, activation='relu')(last_layer)

# %%
last_layer = Dropout(0.2)(last_layer)

# %%
last_layer = Dense(16, activation='relu')(last_layer)

# %%
last_layer = Dropout(0.2)(last_layer)

# %%
last_layer = Dense(8, activation='relu')(last_layer)

# %%
last_layer = Dropout(0.2)(last_layer)

# %%
last_layer = Dense(1, activation='sigmoid')(last_layer)

# %%
from keras.models import Model

# %%
new_model = Model(inputs = model.input , outputs = last_layer)

# %%
new_model.summary()

# %%
from keras.preprocessing.image import ImageDataGenerator

# %%
train_data=ImageDataGenerator(
    rescale=1./225,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_data=ImageDataGenerator(rescale=1./225)

train_set=train_data.flow_from_directory(
    '/cnn_dataset/training_set/',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary')

test_set= test_data.flow_from_directory(
    '/cnn_dataset/test_set/',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary')


# %%
from keras.optimizers import Adam

new_model.compile(loss = 'binary_crossentropy',  optimizer = 'Adam', metrics = ['accuracy'])

history = new_model.fit(
            train_set,
            steps_per_epoch=200,
            epochs=25,
            validation_data=test_set,
            validation_steps=20)


# %%
train_acc = history.history['accuracy']
test_acc = history.history['val_accuracy']
x = range(50)

import matplotlib.pyplot as plt
plt.plot( train_acc,label= 'train_acc' )
plt.plot( test_acc, label= 'test_acc')

plt.xlabel('x - axis') 
# naming the y axis 

plt.ylabel('y - axis') 
  
# giving a title to my graph 
plt.title('My first graph!') 
plt.legend() 
plt.show()



# %%
train_loss = history.history['loss']
test_loss = history.history['val_loss']

import matplotlib.pyplot as plt
plt.plot(train_loss,label='train_loss' )
plt.plot(test_loss, label='test_loss')

plt.xlabel('x - axis') 
# naming the y axis 

plt.ylabel('y - axis') 
  

plt.title('My loss')

plt.legend()
plt.show()

# %%
from keras.preprocessing import image
import numpy

predict_image = image.load_img('/kaggle/input/cnn_dataset/single_prediction/cat_or_dog_1.jpg', 
               target_size=(64,64))

type(predict_image)

# %%
predict_image = image.img_to_array(predict_image)
predict_image = numpy.expand_dims(predict_image, axis=0)

# %%
from keras.applications.vgg16 import preprocess_input

predict_image = preprocess_input(predict_image)

# %%
result = new_model.predict(predict_image)

# %%
result

# %%
new_model.save('my_face_model.h5')

# %%


# %%
