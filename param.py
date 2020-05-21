

from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.applications.vgg16 import VGG16
from keras_vggface.vggface import VGGFace
from keras.preprocessing.image import ImageDataGenerator

face_model = VGGFace(model='vgg16', 
                weights='vggface',
                input_shape=(224,224,3)) 

person_count = 5

last_layer = face_model.get_layer('pool5').output

x = Flatten(name='flatten')(last_layer)
x = Dense(1024, activation='relu', name='fc6')(x)
x = Dense(1024, activation='relu', name='fc7')(x)
out = Dense(person_count, activation='softmax', name='fc8')(x)

custom_face = Model(face_model.input, out)

#custom_face.summary()
