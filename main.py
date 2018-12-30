import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import scipy
from scipy import misc,ndimage
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential  
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.callbacks import ModelCheckpoint
%matplotlib inline

train_path='dataset/training_set'
valid_path='dataset/valid_set'
test_path='dataset/test_set'
img_width,img_height=224,224
epochs=50
batch_size=10

datagen=ImageDataGenerator(rescale=1./255)

from keras import applications
base_model=applications.InceptionV3(include_top=False,weights='imagenet')

generator = datagen.flow_from_directory(  
    train_path,  
    target_size=(img_width, img_height),  
    batch_size=batch_size,  
    class_mode=None,  
    shuffle=False)  
   
numb_train_samples = len(generator.filenames)  
numb_classes = len(generator.class_indices)  
   
predict_size_train = int(math.ceil(numb_train_samples / batch_size))  
   
bottleneck_features_train = base_model.predict_generator(  
    generator, predict_size_train,verbose=1)  
   
np.save('bottleneck_features_train_inc.npy', bottleneck_features_train)  

generator = datagen.flow_from_directory(  
     valid_path,  
     target_size=(img_width, img_height),  
     batch_size=batch_size,  
     class_mode=None,  
     shuffle=False)  
   
numb_validation_samples = len(generator.filenames)  
   
predict_size_validation = int(math.ceil(numb_validation_samples / batch_size))  
   
bottleneck_features_validation = base_model.predict_generator(  
     generator, predict_size_validation)  
   
np.save('bottleneck_features_validation_inc.npy', bottleneck_features_validation)

generator_top = datagen.flow_from_directory(  
         train_path,  
         target_size=(img_width, img_height),  
         batch_size=batch_size,  
         class_mode='categorical',  
         shuffle=False)  
   
numb_train_samples = len(generator_top.filenames)  
numb_classes = len(generator_top.class_indices)  
    
train_data = np.load('bottleneck_features_train_inc.npy')   
train_labels = generator_top.classes     
train_labels = to_categorical(train_labels, num_classes=numb_classes) 

generator_top = datagen.flow_from_directory(  
         valid_path,  
         target_size=(img_width, img_height),  
         batch_size=batch_size,  
         class_mode=None,  
         shuffle=False)  
   
numb_validation_samples = len(generator_top.filenames)  
   
validation_data = np.load('bottleneck_features_validation_inc.npy')  
   
validation_labels = generator_top.classes  
validation_labels = to_categorical(validation_labels, num_classes=numb_classes)  

#model = Sequential()
#model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
#model.add(Dense(21, activation='softmax'))
#model.summary()

model = Sequential()  
model.add(Flatten(input_shape=train_data.shape[1:]))  
model.add(Dense(256, activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(num_classes, activation='softmax')) 

from keras.optimizers import Adam,Adamax
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuraccy'])

checkpointer = ModelCheckpoint(filepath='weights_from_transfer_incv3.hdf5',verbose=1,
                                save_best_only=True)
								
model.fit(train_data,train_labels,
                    validation_data=(validation_data,validation_labels),
                    epochs=epochs,batch_size=batch_size,callbacks=[checkpointer],verbose=1)
					
generator = datagen.flow_from_directory(  
    test_path,  
    target_size=(img_width, img_height),  
    batch_size=batch_size,  
    class_mode=None,  
    shuffle=False)  

nb_test_samples = len(generator.filenames)  
predict_size_test = int(math.ceil(nb_test_samples/batch_size))     
bottleneck_features_test = base_model.predict_generator( generator, 
    predict_size_test, verbose=1)  
   
np.save('bottleneck_features_test_1000inc.npy', bottleneck_features_test)

generator_top = datagen.flow_from_directory(  
         test_path,  
         target_size=(img_width, img_height),  
         batch_size=batch_size,  
         class_mode=None,  
         shuffle=False)  
   
nb_test_samples = len(generator_top.filenames)  
   
test_data = np.load('bottleneck_features_test_1000inc.npy')  
   
test_labels = generator_top.classes  
test_labels = to_categorical(test_labels, num_classes=num_classes) 

tr_predictions=[np.argmax(res_model.predict(np.expand_dims(feature,axis=0)))for feature in test_data]
test_accuracy = 100*np.sum(np.array(tr_predictions)==np.argmax(test_labels, axis=1))/len(tr_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
