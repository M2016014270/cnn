from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = (64,64,3), activation='elu'))
model.add(Conv2D(64, (3,3), activation='elu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), activation = 'elu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), activation = 'elu')) 
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(512, activation='elu'))
model.add(Dropout(0.45))
model.add(Dense(128, activation = 'elu'))
model.add(Dense(units = 3, activation = 'elu'))


model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2, 
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training_set', # diretório da pasta com as imagens para treino
                                                 target_size = (64, 64), # imagem de entrada sendo forçada a ter tamanho 64x64
                                                 batch_size = 8, # 32 camadas de convolução
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('test_set',
    target_size = (64, 64),
    batch_size = 8,
    class_mode = 'categorical'
)

history = model.fit_generator(training_set,
                         steps_per_epoch = len(training_set),
                         epochs = 20,
                         validation_data = test_set,
                         validation_steps = len(test_set))

model.save('64_camadas_20_epochs.h5')

#plotar grafico das curvas de erro e acerto

import matplotlib.pyplot as plt


# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)

plt.show()

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot1.png', show_shapes=True, show_layer_names=True)
