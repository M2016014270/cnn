# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# Initialising the CNN
classifier = Sequential()


# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# 64 camadas de convolução
# matriz 3x3
# imagem de entrada sendo forçada a ter tamanho 64x64x3 (3 == rgb)

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# pooling das matrizes 2x2

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu')) # mesma coisa só q não precisa forçar o tamanho (shape) da iagem q vai entrar na rede
classifier.add(MaxPooling2D(pool_size = (2, 2))) # mesma coisa

# Step 3 - Flattening
classifier.add(Flatten())#transformar os parametros de transformação em vetor linha (ou coluna, tanto faz)

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu')) #conectanto os parametros de entrada da rede (128 - quantidade de parametros entrada)
classifier.add(Dense(units = 1, activation = 'sigmoid'))#conectando os parametros de saida (por isso q tem 1)

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # funções do compilador
# como é um classificador binário (gato ou cachorro), o loss é 'binary_crossentropy'

# Part 2 - Fitting the CNN to the images 

# adicionando as imagens a rede

from keras.preprocessing.image import ImageDataGenerator

#tudo isso vai ser feito em algumas imagens aleatórias da pasta de treino

train_datagen = ImageDataGenerator(rescale = 1./255, # reduzindo a imagem
                                   shear_range = 0.2, # movendo os pixels (0.2 para alguma direção)
                                   zoom_range = 0.2, # dando zoom na imagem
                                   horizontal_flip = True) # girando a imagem na horizontal (não sei quantos graus)

test_datagen = ImageDataGenerator(rescale = 1./255) # reduzindo a imagem

training_set = train_datagen.flow_from_directory('Convolutional_Neural_Networks/Convolutional_Neural_Networks/dataset/training_set', # diretório da pasta com as imagens para treino
                                                 target_size = (64, 64), # imagem de entrada sendo forçada a ter tamanho 64x64
                                                 batch_size = 2, # 32 camadas de convolução
                                                 class_mode = 'binary') #classes da rede


test_set = test_datagen.flow_from_directory('Convolutional_Neural_Networks/Convolutional_Neural_Networks/dataset/test_set', # diretório da pasta com as imagens para teste
                                            target_size = (64, 64), # imagem de entrada sendo forçada a ter tamanho 64x64
                                            batch_size = 2, # 32 camadas de convolução
                                            class_mode = 'binary') #classes da rede

classifier.fit(training_set,
                         steps_per_epoch = len(training_set),
                         epochs = 2, #colocar 100
                         validation_data = test_set,
                         validation_steps = len(test_set))

    #fit_generator
lista = []
    
    lista.append(history.history.keys())

#%%

print(lista)

arq = open("history.txt", "w")
arq.write("{}".format(lista))
#para inserir quebra de linha
arq.write("\n")

#%%

# Part 4 - Save model

classifier.save('camadas_64/32_epoch_2.h5')

