#Importando os parametros salvos da rede previamente treinada
from keras.models import load_model

classifier = load_model('/home/eu/cnn/32_camadas_3_epochs.h5')

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('/home/eu/cnn/0.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)

if result[0][0] > 0.90:
    prediction = 'dog'
    print('é cachorro!')
if result[0][1] > .90:
    prediction = 'cat'
    print('é gato!')
else:
    prediction = 'leao'
    print('sobrou leao!')

print(result)