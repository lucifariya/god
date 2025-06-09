#!/usr/bin/env python
# coding: utf-8

# # Section 16.7

# In[ ]:


from tensorflow.keras.datasets import mnist


# In[ ]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_test.shape


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


import seaborn as sns


# In[ ]:


sns.set(font_scale=2)


# In[ ]:


import numpy as np
index = np.random.choice(np.arange(len(X_train)), 24, replace=False)
figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(16, 9))

for item in zip(axes.ravel(), X_train, y_train):
    axes, image, target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])  # remove x-axis tick marks
    axes.set_yticks([])  # remove y-axis tick marks
    axes.set_title(target)
plt.tight_layout()


# In[ ]:


X_train = X_train.reshape((60000, 28, 28, 1)) 


# In[ ]:


X_train.shape


# In[ ]:


X_train = X_train.astype('float32') / 255


# In[ ]:


X_test = X_test.reshape((10000, 28, 28, 1))


# In[ ]:


X_test.shape


# In[ ]:


X_test = X_test.astype('float32') / 255


# In[ ]:


from tensorflow.keras.utils import to_categorical


# In[ ]:


y_train = to_categorical(y_train)


# In[ ]:


y_train.shape


# In[ ]:


y_train[0]


# In[ ]:


y_test = to_categorical(y_test)


# In[ ]:


y_test.shape


# In[ ]:


from tensorflow.keras.models import Sequential


# In[ ]:


cnn = Sequential()


# In[ ]:


from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D


# In[ ]:


cnn.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', 
               input_shape=(28, 28, 1)))


# In[ ]:


cnn.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:


cnn.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))


# In[ ]:


cnn.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:


cnn.add(Flatten())


# In[ ]:


cnn.add(Dense(units=128, activation='relu'))


# In[ ]:


cnn.add(Dense(units=10, activation='softmax'))


# In[ ]:


cnn.summary()


# In[ ]:


from tensorflow.keras.utils import plot_model
from IPython.display import Image
plot_model(cnn, to_file='convnet.png', show_shapes=True, 
           show_layer_names=True)
Image(filename='convnet.png') 


# In[ ]:


cnn.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])


# In[ ]:


cnn.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)


# In[ ]:


loss, accuracy = cnn.evaluate(X_test, y_test)


# In[ ]:


loss


# In[ ]:


accuracy


# In[ ]:


predictions = cnn.predict(X_test)


# In[ ]:


y_test[0]


# In[ ]:


for index, probability in enumerate(predictions[0]):
    print(f'{index}: {probability:.10%}')


# In[ ]:


images = X_test.reshape((10000, 28, 28))
incorrect_predictions = []

for i, (p, e) in enumerate(zip(predictions, y_test)):
    predicted, expected = np.argmax(p), np.argmax(e)
    
    if predicted != expected:
        incorrect_predictions.append((i, images[i], predicted, expected))


# In[ ]:


len(incorrect_predictions)


# In[ ]:


figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(16, 12))

for axes, item in zip(axes.ravel(), incorrect_predictions):
    index, image, predicted, expected = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])  # remove x-axis tick marks
    axes.set_yticks([])  # remove y-axis tick marks
    axes.set_title(f'index: {index}\np: {predicted}; e: {expected}')
plt.tight_layout()


# In[ ]:


def display_probabilities(prediction):
    for index, probability in enumerate(prediction):
        print(f'{index}: {probability:.10%}')


# In[ ]:


display_probabilities(predictions[495])


# In[ ]:


display_probabilities(predictions[583])


# In[ ]:


display_probabilities(predictions[625])


# In[ ]:


cnn.save('mnist_cnn.h5')


# In[ ]:


##########################################################################
# (C) Copyright 2019 by Deitel & Associates, Inc. and                    #
# Pearson Education, Inc. All Rights Reserved.                           #
#                                                                        #
# DISCLAIMER: The authors and publisher of this book have used their     #
# best efforts in preparing the book. These efforts include the          #
# development, research, and testing of the theories and programs        #
# to determine their effectiveness. The authors and publisher make       #
# no warranty of any kind, expressed or implied, with regard to these    #
# programs or to the documentation contained in these books. The authors #
# and publisher shall not be liable in any event for incidental or       #
# consequential damages in connection with, or arising out of, the       #
# furnishing, performance, or use of these programs.                     #
##########################################################################

