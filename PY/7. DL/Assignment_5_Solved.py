#!/usr/bin/env python
# coding: utf-8

# # MINOR ASSIGNMENT-5: DEEP LEARNING  
# **Course:** Python for Computer Science and Data Science 2 (CSE 3652)  
# **Institute:** Centre for Data Science, Institute of Technical Education & Research, SOA  
# 

# In[1]:


get_ipython().system('pip install numpy pandas scikit-learn tensorflow matplotlib')


# ### **Question 1:**  
# **Explain briefly Single layer perceptron and multilayer perceptron with architecture and illustrate the loss function associated with it.**
# 
# #### 🧠 Single Layer Perceptron:
# - Consists of only one layer of weights connecting inputs to outputs.
# - Suitable only for linearly separable problems.
# - Activation: Step or Sign Function.
# - **Loss Function**: Mean Squared Error (MSE)
# 
# #### 🧠 Multilayer Perceptron (MLP):
# - Composed of input layer, one or more hidden layers, and an output layer.
# - Can solve complex, non-linear problems.
# - Activation: Typically ReLU or sigmoid in hidden layers; softmax in output layer.
# - **Loss Function**: Cross-entropy for classification tasks
# 

# ### **Question 2:**  
# **How would you define the architecture of a simple feed forward ANN for classifying the Iris dataset? Write python code for the same.**
# 
# #### ✅ How to design the architecture:
# How to design the architecture:
# 1. Understand the dataset:
#     * The Iris dataset has 4 numerical input features (sepal length, sepal width, petal length, petal width).
#     * There are 3 output classes: Setosa, Versicolor, Virginica.
# 
# 2. Input layer:
#     * Takes in the 4 features. So, the input shape is `(4,)`.
# 
# 3. Hidden layer(s):
#     * A small number of neurons (e.g., 8 to 10) is typically sufficient due to the dataset's simplicity.
#     * Activation function: `ReLU (Rectified Linear Unit)` introduces non-linearity.
# 
# 4. Output layer:
#     * Since there are 3 classes, use 3 neurons with a softmax activation to output class probabilities.
# 
# 5. Loss Function:
#     * Use categorical cross-entropy, suitable for multiclass classification.
# 
# 6. Optimizer:
#     * Use Adam, a commonly used and efficient optimizer.
# 
# 7. Preprocessing:
#     * Normalize the input features using StandardScaler to improve training performance.
# 

# In[2]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

iris = load_iris()
X = iris.data
y = to_categorical(iris.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(10, input_shape=(4,), activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=1)


# ### **Question 3:**  
# **How can you build and train a simple Artificial Neural Network (ANN) using the MNIST dataset to classify handwritten digits? Write python code for this.**
# 
# #### ✅ How to approach it:
# 1. Understand the MNIST dataset:
#     * It contains 28x28 grayscale images of handwritten digits (0–9).
#     * Each image is flattened into a 784-length vector (`28*28`) for ANN input.
# 
# 2. Preprocess the data:
#     * Normalize pixel values to range `[0, 1]` by dividing by 255.
#     * One-hot encode the labels using `to_categorical`.
# 
# 3. Define the model:
#     * Flatten layer converts 2D input into 1D.
#     * Dense layer (e.g., 128 neurons) with `ReLU` to learn features.
#     * Output Dense layer (10 neurons) with `softmax` to output class probabilities.
# 
# 4. Compile the model:
#     * Loss: `categorical_crossentropy` for multi-class classification.
#     * Optimizer: `adam`.
# 
# 5. Train the model:
#     * Use `.fit()` with training data, epochs (e.g., 10), and batch size (e.g., 32).
#     * Optionally include validation split or test data for performance monitoring.
# 

# In[3]:


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))


# ### **Question 4:**  
# **Find convolution, ReLU and Max Pooling with the following data**  
# **Input (4×4):**  
# ```
# [[1, 2, 0, 1],
#  [3, 1, 2, 2],
#  [1, 0, 1, 3],
#  [2, 1, 2, 1]]
# ```
# 
# **Filter (2×2):**  
# ```
# [[1, 0],
#  [0, -1]]
# ```
# 
# #### ✅ Solution:
# 
# **Step 1: Convolution Output (Stride=1, Valid Padding)**  
# ```
# [[ 0,  0, -2],
#  [ 3,  0, -1],
#  [ 0, -2,  0]]
# ```
# 
# **Step 2: ReLU Activation:**  
# ```
# [[0, 0, 0],
#  [3, 0, 0],
#  [0, 0, 0]]
# ```
# 
# **Step 3: Max Pooling (2×2, stride=1):**  
# ```
# [[3, 0],
#  [0, 0]]
# ```
# 

# ### **Question 5:**  
# **How can you build a Convolutional Neural Network (CNN) with two convolutional layers and one fully connected hidden layer to classify handwritten digits from the MNIST dataset?**
# 
# #### ✅ How to approach it:
# 1. Why CNN instead of ANN?
#     * CNNs preserve spatial relationships in image data.
#     * They are more efficient for image recognition tasks like MNIST.
# 2. Preprocess the data:
#     * Reshape input to `(28, 28, 1)` to add a channel dimension.
#     * Normalize pixel values to `[0, 1]`.
# 3. Define the model architecture:
#     * Conv2D layer 1: Detects basic features (edges, corners).
#         * Filters: 32, Kernel size: (3x3), Activation: `ReLU`.
#     * MaxPooling2D layer 1: Downsamples the feature map.
#     * Conv2D layer 2: Learns more complex features.
#         * Filters: 64, Kernel size: (3x3), Activation: `ReLU`.
#     * MaxPooling2D layer 2: Further downsampling.
#     * Flatten the 3D output to 1D.
#     * Dense layer (fully connected): Typically 128 neurons with `ReLU`.
#     * Output Dense layer: 10 neurons with `softmax` for digit classification.
# 4. Compile the model:
#     * Loss: `categorical_crossentropy`
#     * Optimizer: `adam`
# 5. Train the model:
#     * Use `.fit()` with suitable batch size and number of epochs (e.g., 5–10).
#     * Validate with test set or a validation split.

# In[4]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

