#!/usr/bin/env python
# coding: utf-8

# # 16: Deep Learning
# ### github : https://github.com/pdeitel/IntroToPython/tree/master/examples/ch16 
# ### book   : http://localhost:8888/files/2241016309/Python%202/Python%20Book.pdf 
#             (only works in lab comp)

# ## 16.1 Introduction:-
# 
# ### Keras and TensorFlow
# Keras offers a friendly interface to Google’s TensorFlow—the most widely used deep-learning library.1 François Chollet of the Google Mind team developed Keras to make deep-learning capabilities more accessible.
# 
# ### Deep Learning Models
# Deep learning models are complex and require an extensive mathematical background tounderstand their inner workings.  
# Keras is to deep learning as Scikit-learn is to machine learning. Each encapsulates the
# sophisticated mathematics, so developers need only define, parameterize and manipulate
# objects. With Keras, you build your models from pre-existing components and quickly
# parameterize those components to your unique requirements. This is what we’ve been
# referring to as object-based programming throughout the book. 
# 
# ### Experiment with Your Models
# Machine learning and deep learning are empirical rather than theoretical fields. You’llexperiment with many models, tweaking them in various ways until you find the models
# that perform best for your applications. Keras facilitates such experimentation.  
# 
# ### Dataset Sizes
# Deep learning works well when you have lots of data, but it also can be effective for smaller datasets when combined with techniques like **transfer learning** and **data augmentation**. **Transfer learning** uses existing knowledge from a previously trained model as the foundation for a new model. **Data augmentation** adds data to a dataset by deriving new data from existing data.  
# For example, in an image dataset, you might rotate the images left and right so the model can learn about objects in different orientations. In general, though, the more data you have, the better you’ll be able to train a deep learning model.
# 
# ### Processing Power
# Deep learning can require significant processing power. Complex models trained on bigdata datasets can take hours, days or even more to train. The models we present in this chapter can be trained in minutes to just less than an hour on computers with conventional CPUs. You’ll need only a reasonably current personal computer. We’ll discuss the special high-performance hardware called GPUs (Graphics Processing Units) and TPUs (Tensor Processing Units) developed by NVIDIA and Google to meet the extraordinary processing demands of edge-of-the-practice deep-learning applications.
# 
# ### Bundled Datasets
# Keras comes packaged with some popular datasets. You’ll work with two of these datasets in the chapter’s examples and several more in the exercises. In this chapter you’ll work with the full **MNIST dataset**. You’ll build a **Keras convolutional neural network(CNN or convnet) model** that will achieve high performance recognizing digit images in the test set. Convnets are especially appropriate for computer vision tasks, such as recognizing handwritten digits and characters or recognizing objects (including faces) in images
# and videos. 
# 
# You’ll also work with a **Keras recurrent neural network**. In that example, you’ll perform **sentiment analysis** using the **IMDb Movie reviews dataset**, in which the reviews in the training and testing sets are labeled as positive or negative.
# 
# ### Future of Deep Learning
# Newer automated deep learning capabilities are making it even easier to build deep-learning solutions. These include **Auto-Keras** from Texas A&M University’s DATA Lab, Baidu’s **EZDL** and Google’s **AutoML**. You’ll explore Auto-Keras in the exercises. 

# ## 16.1.1 Deep Learning Applications
# Deep learning is being used in a wide range of applications, such as:
# * Game playing  
# * Computer vision: Object recognition, pattern recognition, facial recognition  
# * Self-driving cars  
# * Robotics  
# * Improving customer experiences  
# * Chatbots  
# * Diagnosing medical conditions  
# * Google Search  
# * Facial recognition  
# * Automated image captioning and video closed captioning  
# * Enhancing image resolution  
# * Speech recognition  
# * Language translation  
# * Predicting election results  
# * Predicting earthquakes and weather
# * Google Sunroof to determine whether you can put solar panels on your roof
# * Generative applications—Generating original images, processing existing images to look like a specified artist’s style, adding color to black-and-white images and video, creating music, creating text (books, poetry) and much more.

# ## 16.1.2 Deep Learning Demos
# Check out these four deep-learning demos and search online for lots more, including practical applications like we mentioned in the preceding section:
# * DeepArt.io—Turn a photo into artwork by applying an art style to the photo. 
# > https://deepart.io/.
# 
# * DeepWarp Demo—Analyzes a person’s photo and makes the person’s eyes movein different directions. 
# > https://sites.skoltech.ru/sites/compvision_wiki/static_pages/projects/deepwarp/.
# 
# * Image-to-Image Demo—Translates a line drawing into a picture. 
# > https://affinelayer.com/pixsrv/.
# 
# * Google Translate Mobile App (download from an app store to your smartphone)—Translate text in a photo to another language (e.g., take a photo of a sign or a restaurant menu in Spanish and translate the text to English).

# ## 16.4 Neural Networks
# Deep learning is a form of machine learning that uses artificial neural networks to learn. An **artificial neural network** (or just neural network) is a software construct that operates similarly to how scientists believe our brains work. Our biological nervous systems are controlled via neurons that communicate with one another along pathways called synapses. As we learn, the specific neurons that enable us to perform a given task, like walking, communicate with one another more efficiently. These neurons activate anytime we need to walk.
# 
# ### Artificial Neurons
# In a neural network, interconnected **artificial neurons** simulate the human brain’s neurons to help the network learn. The connections between specific neurons are reinforced during the learning process with the goal of achieving a specific result. In **supervised deep learning**—which we’ll use in this chapter—we aim to predict the target labels supplied with data samples. To do this, we’ll train a general neural network model that we can then use
# to make predictions on unseen data.
# 
# ### Artificial Neural Network Diagram
# The following diagram shows a three-layer neural network. Each circle represents a neuron, and the lines between them simulate the synapses. The output of a neuron becomes the input of another neuron, hence the term neural network. This particular diagram shows a **fully connected network**—every neuron in a given layer is connected to all the neurons in the next layer:
# 
# ![cnn](image1.png) 

# ### Learning Is an Iterative Process
# When you were a baby, you did not learn to walk instantaneously. You learned that process over time with repetition. You built up the smaller components of the movements that enabled you to walk—learning to stand, learning to balance to remain standing, learning to lift your foot and move it forward, etc. And you got feedback from your environment. When you walked successfully your parents smiled and clapped. When you fell, you might have bumped your head and felt pain.  
# Similarly, we train neural networks iteratively over time. Each iteration is known as an **epoch** and processes every sample in the training dataset once. There’s no “correct” number of epochs. This is a hyperparameter that may need tuning, based on your training data and your model. The inputs to the network are the features in the training samples. Some layers learn new features from previous layers’ outputs and others interpret those features to make predictions. 

# ### How Artificial Neurons Decide Whether to Activate Synapses
# During the training phase, the network calculates values called **weights** for every connection between the neurons in one layer and those in the next. On a neuron-by-neuron basis, each of its inputs is multiplied by that connection’s weight, then the sum of those weighted inputs is passed to the neuron’s **activation function**. This function’s output determines which neurons to activate based on the inputs—just like the neurons in your brain passing information around in response to inputs coming from your eyes, nose, ears and more.  
# The following diagram shows a neuron receiving three inputs (the black dots) and producing an output (the hollow circle) that would be passed to all or some of neurons in the next layer, depending on the types of the neural network’s layers:
#   
# ![cnn2](image2.png)
#   
# The values w1, w2 and w3 are weights. In a new model that you train from scratch, these values are initialized randomly by the model. As the network trains, it tries to minimize the error rate between the network’s predicted labels and the samples’ actual labels. The error rate is known as the **loss**, and the calculation that determines the loss is called the **loss function**. Throughout training, the network determines the amount that each neuron contributes to the overall loss, then goes back through the layers and adjusts the weights in an effort to minimize that loss. This technique is called **backpropagation**. Optimizing these weights occurs gradually—typically via a process called **gradient descent**.

# In[ ]:




