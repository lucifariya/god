#!/usr/bin/env python
# coding: utf-8

# # Machine Learning: Classification, Regression and Clustering 
# ### github : https://github.com/pdeitel/IntroToPython/tree/master/examples/ch15 
# ### book   : http://localhost:8888/files/2241016309/Python%202/Python%20Book.pdf 
#             (only works in lab comp)

# In[1]:


get_ipython().system('pip install scikit-learn')


# ### 15.2.2 Loading the Dataset
# The **load_digits** function from the **sklearn.datasets module** returns a scikit-learn
# **Bunch** object containing the digits data and information about the Digits dataset (called
# **metadata**):

# In[2]:


from sklearn.datasets import load_digits


# In[3]:


digits = load_digits()


# **Bunch** is a subclass of dict that has additional attributes for interacting with the dataset.
#   
# #### Displaying the Description  
#   
# The Digits dataset bundled with scikit-learn is a subset of the UCI (University of California Irvine) ML hand-written digits dataset at:
# > http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits  
#   
# The original UCI dataset contains 5620 samples—3823 for training and 1797 for testing. The version of the dataset bundled with scikit-learn contains only the 1797 testing samples. A Bunch’s **DESCR attribute** contains a description of the dataset. According to the Digits dataset’s description4, each sample has 64 features (as specified by Number of Attributes) that represent an 8-by-8 image with pixel values in the range 0–16 (specified by Attribute Information). This dataset has no missing values (as specified by Missing Attribute Values). The 64 features may seem like a lot, but real-world datasets can sometimes have hundreds, thousands or even millions of features. 

# In[4]:


print(digits.DESCR)    # DESCR bole to description


# #### Checking the Sample and Target Sizes  
# The Bunch object’s **data** and **target attributes** are NumPy arrays:  
# * The data array contains the 1797 samples (the digit images), each with 64 features, having values in the range 0–16, representing pixel intensities. With Matplotlib, we’ll visualize these intensities in grayscale shades from white (0) to black (16):  
# * The target array contains the images’ labels—that is, the classes indicating which digit each image represents. The array is called target because, when you make predictions, you’re aiming to “hit the target” values. To see labels of samples throughout the dataset, let’s display the target values of every 100th sample:

# In[5]:


digits.target[::100]


# We can confirm the number of samples and features (per sample) by looking at the
# data array’s shape attribute, which shows that there are 1797 rows (samples) and 64 columns (features):

# In[6]:


digits.data.shape


# You can confirm that the number of target values matches the number of samples by looking at the target array’s shape:

# In[7]:


digits.target.shape


# #### A Sample Digit Image
# Each image is two-dimensional—it has a width and a height in pixels. The Bunch object
# returned by load_digits contains an images attribute—an array in which each element
# is a two-dimensional 8-by-8 array representing a digit image’s pixel intensities. Though
# the original dataset represents each pixel as an integer value from 0–16, scikit-learn stores
# these values as floating-point values (NumPy type float64). For example, here’s the twodimensional array representing the sample image at index 13:

# In[8]:


digits.images[13]


# #### Preparing the Data for Use with Scikit-Learn
# Scikit-learn’s machine-learning algorithms require samples to be stored in a two-dimensional array of floating-point values (or two-dimensional array-like collection, such as a list
# of lists or a pandas DataFrame):  
# * Each row represents one sample.  
# * Each column in a given row represents one feature for that sample.  
# To represent every sample as one row, multi-dimensional data like the two-dimensional
# image array shown in previous shell must be flattened into a one-dimensional array.  
#   
# The load_digits function returns the preprocessed data ready
# for machine learning. The Digits dataset is numerical, so load_digits simply flattens each
# image’s two-dimensional array into a one-dimensional array. For example, the 8-by-8
# array digits.images[13] shown in previous shell corresponds to the 1-by-64 array digits.data[13] shown below: 

# In[9]:


digits.data[13]


# ### 15.2.3 Visualizing the Data  
#   
# You should always familiarize yourself with your data. This process is called **data exploration**. For the digit images, you can get a sense of what they look like by displaying them with the Matplotlib implot function.  
#   
# #### Creating the Diagram:

# In[10]:


import matplotlib.pyplot as plt


# In[11]:


figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))


# In[12]:


figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))

### Displaying Each Image and Removing the Axes Labels 

for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])  # remove x-axis tick marks
    axes.set_yticks([])  # remove y-axis tick marks
    axes.set_title(target)
plt.tight_layout() 


# ### 15.2.4 Splitting the Data for Training and Testing  
#   
# You typically train a machine-learning model with a subset of a dataset. Typically, **the
# more data you have for training, the better you can train the model**. It’s important to set
# aside a portion of your data for testing, so you can evaluate a model’s performance using
# data that the model has not yet seen. Once you’re confident that the model is performing
# well, you can use it to make predictions using new data it hasn’t seen.  
#   
# We first break the data into a **training set** and a **testing set** to prepare to train and test
# the model. The function **train_test_split** from the **sklearn.model_selection** module
# shuffles the data to randomize it, then splits the samples in the data array and the target
# values in the target array into training and testing sets. This helps ensure that the training
# and testing sets have similar characteristics. The shuffling and splitting is performed conveniently for you by a **ShuffleSplit** object from the sklearn.model_selection module.  
#   
# Function train_test_split returns a tuple of four elements in which the first two are the
# samples split into training and testing sets, and the last two are the corresponding target
# values split into training and testing sets. By convention, uppercase X is used to represent
# the samples, and lowercase y is used to represent the target values: 

# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=11)


# We assume the data has **balanced classes**—that is, the samples are divided evenly among
# the classes. This is the case for each of scikit-learn’s bundled classification datasets. Unbalanced classes could lead to incorrect results.  
#   
# In the “Functions” chapter, you saw how to seed a random-number generator for
# reproducibility. In machine-learning studies, this helps others confirm your results by
# working with the same randomly selected data. Function train_test_split provides the
# keyword argument random_state for reproducibility. When you run the code in the future
# with the same seed value, train_test_split will select the same data for the training set
# and the same data for the testing set. We chose the seed value (11) arbitrarily.  
#   
# #### Training and Testing Set Sizes  
# Looking at X_train’s and X_test’s shapes, you can see that, by default, train_test_split reserves 75% of the data for training and 25% for testing:   

# In[15]:


X_train.shape


# In[16]:


X_test.shape


# To specify different splits, you can set the sizes of the testing and training sets with the train_test_split function’s keyword arguments test_size and train_size. Use floating-point values from 0.0 through 1.0 to specify the percentages of the data to use for each.  
# You can use integer values to set the precise numbers of samples. If you specify one of these keyword arguments, the other is inferred. For example, the statement
# ```python
# X_train, X_test, y_train, y_test = train_test_split(
#  digits.data, digits.target, random_state=11, test_size=0.20)
# ```  
# specifies that 20% of the data is for testing, so train_size is inferred to be 0.80.  
#   
# ### 15.2.5 Creating the Model  
#   
#  The **KNeighborsClassifier** estimator (module **sklearn.neighbors**) implements the knearest neighbors algorithm. First, we create the KNeighborsClassifier estimator object: 

# In[17]:


from sklearn.neighbors import KNeighborsClassifier


# In[18]:


knn = KNeighborsClassifier()


# To create an estimator, you simply create an object. The internal details of how this object
# implements the k-nearest neighbors algorithm are hidden in the object. You’ll simply call
# its methods. This is the essence of Python object-based programming.
# 
# ### 15.2.6 Training the Model   
#   
# Next, we invoke the KNeighborsClassifier object’s **fit method**, which loads the sample
# training set (X_train) and target training set (y_train) into the estimator:

# In[19]:


knn.fit(X=X_train, y=y_train)


# For most, scikit-learn estimators, the fit method loads the data into the estimator
# then uses that data to perform complex calculations behind the scenes that learn from the
# data and train the model. The KNeighborsClassifier’s fit method just loads the data
# into the estimator, because k-NN actually has no initial learning process. The estimator is
# said to be **lazy** because its work is performed only when you use it to make predictions. In
# this and the next chapter, you’ll use lots of models that have significant training phases. In
# the real-world machine-learning applications, it can sometimes take minutes, hours, days
# or even months to train your models. We’ll see in the next chapter, “Deep Learning,” that
# special-purpose, high-performance hardware called GPUs and TPUs can significantly
# reduce model training time.  
#   
# As shown in prev shell's output, the fit method returns the estimator, so IPython
# displays its string representation, which includes the estimator’s default settings. The
# n_neighbors value corresponds to k in the k-nearest neighbors algorithm. By default, a
# KNeighborsClassifier looks at the five nearest neighbors to make its predictions. For
# simplicity, we generally use the default estimator settings. For KNeighborsClassifier,
# these are described at:  
# > http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html  
#   
# Many of these settings are beyond the scope of this book. In Part 2 of this case study, we’ll
# discuss how to choose the best value for n_neighbors. 
#   
# ### 15.2.7 Predicting Digit Classes  
#   
# Now that we’ve loaded the data into the KNeighborsClassifier, we can use it with the test samples to make predictions. Calling the estimator’s **predict method** with X_test as an argument returns an array containing the predicted class of each test image: 

# In[20]:


predicted = knn.predict(X=X_test)


# In[21]:


expected = y_test


# Let’s look at the predicted digits vs. expected digits for the first 20 test samples: 

# In[22]:


predicted[:20]


# In[23]:


expected[:20]


# As you can see, in the first 20 elements, only the predicted and expected arrays’ values at index 18 do not match. We expected a 3, but the model predicted a 5.  
# Let’s use a list comprehension to locate all the incorrect predictions for the entire test set—that is, the cases in which the predicted and expected values do not match: 

# In[24]:


wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]


# In[25]:


wrong


# The list comprehension uses zip to create tuples containing the corresponding elements
# in predicted and expected. We include a tuple in the result only if its p (the predicted
# value) and e (the expected value) differ—that is, the predicted value was incorrect. In this
# example, the estimator incorrectly predicted only 10 of the 450 test samples. So the prediction accuracy of this estimator is an impressive 97.78%, even though we used only the
# estimator’s default parameters.   
#   
# ## 15.3 Case Study: Classification with k-Nearest Neighbors and the Digits Dataset, Part 2  
#   
# In this section, we continue the digit classification case study. We’ll:  
# * evaluate the k-NN classification estimator’s accuracy,  
# * execute multiple estimators and can compare their results so you can choose the best one(s), and  
# * show how to tune k-NN’s hyperparameter k to get the best performance out of a KNeighborsClassifier.  
#   
# ### 15.3.1 Metrics for Model Accuracy   
# Once you’ve trained and tested a model, you’ll want to measure its accuracy. Here, we’ll look at two ways of doing this—a classification estimator’s score method and a confusion matrix. 
#   
# #### Estimator Method `score`  
# Each estimator has a **score** method that returns an indication of how well the estimator performs for the test data you pass as arguments. For classification estimators, this method returns the prediction accuracy for the test data:

# In[26]:


print(f'{knn.score(X_test, y_test):.2%}')


# The kNeighborsClassifier’s with its default k (that is, n_neighbors=5) achieved 97.78% prediction accuracy. Shortly, we’ll perform hyperparameter tuning to try to determine the optimal value for k, hoping that we get even better accuracy.  
#   
# #### Confusion Matrix  
# Another way to check a classification estimator’s accuracy is via a **confusion matrix**, which shows the correct and incorrect predicted values (also known as the hits and misses) for a given class. Simply call the function **confusion_matrix** from the **sklearn.metrics module**, passing the expected classes and the predicted classes as arguments, as in: 

# In[27]:


from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_true=expected, y_pred=predicted)


# The y_true keyword argument specifies the test samples’ actual classes. People looked at the dataset’s images and labeled them with specific classes (the digit values). The y_pred keyword argument specifies the predicted digits for those test images.   
#   
# Below is the confusion matrix produced by the preceding call. The correct predictions are shown on the diagonal from top-left to bottom-right. This is called the **principal diagonal**. The nonzero values that are not on the principal diagonal indicate incorrect predictions: 

# In[28]:


confusion


# Each row represents one distinct class—that is, one of the digits 0–9. The columns within a row specify how many of the test samples were classified into each distinct class.  
#   
# For example, row 0:
# > [45, 0, 0, 0, 0, 0, 0, 0, 0, 0]  
#   
# represents the digit 0 class. The columns represent the ten possible target classes 0 through 9. Because we’re working with digits, the classes (0–9) and the row and column index numbers (0–9) happen to match. According to row 0, 45 test samples were classified as the digit 0, and none of the test samples were misclassified as any of the digits 1 through 9. So 100% of the 0s were correctly predicted.  
#   
# On the other hand, consider row 8 which represents the results for the digit 8:
# > [ 0, 1, 1, 2, 0, 0, 0, 0, 39, 1]  
#   
# * The 1 at column index 1 indicates that one 8 was incorrectly classified as a 1.  
# * The 1 at column index 2 indicates that one 8 was incorrectly classified as a 2.  
# * The 2 at column index 3 indicates that two 8s were incorrectly classified as 3s.  
# * The 39 at column index 8 indicates that 39 8s were correctly classified as 8s.  
# * The 1 at column index 9 indicates that one 8 was incorrectly classified as a 9.  
#   
# So the algorithm correctly predicted 88.63% (39 of 44) of the 8s. Earlier we saw that the
# overall prediction accuracy of this estimator was 97.78%. The lower prediction accuracy
# for 8s indicates that they’re apparently harder to recognize than the other digits.
#   
# #### Classification Report  
#   
# The sklearn.metrics module also provides function **classification_report**, which produces a table of **classification metrics** based on the expected and predicted values:

# In[29]:


from sklearn.metrics import classification_report


# In[30]:


names = [str(digit) for digit in digits.target_names]


# In[31]:


print(classification_report(expected, predicted, target_names=names))


# In the report:  
# * precision is the total number of correct predictions for a given digit divided by the total number of predictions for that digit. You can confirm the precision by looking at each column in the confusion matrix. For example, if you look at column index 7, you’ll see 1s in rows 3 and 4, indicating that one 3 and one 4 were incorrectly classified as 7s and a 45 in row 7 indicating the 45 images were correctly classified as 7s. So the precision for the digit 7 is 45/47 or 0.96.  
# * recall is the total number of correct predictions for a given digit divided by the total number of samples that should have been predicted as that digit. You can confirm the recall by looking at each row in the confusion matrix. For example, if you look at row index 8, you’ll see three 1s and a 2 indicating that some 8s were incorrectly classified as other digits and a 39 indicating that 39 images were correctly classified. So the recall for the digit 8 is 39/44 or 0.89.  
# * f1-score—This is the average of the precision and the recall. (sir's note: F1 Score is actually Harmonic Mean)  
# * support—The number of samples with a given expected value. For example, 50 samples were labeled as 4s, and 38 samples were labeled as 5s.  
#   
# For details on the averages displayed at the bottom of the report, see:
# >http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html  
#   
# #### Predicitng from hand-drawn image:  

# In[32]:


from PIL import Image
import numpy as np

# Load the hand-drawn image
image_path = "20250319_104940.jpg"  # replace with the path to your image
image = Image.open(image_path)

# Convert the image to grayscale (if it is in color)
image = image.convert('L')

# Resize the image to match the size of the training images (e.g., 28x28)
image = image.resize((8, 8))

image.show()

# Convert the image to a numpy array and flatten it to a 1D array
image_array = np.array(image).flatten()

# Normalize the image (assuming pixel values were scaled in training)
# image_array = image_array / 255.0  # scaling to [0, 1] if needed


# In[33]:


# Predict the digit using the trained KNN model
predicted_label = knn.predict([image_array])  # We pass the image as a 2D array with shape (1, 784)
print(f"Predicted label: {predicted_label[0]}")


# In[34]:


print(image_array.shape) 


# #### Visualizing the Confusion Matrix  
# A **heat map** displays values as colors, often with values of higher magnitude displayed as more intense colors. Seaborn’s graphing functions work with two-dimensional data. When using a pandas DataFrame as the data source, Seaborn automatically labels its visualizations using the column names and row indices. Let’s convert the confusion matrix into a DataFrame, then graph it: 

# In[35]:


import pandas as pd


# In[36]:


confusion_df = pd.DataFrame(confusion, index=range(10), columns=range(10))


# In[37]:


import seaborn as sns


# In[38]:


axes = sns.heatmap(confusion_df, annot=True, cmap='nipy_spectral_r')


# The Seaborn function **heatmap** creates a heat map from the specified DataFrame. The
# keyword argument annot=True (short for “annotation”) displays a color bar to the right
# of the diagram, showing how the values correspond to the heat map’s colors. The
# cmap='nipy_spectral_r' keyword argument specifies which color map to use. We used
# the nipy_spectral_r color map with the colors shown in the heat map’s color bar. When
# you display a confusion matrix as a heat map, the principal diagonal and the incorrect predictions stand out nicely. 

# ### 15.3.2 K-Fold Cross-Validation  
# **K-fold cross-validation** enables you to use all of your data for both training and testing, to get a better sense of how well your model will make predictions for new data by repeatedly training and testing the model with different portions of the dataset. K-fold cross-validation splits the dataset into k equal-size folds (this k is unrelated to k in the k-nearest neighbors algorithm). You then repeatedly train your model with k – 1 folds and test the model with the remaining fold. For example, consider using k = 10 with folds numbered 1 through 10. With 10 folds, we’d do 10 successive training and testing cycles:
# * First, we’d train with folds 1–9, then test with fold 10.  
# * Next, we’d train with folds 1–8 and 10, then test with fold 9.  
# * Next, we’d train with folds 1–7 and 9–10, then test with fold 8.  
# This training and testing cycle continues until each fold has been used to test the model.  
#   
# #### KFold Class  
# Scikit-learn provides the **KFold class** and the **cross_val_score function** (both in the module sklearn.model_selection) to help you perform the training and testing cycles described above. Let’s perform k-fold cross-validation with the Digits dataset and the KNeighborsClassifier created earlier. First, create a KFold object:

# In[39]:


from sklearn.model_selection import KFold


# In[40]:


kfold = KFold(n_splits=10, random_state=11, shuffle=True)


# The keyword arguments are:   
# * n_splits=10, which specifies the number of folds.   
# * random_state=11, which seeds the random number generator for reproducibility.    
# * shuffle=True, which causes the KFold object to randomize the data by shuffling it before splitting it into folds. This is particularly important if the samples might be ordered or grouped. For example, the Iris dataset we’ll use later in this chapter has 150 samples of three Iris species—the first 50 are Iris setosa, the next 50 are Iris versicolor and the last 50 are Iris virginica. If we do not shuffle the samples, then the training data might contain none of a particular Iris species and the test data might be all of one species.  
#   
# #### Using the KFold Object with Function cross_val_score
# Next, use function cross_val_score to train and test your model:

# In[41]:


from sklearn.model_selection import cross_val_score


# In[42]:


scores = cross_val_score(estimator=knn, X=digits.data, y=digits.target, cv=kfold)


# The keyword arguments are:  
# * estimator=knn, which specifies the estimator you’d like to validate.  
# * X=digits.data, which specifies the samples to use for training and testing.  
# * y=digits.target, which specifies the target predictions for the samples.  
# * cv=kfold, which specifies the cross-validation generator that defines how to split the samples and targets for training and testing.  
# Function cross_val_score returns an array of accuracy scores—one for each fold. As you can see below, the model was quite accurate. Its lowest accuracy score was 0.97777778 (97.78%) and in one case it was 100% accurate in predicting an entire fold:

# In[43]:


scores


# Once you have the accuracy scores, you can get an overall sense of the model’s accuracy by calculating the mean accuracy score and the standard deviation among the 10 accuracy scores (or whatever number of folds you choose): 

# In[44]:


print(f'Mean accuracy: {scores.mean():.2%}')


# In[45]:


print(f'Accuracy standard deviation: {scores.std():.2%}')


# On average, the model was 98.72% accurate—even better than the 97.78% we achieved
# when we trained the model with 75% of the data and tested the model with 25% earlier.

# ## Questions: 
#   
# 1. **Name of Dataset?** - digits  
# 
# 2. **Name of input Data?** - digits.data()  
# 
# 3. **Name of output data?** - digits.target()  
# 
# 4. **Why 64 Columns?** - 8 x 8 pixels make 64 features  
# 
# 5. **Which function splits data?** - train_test_split()  
# 
# 6. **How to fix test Size during split?** - test_size()  
# 
# 7. **Name of model?** - knn  
# 
# 8. **Which line no. creates Model?** - knn = KNeighborsClassifier()  
# 
# 9. **Which line no. trains model?** - knn.fit(X=X_train, y=y_train)  
# 
# 10. **Which method predicts output?** - knn.predict(X=X_test)  
# 
# 11. **Which function finds accuracy and how?** - knn.score(X_test, y_test) X_test ru predicted bahariki y_test sangare compare houchi. ( X_test is iput data & y_test is actual output data ).     
# 
# 12. **Cell of CM Means?** - How many actual [row index] are predicted as [column index]  
# 
# 13. **Row of CM Means?** - row index represent actual data  
# 
# 14. **Column of CM Means?** - column index represent predicted data  
# 
# 15. **Why to use CV?** - More reliable  
# 
# 16. **Which function partitions data in CV?** - KFold(n_splits=10, random_state=11, shuffle=True) - the n_splits parameter is responsible for no. of partitions 
# 

# In[46]:


from textblob import TextBlob

blob = TextBlob("Python is a high-level language.")

print(blob.noun_phrases)


# ## 15.3.3 Running Multiple Models to Find the Best One  
# It’s difficult to know in advance which machine learning model(s) will perform best for a
# given dataset, especially when they hide the details of how they operate from their users.
# Even though the KNeighborsClassifier predicts digit images with a high degree of accuracy, it’s possible that other scikit-learn estimators are even more accurate. Scikit-learn
# provides many models with which you can quickly train and test your data. This encourages you to run multiple models to determine which is the best for a particular machine
# learning study.  
#   
# Let’s use the techniques from the preceding section to compare several classification
# estimators—KNeighborsClassifier, SVC and GaussianNB (there are more). Though we
# have not studied the SVC and GaussianNB estimators, scikit-learn nevertheless makes it easy
# for you to test-drive them by using their default settings.6 First, let’s import the other two
# estimators:

# In[47]:


from sklearn.svm import SVC


# In[48]:


from sklearn.naive_bayes import GaussianNB


# Next, let’s create the estimators. The following dictionary contains key–value pairs for
# the existing KNeighborsClassifier we created earlier, plus new SVC and GaussianNB estimators:

# In[49]:


estimators = {
    'KNeighborsClassifier': knn,
    'SVC': SVC(gamma='scale'),
    'GaussianNB': GaussianNB()}


# In[50]:


# using cross validation:
for estimator_name, estimator_object in estimators.items():
    kfold = KFold(n_splits=10, random_state=11, shuffle=True)
    scores = cross_val_score(estimator=estimator_object, X=digits.data, y=digits.target, cv=kfold)
    print(f'{estimator_name:>20}: ' + f'mean accuracy={scores.mean():.2%}; ' + f'standard deviation={scores.std():.2%}')
    
# using standard score:
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=11, test_size=0.3)

for estimator_name, estimator_object in estimators.items():
    estimator_object.fit(X_train, y_train)
    accuracy = estimator_object.score(X_test, y_test) 
    print(f'{estimator_name:>20}: accuracy={accuracy:.2%}')


# This loop iterates through items in the estimators dictionary and for each key-value pair performs the following tasks:
# - Unpacks the key into estimator_name and value into estimator_object.  
# - Creates a KFold object that shuffles the data and produces 10 folds. The keyword argument random_state is particularly important here because it ensures that each estimator works with identical folds, so we’re comparing “apples to apples.”  
# - Evaluates the current estimator_object using cross_val_score.  
# - Prints the estimator’s name, followed by the mean and standard deviation of the accuracy scores’ computed for each of the 10 folds.  
#   
# Based on the results, it appears that we can get slightly better accuracy from the SVC estimator—at least when using the estimator’s default settings. It’s possible that by tuning some of the estimators’ settings, we could get even better results. The KNeighborsClassifier and SVC estimators’ accuracies are nearly identical so we might want to perform hyperparameter tuning on each to determine the best.  
#   
# #### Scikit-Learn Estimator Diagram  
# The scikit-learn documentation provides a helpful diagram for choosing the right estimator, based on the kind and size of your data and the machine learning task you wish to perform:  
# > https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

# In[55]:


from IPython.display import SVG

# Display an SVG file
SVG(filename='ml_map.svg')


# # 15.4 Case Study: Time Series and Simple Linear Regression  
#   
# [ Restart Kernel Here ]

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd


# In[2]:


nyc = pd.read_csv('ave_hi_nyc_jan_1895-2018.csv')


# In[3]:


nyc.columns = ['Date', 'Temperature', 'Anomaly']


# In[4]:


nyc.Date = nyc.Date.floordiv(100)


# In[5]:


nyc.head(3)


# ### Splitting the Data for Training and Testing

# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(
     nyc.Date.values.reshape(-1, 1), nyc.Temperature.values, 
     random_state=11)


# In[8]:


X_train.shape


# In[9]:


X_test.shape


# ### Training the Model

# In[10]:


from sklearn.linear_model import LinearRegression


# In[11]:


linear_regression = LinearRegression()


# In[12]:


linear_regression.fit(X=X_train, y=y_train)


# In[13]:


linear_regression.coef_


# In[14]:


linear_regression.intercept_


# ### Testing the Model

# In[15]:


predicted = linear_regression.predict(X_test)


# In[16]:


expected = y_test


# In[17]:


for p, e in zip(predicted[::5], expected[::5]):
     print(f'predicted: {p:.2f}, expected: {e:.2f}')


# ### Predicting Future Temperatures and Estimating Past Temperatures

# In[18]:


predict = (lambda x: linear_regression.coef_ * x + 
                      linear_regression.intercept_)


# In[19]:


predict(2019)


# In[20]:


predict(1890)


# ### Visualizing the Dataset with the Regression Line

# In[21]:


import seaborn as sns


# In[28]:


axes = sns.scatterplot(data=nyc, x='Date', y='Temperature',
     hue='Temperature', palette='winter', legend=False)

axes.set_ylim(10, 70)

import numpy as np

x = np.array([min(nyc.Date.values), max(nyc.Date.values)])

y = predict(x)

import matplotlib.pyplot as plt 

line = plt.plot(x, y)


# ### Overfitting/Underfitting  
#   
# When creating a model, a key goal is to ensure that it is capable of making accurate predictions for data it has not yet seen. Two common problems that prevent accurate predictions are overfitting and underfitting:  
# - Underfitting occurs when a model is too simple to make predictions, based on its training data. For example, you may use a linear model, such as simple linear regression, when in fact, the problem really requires a non-linear model. For example, temperatures vary significantly throughout the four seasons. If you’re trying to create a general model that can predict temperatures year-round, a simple linear regression model will underfit the data.
# - Overfitting occurs when your model is too complex. The most extreme case, would be a model that memorizes its training data. That may be acceptable if your new data looks exactly like your training data, but ordinarily that’s not the case. When you make predictions with an overfit model, new data that matches the training data will produce perfect predictions, but the model will not know what to do with data it has never seen.  
#   
# For additional information on underfitting and overfitting, see
# - > https://en.wikipedia.org/wiki/Overfitting  
# - > https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/
# 
# **What? Why? & How? asipare exam re.**
# 

# In[31]:


from IPython.display import SVG
SVG(filename='1718273106637.svg')


# In[35]:


import matplotlib.pyplot as plt

data = [
    ["Aspect", "Underfitting", "Overfitting"],
    ["Model Complexity", "Too simple", "Too complex"],
    ["Training Performance", "Poor", "Excellent"],
    ["Training Error", "More", "Less"],
    ["Test Performance", "Poor", "Poor"],
    ["Test Error", "More", "More"],
    ["Common Causes", "Not enough features, too much regularization", "Too many features, memorizing noise"],
    ["Solutions", "Use complex model, add features", "Regularization, feature selection, more data"]
]

fig, ax = plt.subplots(figsize=(7, 4))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=data, cellLoc='center', loc='center')

plt.show()


# ## 15.5 Case Study: Multiple Linear Regression with the California Housing Dataset  
#   
# The California Housing dataset7 bundled with scikit-learn has 20,640 samples, each with eight numerical features. We’ll perform a multiple linear regression that uses all eight numerical features to make more sophisticated housing price predictions than if we were to use only a single feature or a subset of the features. 
#   
# `[ Restart Kernel Here ]`
# 
# ### Loading the Dataset   

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import fetch_california_housing


# In[2]:


california = fetch_california_housing()


# ### Displaying the Dataset’s Description

# In[3]:


print(california.DESCR)


# In[4]:


california.data.shape


# In[5]:


california.target.shape


# The Bunch’s **feature_names attribute** contains the names that correspond to each column
# in the data array:

# In[6]:


california.feature_names


# ### Exploring the Data with Pandas

# In[8]:


import pandas as pd
pd.set_option('display.precision', 4)
pd.set_option('display.max_columns', 9)
pd.set_option('display.width', None)
california_df = pd.DataFrame(california.data, 
                              columns=california.feature_names)
 
california_df['MedHouseValue'] = pd.Series(california.target)


# In[9]:


california_df.head()


# In[10]:


california_df.describe()


# In[11]:


#15.5.3 Visualizing the Features
sample_df = california_df.sample(frac=0.1, random_state=17)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)
sns.set_style('whitegrid')                                    
for feature in california.feature_names:
     plt.figure(figsize=(16, 9))
     sns.scatterplot(data=sample_df, x=feature, 
                     y='MedHouseValue', hue='MedHouseValue', 
                     palette='cool', legend=False)


# In[12]:


#5.5.4 Splitting the Data for Training and Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     california.data, california.target, random_state=11)
X_train.shape
X_test.shape


# In[14]:


#15.5.5 Training the Model
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X=X_train, y=y_train)
for i, name in enumerate(california.feature_names):
     print(f'{name:>10}: {linear_regression.coef_[i]}')
linear_regression.intercept_


# In[15]:


#5.5.6 Testing the Model
predicted = linear_regression.predict(X_test)
expected = y_test
predicted[:5]
expected[:5]


# In[16]:


#15.5.7 Visualizing the Expected vs. Predicted Prices
df = pd.DataFrame()
df['Expected'] = pd.Series(expected)
df['Predicted'] = pd.Series(predicted)
figure = plt.figure(figsize=(9, 9))

axes = sns.scatterplot(data=df, x='Expected', y='Predicted', 
     hue='Predicted', palette='cool', legend=False)

start = min(expected.min(), predicted.min())

end = max(expected.max(), predicted.max())

axes.set_xlim(start, end)

axes.set_ylim(start, end)

line = plt.plot([start, end], [start, end], 'k--')


# In[ ]:


#Binary Classification
# EMAI RA TP, FP, TN, FN, Accuracy=TP+TN/ALL, Precission=TP/TP+FP, Recall=TP/TP+FN


# # 15.7 Case Study: Unsupervised Machine Learning, Part 2—k-Means Clustering

# In[1]:


from sklearn.datasets import load_iris


# In[2]:


iris = load_iris()


# In[3]:


print(iris.DESCR)


# In[4]:


iris.data.shape


# In[5]:


iris.target.shape


# In[6]:


iris.target_names


# In[7]:


iris.feature_names


# In[10]:


import pandas as pd

pd.set_option('display.max_columns', 5)
pd.set_option('display.width', None)

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = [iris.target_names[i] for i in iris.target]
iris_df.head()


# In[12]:


pd.set_option('display.precision', 2)


# In[13]:


iris_df.describe()


# In[14]:


iris_df['species'].describe()


# In[15]:


# 15.7.3 Visualizing the Dataset with a Seaborn pairplot

import seaborn as sns

sns.set(font_scale=1.1)
sns.set_style('whitegrid')
grid = sns.pairplot(data=iris_df, vars=iris_df.columns[0:4],hue='species')


# In[16]:


# Displaying the pairplot in One Color

grid = sns.pairplot(data=iris_df, vars=iris_df.columns[0:4])


# In[17]:


#15.7.4 Using a KMeans Estimator
##Creating the Estimator
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=11)

##Fitting the Model

kmeans.fit(iris.data)


# In[18]:


#Comparing the Computer Cluster Labels to the Iris Dataset’s Target Values

print(kmeans.labels_[0:50])


# In[19]:


print(kmeans.labels_[50:100])


# In[20]:


print(kmeans.labels_[100:150])


# In[21]:


kmeans2 = KMeans(n_clusters=2, random_state=11)
kmeans2.fit(iris.data)
print(kmeans2.labels_[0:50])
print(kmeans2.labels_[50:150])


# In[22]:


#15.7.5 Dimensionality Reduction with Principal Component Analysis
##Creating the PCA Object

from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=11)
   
##Transforming the Iris Dataset’s Features into Two Dimensions
pca.fit(iris.data)


# In[23]:


iris_pca = pca.transform(iris.data)
iris_pca.shape


# In[24]:


##Visualizing the Reduced Data

iris_pca_df = pd.DataFrame(iris_pca,columns=['Component1', 'Component2'])
iris_pca_df['species'] = iris_df.species
axes = sns.scatterplot(data=iris_pca_df, x='Component1', y='Component2', hue='species', legend='brief') 

iris_centers = pca.transform(kmeans.cluster_centers_)

import matplotlib.pyplot as plt

dots = plt.scatter(iris_centers[:,0], iris_centers[:,1], s=100, c='k')


# In[26]:


#15.7.6 Choosing the Best Clustering Estimator

from sklearn.cluster import DBSCAN, MeanShift, SpectralClustering, AgglomerativeClustering

estimators = {
    'KMeans': kmeans,
    'DBSCAN': DBSCAN(),
    'MeanShift': MeanShift(),
    'SpectralClustering': SpectralClustering(n_clusters=3),
    'AgglomerativeClustering': 
        AgglomerativeClustering(n_clusters=3)
}

import numpy as np

for name, estimator in estimators.items():
    estimator.fit(iris.data)
    print(f'\n{name}:')
    for i in range(0, 101, 50):
        labels, counts = np.unique(estimator.labels_[i:i+50], return_counts=True)
        print(f'{i}-{i+50}:')
        for label, count in zip(labels, counts):
            print(f'   label={label}, count={count}')


# # Extra : sk.learn - Logistic Regression

# In[2]:


get_ipython().system('pip install --upgrade numpy scikit-learn')


# In[3]:


get_ipython().system('pip install "numpy<2.0"')


# In[4]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load data
california = fetch_california_housing()

# Convert regression target to binary classification (e.g., high vs low value)
median_price = california.target.mean()
y_class = (california.target >= median_price).astype(int)  # 1 if high price, else 0

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    california.data, y_class, random_state=11)


# In[6]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_train_scaled, y_train)


# In[7]:


predicted = logistic_regression.predict(X_test)
expected = y_test

# Show first 5 predictions and expected values
print("Predicted:", predicted[:5])
print("Expected:", expected[:5])


# In[8]:


print("\nClassification Report:\n", classification_report(expected, predicted))
print("Confusion Matrix:\n", confusion_matrix(expected, predicted))


# In[9]:


df = pd.DataFrame()
df['Expected'] = expected
df['Predicted'] = predicted

plt.figure(figsize=(7, 7))
sns.scatterplot(data=df, x='Expected', y='Predicted', hue='Predicted', palette='cool', legend=False)
plt.title("Expected vs Predicted Classes (Logistic Regression)")
plt.xlabel("Expected Class")
plt.ylabel("Predicted Class")
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.grid(True)
plt.show()


# In[10]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Predict
y_pred = log_reg.predict(X_test_scaled)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


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

