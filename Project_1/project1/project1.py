from string import punctuation, digits

import numpy as np
import matplotlib.pyplot as plt

### Parameters
T_perceptron = 5

T_avperceptron = 5

T_avgpa = 5
L_avgpa = 10
###

### Part I

def hinge_loss(feature_matrix, labels, theta, theta_0):
    """
    Section 1.2
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """

    if feature_matrix.ndim == 1:
        feature_matrix = np.array([feature_matrix])

    dim_feature_vector = feature_matrix.shape
    num_rows = dim_feature_vector[0]
    sum = 0
    for j in range(0, num_rows): # j is each row (feature vector)
        x = feature_matrix[j,:]
        y = labels[j]

        if y*(np.dot(theta, x) + theta_0) <= 1: 
            sum += 1-y*(np.dot(theta,x)+theta_0)
        else:
            sum += 0
    return float(sum)/num_rows


def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Section 1.3
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    predicted_label = np.dot(current_theta, feature_vector) + current_theta_0
    if predicted_label > 0:
        predicted_label = 1
    else:
        predicted_label = -1 

    if label != predicted_label:
        current_theta += label*feature_vector
        current_theta_0 += label
    return (current_theta, current_theta_0)


def perceptron(feature_matrix, labels, T):
    """
    Section 1.4
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    
    dim_feature_vector = feature_matrix.shape
    num_rows = dim_feature_vector[0]
    num_col = dim_feature_vector[1]

    theta = np.zeros(num_col)
    theta_0 = 0
    for i in range(0, T): # i is iteration of the perceptron 
        for j in range(0, num_rows): # j is each row (feature vector)
            feature_vector = feature_matrix[j,:]
            label = labels[j]
            (current_theta, current_theta_0) = perceptron_single_step_update(feature_vector, label, theta, theta_0)
            (theta, theta_0) = (current_theta, current_theta_0)
    return (theta, theta_0)


def passive_aggressive_single_step_update(feature_vector, label, L, current_theta, current_theta_0):
    """
    Section 1.5
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the passive-aggressive algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the passive-aggressive
            algorithm parameters.
        current_theta - The current theta being used by the passive-aggressive
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            passive-aggressive algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
 
    if L == 0:
        eta = float(hinge_loss(feature_vector, label, current_theta, current_theta_0)/ (np.linalg.norm(feature_vector))**2)
    else:
        eta = min(float(hinge_loss(feature_vector, label, current_theta, current_theta_0)/ (np.linalg.norm(feature_vector))**2), 1.0 /L)

    predicted_label = np.dot(current_theta, feature_vector) + current_theta_0
    if predicted_label > 0:
        label = 1
    else:
        label = -1

    new_theta = current_theta
    new_theta_0 = current_theta_0

    new_theta = current_theta + eta*label*feature_vector
    new_theta_0 = current_theta_0 + eta*label
    return (new_theta, new_theta_0)



def average_perceptron(feature_matrix, labels, T):
    """
    Section 1.6
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    (num_rows, num_col) = feature_matrix.shape
    theta = np.zeros(num_col)
    theta_0 = 0
    theta_sum = np.zeros(num_col)
    theta_0_sum = 0
    for i in range(0, T): # i is iteration of the perceptron 
        for j in range(0, num_rows): # j is each row (feature vector)
            feature_vector = feature_matrix[j,:]
            label = labels[j]            
            (current_theta, current_theta_0) = perceptron_single_step_update(feature_vector, label, theta, theta_0)
            (theta, theta_0) = (current_theta, current_theta_0)
            theta_sum += current_theta
            theta_0_sum += current_theta_0
    return ((theta_sum / (1.0*T*num_rows)), (theta_0_sum / (1.0*T*num_rows)))

def average_passive_aggressive(feature_matrix, labels, T, L):
    """
    Section 1.6
    Runs the average passive-agressive algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the passive-agressive
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    (num_rows, num_col) = feature_matrix.shape
    theta = np.zeros(num_col)
    theta_0 = 0
    theta_sum = np.zeros(num_col)
    theta_0_sum = 0
    for i in range(0, T): # i is iteration of the perceptron 
        for j in range(0, num_rows): # j is each row (feature vector)
            feature_vector = feature_matrix[j, :]
            label = labels[j]
            (current_theta, current_theta_0) = passive_aggressive_single_step_update(feature_vector, label, L, theta, theta_0)
            (theta, theta_0) = (current_theta, current_theta_0)
            theta_sum += current_theta
            theta_0_sum += current_theta_0
    return ((theta_sum / (T*num_rows)), (theta_0_sum / (T*num_rows)))


### Part II

def classify(feature_matrix, theta, theta_0):
    """
    Section 2.8
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is the predicted
    classification of the kth row of the feature matrix using the given theta
    and theta_0.
    """
    (num_rows, num_col) = feature_matrix.shape
    label_vector = np.zeros([num_rows])

    for i in range(0, num_rows):
        if (np.dot(theta, feature_matrix[i, :]) + theta_0) > 0:
            label_vector[i] = 1
        else:
            label_vector[i] = -1 
    return label_vector     



def perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T):
    """
    Section 2.9
    Trains a linear classifier using the perceptron algorithm with a given T
    value. The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the perceptron algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    (train_theta, train_theta_0) = perceptron(train_feature_matrix, train_labels, T)
    

    train_guess_labels = classify(train_feature_matrix, train_theta, train_theta_0)
    val_guess_labels = classify(val_feature_matrix, train_theta, train_theta_0)

    train_labels_accuracy = accuracy(train_guess_labels, train_labels)
    val_labels_accuracy = accuracy(val_guess_labels, val_labels)
    

    return (train_labels_accuracy, val_labels_accuracy)


def average_perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T):
    """
    Section 2.9
    Trains a linear classifier using the average perceptron algorithm with
    a given T value. The classifier is trained on the train data. The
    classifier's accuracy on the train and validation data is then returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the average perceptron
            algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    (train_theta, train_theta_0) = average_perceptron(train_feature_matrix, train_labels, T)
    

    train_guess_labels = classify(train_feature_matrix, train_theta, train_theta_0)
    val_guess_labels = classify(val_feature_matrix, train_theta, train_theta_0)

    train_labels_accuracy = accuracy(train_guess_labels, train_labels)
    val_labels_accuracy = accuracy(val_guess_labels, val_labels)

    

    return (train_labels_accuracy, val_labels_accuracy)

def average_passive_aggressive_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T, L):
    """
    Section 2.9
    Trains a linear classifier using the average passive aggressive algorithm
    with given T and L values. The classifier is trained on the train data.
    The classifier's accuracy on the train and validation data is then
    returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the average passive
            aggressive algorithm.
        L - The value of L to use for training with the average passive
            aggressive algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    (train_theta, train_theta_0) = average_passive_aggressive(train_feature_matrix, train_labels, T, L)
    

    train_guess_labels = classify(train_feature_matrix, train_theta, train_theta_0)
    val_guess_labels = classify(val_feature_matrix, train_theta, train_theta_0)

    train_labels_accuracy = accuracy(train_guess_labels, train_labels)
    val_labels_accuracy = accuracy(val_guess_labels, val_labels)


    return (train_labels_accuracy, val_labels_accuracy)

def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()

def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
    """
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary

def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.
    """

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1
    return feature_matrix

def extract_additional_features(reviews):
    """
    Section 3.12
    Inputs a list of string reviews
    Returns a feature matrix of (n,m), where n is the number of reviews
    and m is the total number of additional features of your choice

    YOU MAY CHANGE THE PARAMETERS
    """
    new_column = np.zeros((len(reviews),1))
    for i in range(len(reviews)):
        new_column[i]=len(reviews[i])
    return new_column

def extract_final_features(reviews, dictionary):
    """
    Section 3.12
    Constructs a final feature matrix using the improved bag-of-words and/or additional features
    """
    bow_feature_matrix = extract_bow_feature_vectors(reviews,dictionary)
    additional_feature_matrix = extract_additional_features(reviews)
    return np.hstack((bow_feature_matrix, additional_feature_matrix))

def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
