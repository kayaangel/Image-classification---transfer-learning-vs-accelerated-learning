# -*- coding: utf-8 -*-
'''

The functions and classes defined in this module will be called by a marker script. 
You should complete the functions and classes according to their specified interfaces.

No partial marks will be awarded for functions that do not meet the specifications
of the interfaces.

Last modified 2024-05-07 by Anthony Vanderkop.
Hopefully without introducing new bugs.
'''

### LIBRARY IMPORTS HERE ###
import os
import cv2
import keras
import random
import numpy as np
import tensorflow as tf
from imutils import paths
import matplotlib.pyplot as plt
from keras.src.utils import img_to_array

image_dims = (224, 224, 3)


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''

    return [(11921048, 'Isabell Sophie', 'Hans'), (11220902, 'Kayathri', 'Arumugam'),
            (11477296, 'Nasya Sze Yuen', 'Liew')]


def load_model():
    """
    Load a pre-trained MobileNetV2 model, add a new output layer, and return the model.

    Returns:
        tf.keras.Model: A TensorFlow Keras Model with the loaded MobileNetV2 base model and a new output layer.
    """

    num_classes = 5

    # Task 2: Download a pre-trained MobileNetV2 model
    base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(224, 224, 3))

    # Task 3: Replace output layer to match number of classes
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    base_model.trainable = False 

    return model


def accelerated_learning_model():
    """
    Load a pre-trained MobileNetV2 model, add data augmentation, and return the model.

    Returns:
        tf.keras.Model: A TensorFlow Keras Model with data augmentation and feature extraction layers.
    """
    
    # Load the pre-trained MobileNetV2 model without the top layer (include_top=False)
    base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(224, 224, 3))
    

    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    feature_extraction_model = tf.keras.Model(inputs, x)
    
    return feature_extraction_model


def freeze_layers(model):
    """
    Freeze all of the layers in the model except the last one.

    Args:
        tf.keras.Model: A TensorFlow Keras Model

    Returns:
        tf.keras.Model: A TensorFlow Keras Model with the base model freezed

    """

    # Freeze all layers except the last one
    for layer in model.layers[:-1]:
        layer.trainable = False

    return model


def load_data(path):
    """
    Load the data from the specified path and return the data and labels as numpy arrays.

    Args:
        path (str): The path to the data directory.

    Returns:
        tuple: A numpy array containing the data and the labels.
    """

    imagePaths = sorted(list(paths.list_images(path)))
    class_to_int = map_classes_to_int(path)
    random.seed(42)
    random.shuffle(imagePaths)
    combined = []

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)[..., ::-1]  # Convert BGR to RGB
        image = cv2.resize(image, (image_dims[1], image_dims[0]))  # Resize image
        image = img_to_array(image)  # Convert image to array
        label = imagePath.split(os.path.sep)[-2]  # Get label from path
        int_label = class_to_int[label]  # Convert label to integer
        combined.append(np.append(image, int_label))

    combined = np.array(combined)

    return combined


def split_combined_numpy(combined):
    """
    Split the combined array into features (data) and labels, matching the format of data and labels returned by the original function.

    Args:
        combined (numpy.ndarray): The combined array containing both features and labels.

    Returns:
        tuple: A tuple containing features (data) and labels as separate numpy arrays.
    """

    data = []
    labels = []

    for row in combined:
        image = row[:-1].reshape(image_dims)  # Reshape image data to original shape
        int_label = int(row[-1])  # Convert label to integer
        labels.append(int_label)
        data.append(image)

    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    return data, labels


def map_classes_to_int(path):
    """
    Map class names to numbers based on the folder structure in the given path.

    Args:
        path (str): The path to the directory containing class folders.

    Returns:
        dict: A dictionary mapping class names to numbers.
    """

    class_to_int = {}
    classes = sorted(os.listdir(path))
    for i, class_name in enumerate(classes):
        class_to_int[class_name] = i
    return class_to_int


def split_data(X, Y, train_fraction, randomize=False, eval_set=True):
    """
    Split the data into training and testing sets. If eval_set is True, also create
    an evaluation dataset. There should be two outputs if eval_set there should
    be three outputs (train, test, eval), otherwise two outputs (train, test).
    
    Args:
        X (numpy.ndarray): Input features.
        Y (numpy.ndarray): Corresponding labels.
        train_fraction (float): Fraction of data to use for training.
        randomize (bool, optional): Whether to randomly shuffle the data. Defaults to False.
        eval_set (bool, optional): Whether to create an evaluation dataset. Defaults to True.

    Returns:
        tuple: If eval_set is True, returns train, test, eval
               If eval_set is False, returns train, test.
    """

    num_samples = len(X)
    train_samples = int(num_samples * train_fraction)
    test_samples = int(num_samples * ((1 - train_fraction) / 2))

    if randomize:
        indices = np.random.permutation(num_samples)
        X = X[indices]
        Y = Y[indices]

    train_X = X[:train_samples]
    train_Y = Y[:train_samples]
    test_X = X[train_samples:train_samples + test_samples]
    test_Y = Y[train_samples:train_samples + test_samples]

    train = (train_X, train_Y)
    test = (test_X, test_Y)

    if eval_set:
        eval_X = X[train_samples + test_samples:]
        eval_Y = Y[train_samples + test_samples:]
        eval = (eval_X, eval_Y)
        return train, test, eval
    else:
        return train, test


def confusion_matrix(predictions, ground_truth, plot=False, all_classes=None):
    '''
    Given a set of classifier predictions and the ground truth, calculate and
    return the confusion matrix of the classifier's performance.

    Args:
        predictions: np.ndarray of length n where n is the number of data
                       points in the dataset being classified and each value
                       is the class predicted by the classifier
        ground_truth: np.ndarray of length n where each value is the correct
                        value of the class predicted by the classifier
        plot: boolean. If true, create a plot of the confusion matrix with
                either matplotlib or with sklearn.
        classes: a set of all unique classes that are expected in the dataset.
                   If None is provided we assume all relevant classes are in 
                   the ground_truth instead.
    Returns:
        cm: type np.ndarray of shape (c,c) where c is the number of unique
              classes in the ground_truth
              
              Each row corresponds to a unique class in the ground truth and
              each column to a prediction of a unique class by a classifier
    '''

    # Ensure predictions and ground_truth are numpy arrays
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Convert predictions to class labels if they are probabilities
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)

    # Get unique classes from ground_truth if not provided
    if all_classes is None:
        all_classes = np.unique(ground_truth)

    # Initialize confusion matrix
    cm = np.zeros((len(all_classes), len(all_classes)), dtype=int)

    # Populate confusion matrix
    for pred, true in zip(predictions, ground_truth):
        cm[true, pred] += 1

    if plot:
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(all_classes))
        plt.xticks(tick_marks, all_classes)
        plt.yticks(tick_marks, all_classes)
        plt.savefig('confusion_matrix.png')
        plt.show()

    return cm


def precision(predictions, ground_truth):
    '''
    Calculates the classifier's precision
    
    Args:
        predictions: np.ndarray of length n where n is the number of data
                       points in the dataset being classified and each value
                       is the class predicted by the classifier
        ground_truth: np.ndarray of length n where each value is the correct
                        value of the class predicted by the classifier
    Returns:
        precision: type np.ndarray of length c,
                     values are the precision for each class
    '''

    # Convert predictions and ground truth to numpy arrays
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # If predictions are one-hot encoded, convert to class indices
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)

    # Determine the number of classes
    num_classes = len(np.unique(ground_truth))
    
    # Initialize precision array for each class
    precision = np.zeros(num_classes)

    # Calculate precision for each class
    for cls in range(num_classes):
        true_positives = np.sum((predictions == cls) & (ground_truth == cls))
        all_positives = np.sum(predictions == cls)
        precision[cls] = true_positives / max(all_positives, 1)

    return precision


def recall(predictions, ground_truth):
    '''
    Calculates the classifier's recall
    
    Args:
        predictions: np.ndarray of length n where n is the number of data
                       points in the dataset being classified and each value
                       is the class predicted by the classifier
        ground_truth: np.ndarray of length n where each value is the correct
                        value of the class predicted by the classifier
    Returns:
        recall: type np.ndarray of length c,
                     values are the recall for each class
    '''

    # Convert predictions and ground truth to numpy arrays
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # If predictions are one-hot encoded, convert to class indices
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)

    # Determine the number of classes
    num_classes = len(np.unique(ground_truth))
    
    # Initialize recall array for each class
    recall = np.zeros(num_classes)

    # Calculate recall for each class
    for cls in range(num_classes):
        true_positives = np.sum((predictions == cls) & (ground_truth == cls))
        all_actual = np.sum(ground_truth == cls)
        recall[cls] = true_positives / max(all_actual, 1)

    return recall


def f1(predictions, ground_truth):
    '''
    Calculates the classifier's f1 score
    Args:
        predictions: np.ndarray of length n where n is the number of data
                       points in the dataset being classified and each value
                       is the class predicted by the classifier
        ground_truth: np.ndarray of length n where each value is the correct
                        value of the class predicted by the classifier
    Returns:
        f1: type nd.ndarry of length c where c is the number of classes
    '''

    # Calculate precision for each class
    prec = precision(predictions, ground_truth)
    
    # Calculate recall for each class
    rec = recall(predictions, ground_truth)
    
    # Calculate F1 score for each class, adding a small constant to avoid division by zero
    f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
    
    return f1



def k_fold_validation(features, ground_truth, classifier, k=2):
    '''
    Perform k-fold cross validation on the given classifier.
    Args:
        features: np.ndarray of features in the dataset
        ground_truth: np.ndarray of class values associated with the features
        fit_func: f
        classifier: class object with both fit() and predict() methods which
                    can be applied to subsets of the features and ground_truth inputs.
        predict_func: function, calling predict_func(features) should return
                    a numpy array of class predictions which can in turn be input to the
                    functions in this script to calculate performance metrics.
        k: int, number of sub-sets to partition the data into. default is k=2
    Returns:
        avg_metrics: np.ndarray of shape (3, c) where c is the number of classes.
                     The first row is the average precision for each class over the k
                     validation steps. Second row is recall and third row is f1 score.
        sigma_metrics: np.ndarray, each value is the standard deviation of
                     the performance metrics [precision, recall, f1_score]
    '''

    num_classes = len(np.unique(ground_truth))
    avg_prec = np.zeros((k, num_classes))
    avg_rec = np.zeros((k, num_classes))
    avg_f1_score = np.zeros((k, num_classes))

    # Calculate fold size and shuffle the data
    num_samples = len(features)
    fold_size = num_samples // k
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    shuffled_features = features[indices]
    shuffled_ground_truth = ground_truth[indices]

    # Perform k-fold cross validation
    for partition_no in range(k):
        start = partition_no * fold_size
        end = (partition_no + 1) * fold_size if partition_no != k - 1 else num_samples

        test_features = shuffled_features[start:end]
        test_classes = shuffled_ground_truth[start:end]

        train_features = np.concatenate((shuffled_features[:start], shuffled_features[end:]), axis=0)
        train_classes = np.concatenate((shuffled_ground_truth[:start], shuffled_ground_truth[end:]), axis=0)

        # Fit model to training data & perform predictions on the test set
        classifier.fit(train_features, train_classes)
        predictions = classifier.predict(test_features)

        # Calculate performance metrics
        prec = precision(predictions, test_classes)
        rec = recall(predictions, test_classes)
        f1_score = f1(predictions, test_classes)

        # Accumulate metrics across all folds
        avg_prec[partition_no] = prec
        avg_rec[partition_no] = rec
        avg_f1_score[partition_no] = f1_score

    # Calculate average metrics across all folds
    avg_prec_mean = np.mean(avg_prec, axis=0)
    avg_rec_mean = np.mean(avg_rec, axis=0)
    avg_f1_score_mean = np.mean(avg_f1_score, axis=0)

    # Calculate standard deviation of performance metrics
    sigma_prec = np.std(avg_prec, axis=0)
    sigma_rec = np.std(avg_rec, axis=0)
    sigma_f1_score = np.std(avg_f1_score, axis=0)

    avg_metrics = np.vstack((avg_prec_mean, avg_rec_mean, avg_f1_score_mean))
    sigma_metrics = np.array([sigma_prec, sigma_rec, sigma_f1_score])

    return avg_metrics, sigma_metrics


##################### MAIN ASSIGNMENT CODE FROM HERE ######################

def transfer_learning(train_set, eval_set, test_set, model, parameters):
    '''
    Performs standard transfer learning.

    Args:
        train_set: list or tuple of the training images and labels in the
            form (images, labels) for training the classifier
        eval_set: list or tuple of the images and labels used in evaluating
            the model during training, in the form (images, labels)
        test_set: list or tuple of the training images and labels in the
            form (images, labels) for testing the classifier after training
        model: an instance of tf.keras.applications.MobileNetV2
        parameters: list or tuple of parameters to use during training:
            (learning_rate, momentum, nesterov)

    Returns:
        model : an instance of tf.keras.applications.MobileNetV2
        metrics : list of class-wise recall, precision, and f1 scores of the
            model on the test_set (list of np.ndarray)

    '''

    learning_rate, momentum, nesterov = parameters
    metrics = ['accuracy']

    # set the optimizer
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov
    )

    # compile the model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=metrics)

    # train the model
    history = model.fit(
        x=train_set[0],
        y=train_set[1],
        validation_data=eval_set,
        epochs=30
    )

    # Task 7: Plot the training and validation errors and accuracies of standard transfer learning
    plot_learning_curves(history, parameters)

    return model, metrics


def accelerated_learning(train_set, eval_set, test_set, model, parameters):
    '''
    Performs accelerated transfer learning.

    Args:
        train_set: list or tuple of the training images and labels in the
            form (images, labels) for training the classifier
        eval_set: list or tuple of the images and labels used in evaluating
            the model during training, in the form (images, labels)
        test_set: list or tuple of the training images and labels in the
            form (images, labels) for testing the classifier after training
        model: an instance of tf.keras.applications.MobileNetV2
        parameters: list or tuple of parameters to use during training:
            (learning_rate, momentum, nesterov)
    '''
    learning_rate, momentum, nesterov = parameters
    metrics = ['accuracy']
    
    # Extract features from train and eval set using pre-trained model
    train_features = model.predict(train_set[0])
    eval_features = model.predict(eval_set[0])
    
    input_shape = (train_features.shape[1:])
        
    # Define custom model layers
    custom_classification_model = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(input_shape=input_shape),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    # set the optimizer
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov
    )

    # Compile the classification model
    custom_classification_model.compile(optimizer=optimizer,
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                         metrics=metrics)
    
    # Train the classification model on the precomputed features
    history = custom_classification_model.fit(x=train_features, y=train_set[1],
                               epochs=10,
                               validation_data=(eval_features, eval_set[1]))
    
    # Plot the training and validation errors and accuracies of accelerated transfer learning
    plot_learning_curves(history, parameters)


def plot_learning_curves(history, parameters):
    """
    A function that plots the learning curves of a model based on the training history.

    Args:
        history: the training history of the model
        parameters: the parameters used during training
    """

    learning_rate, momentum, nesterov = parameters

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy (learning rate: ' + str(learning_rate) + ' momentum: ' + str(momentum) + ')')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy_lr' + str(learning_rate) + '_mo' + str(momentum) + '.png')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss (learning rate: ' + str(learning_rate) + ' momentum: ' + str(momentum) + ')')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig('loss_lr' + str(learning_rate) + '_mo' + str(momentum) + '.png')
    plt.show()


def task_1():
    """
    Task 1:
    This function returns the path to the small_flower_dataset.

    Returns:
        str: The path to the small_flower_dataset.
    """

    path = 'small_flower_dataset'
    return path


def task_2_3():
    """
    Task 2:
    Download a pre-trained MobileNetV2 model and
    Task 3:
    Replace the last layer of the model

    Returns:
        tf.keras.Model: A TensorFlow Keras Model with the loaded MobileNetV2 base model and a new output layer.
    """

    model = load_model()

    return model


def task_4(path):
    """
    Task 4:
    Prepare the data for standard transfer learning.

    Args:
        path (str): The path to the dataset.

    Returns:
        tuple: A tuple containing the train, test, and evaluation data and labels.
            - train (np.ndarray): The training data.
            - test (np.ndarray): The test data.
            - eval (np.ndarray): The evaluation data.
    """

    data, labels = split_combined_numpy(load_data(path))
    train, test, eval = split_data(data, labels, 0.8, True, True)
    return train, test, eval


def task_5_7(train, test, eval, model):
    """
    Task 5: Compile and train model with SGD optimizer and
    Task 7: Plot the curves.

    Args:
        train (tuple): A tuple containing the training images and labels in the form (images, labels).
        test (tuple): A tuple containing the test images and labels in the form (images, labels).
        eval (tuple): A tuple containing the evaluation images and labels in the form (images, labels).
        model (tf.keras.Model): An instance of tf.keras.applications.MobileNetV2.

    Returns:
        tuple: A tuple containing the trained model and the metrics of the model on the test set.
            - model (tf.keras.Model): The trained model.
            - metrics (list): A list of class-wise recall, precision, and f1 scores of the model on the test set.
    """

    model, metrics = transfer_learning(train, test, eval, model, (0.01, 0.0, False))
    return model, metrics


def task_8(train, test, eval):
    """
    Task 8: Experiment with 3 different orders of magnitude for the learning rate.

    This function performs transfer learning with three different learning rates: 0.001, 0.1, and 1. It uses the
    `transfer_learning` function to train the model with each learning rate.
    It returns the model with the best results. The discussion on the best results can be found in the report.

    Args:
        train (tuple): A tuple containing the training images and labels in the form (images, labels).
        test (tuple): A tuple containing the test images and labels in the form (images, labels).
        eval (tuple): A tuple containing the evaluation images and labels in the form (images, labels).

    Returns:
        model_small_lr (tf.keras.Model): The model trained with the smallest learning rate which turned out to be the best.
    """

    model = load_model()
    model_small_lr, metrics_small_lr = transfer_learning(train, test, eval, model,
                                                         (0.001, 0.0, False))  # Deemed best learning rate
    model = load_model()
    model_medium_lr, metrics_medium_lr = transfer_learning(train, test, eval, model, (0.1, 0.0, False))
    model = load_model()
    model_large_lr, metrics_large_lr = transfer_learning(train, test, eval, model, (1.0, 0.0, False))

    return model_small_lr


def task_9(model, test):
    """
    Task 9:
    Calculate the confusion matrix for a trained classifier and print it.

    Args:
        model (tf.keras.Model): The trained classifier model.
        test (tuple): A tuple containing the test images and labels.

    Returns:
        tuple: A tuple containing the predicted classes and the true labels.
    """

    # Assuming you have trained your classifier and obtained predictions
    test_predictions = model.predict(test[0])
    test_predictions_classes = np.argmax(test_predictions, axis=1)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(test_predictions_classes, test[1], plot=True)

    return test_predictions_classes, test[1]


def task_10(test_predictions_classes, ground_truth):
    """
    Task 10:
    Calculate and print the precision, recall, and F1 scores for a classifier's performance on a test set.

    Args:
        test_predictions_classes (np.ndarray): An array of predicted classes for the test set.
        ground_truth (np.ndarray): An array of the correct classes for the test set.
    """

    precision_scores = precision(test_predictions_classes, test[1])
    recall_scores = recall(test_predictions_classes, test[1])
    f1_scores = f1(test_predictions_classes, test[1])

    print("Precision Scores: \n" + str(precision_scores))
    print("Recall Scores: \n" + str(recall_scores))
    print("F1 Scores: \n" + str(f1_scores))


def task_11(path, model):
    """
    Task 11:
    Perform k-fold validation for different values of k and print the average metrics and sigma metrics.

    Args:
        path (str): The path to the dataset.
        model (tf.keras.Model): The model to be trained.
    """

    data, labels = split_combined_numpy(load_data(path))
    k = 3
    avg_metrics, sigma_metrics = k_fold_validation(data, labels, model, k)
    print("Average Metrics (k=3): \n" + str(avg_metrics))
    print("Sigma Metrics (k=3): \n" + str(sigma_metrics))

    # Repeat for two different values of k
    k_values = [5, 10]
    for k in k_values:
        avg_metrics, sigma_metrics = k_fold_validation(data, labels, model, k)
        print(f"Average Metrics (k={k}): \n" + str(avg_metrics))
        print(f"Sigma Metrics (k={k}): \n" + str(sigma_metrics))


def task_12(train, test, eval):
    """
    Perform transfer learning with different momentum values for the given training, test, and evaluation data using the provided model and the best learning rate.

    Args:
        train (tuple): A tuple containing the training images and labels in the form (images, labels).
        test (tuple): A tuple containing the test images and labels in the form (images, labels).
        eval (tuple): A tuple containing the evaluation images and labels in the form (images, labels).
    """

    model = load_model()
    model_small_mo, metrics_small_mo = transfer_learning(train, test, eval, model, (0.001, 0.1, False))
    model = load_model()
    model_medium_mo, metrics_medium_mo = transfer_learning(train, test, eval, model, (0.001, 0.5, False))
    model = load_model()
    model_large_mo, metrics_large_mo = transfer_learning(train, test, eval, model, (0.001, 0.9, False))


def task_14(path):
    """
    Task 14:
    Perform accelerated learning on the given training, test, and evaluation data using the provided model.

    Args:
        path (str): The file path to the combined dataset.
    """
    data, labels = split_combined_numpy(load_data(path))
        
    # Split the data
    train, test, eval = split_data(data, labels, 0.8, True, True)
    
    # Load model
    model = accelerated_learning_model()
    
    # Accelerated learning model without momentum
    accelerated_learning(train, test, eval, model, (0.005, 0.0, False))
    
    # Accelerated learning model with non-zero momentum of 0.1
    accelerated_learning(train, test, eval, model, (0.005, 0.1, False))
    
    # Accelerated learning model with non-zero momentum of 0.5
    accelerated_learning(train, test, eval, model, (0.005, 0.5, False))
    
    # Accelerated learning model with non-zero momentum of 0.9
    accelerated_learning(train, test, eval, model, (0.005, 0.9, False))


if __name__ == "__main__":
    # Task 1: Use the small_flower_datatset
    path = task_1()

    # Task 2 and 3: Download a pre-trained MobileNetV2 model and replace last layer
    model = task_2_3()

    # Task 4: Prepare the data for standard transfer learning
    train, test, eval = task_4(path)

    # Task 5 and 7: Compile and train model with SGD optimizer and plot the curves
    std_model, std_metrics = task_5_7(train, test, eval, model)

    # Task 8: Experiment with 3 different orders of magnitude for the learning rate.
    best_model = task_8(train, test, eval)

    # Task 9: Run the classifier and plot the confusion matrix
    test_predictions_classes, ground_truth = task_9(best_model, test)

    # Task 10: Calculate and print the precision, recall, and f1 scores
    task_10(test_predictions_classes, ground_truth)

    # Task 11: Perform k-fold validation
    task_11(path, model)

    # Task 12: Add non-zero momentum
    task_12(train, test, eval)

    # Task 14: Perform accelerated learning
    task_14(path)

#########################  CODE GRAVEYARD  #############################
