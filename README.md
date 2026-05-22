# Standard transfer learning vs Accelerated learning

## Project Description
In this project, my team and I investigate the application of transfer learning using the MobileNetV2 architecture on the small flower dataset. The purpose of transfer learning is to address a fairly common problem -- not enough data to build a Convolutional Neural Network (CNN). Hence, transfer learning is a useful strategy when faced with this challenge by using an existing pre-trained CNN on a large dataset (like ImageNet in this example) as a fixed feature extractor for the classification task. Bypassing the need to train a full-scale model from scratch.

In **standard transfer learning**, 
1. Take the base layers of an existing pre-trained CNN
2. Freeze the weights to retain the useful features learned from a pre-trained model for feature extraction
3. Add new layers on top of the frozen layers that are trainable
4. Train the new layers on the new dataset

In **accelerated transfer learning**, it works in 2 stages.
1. Similar to the standard transfer learning, the images go through the frozen layers of the pre-trained network. However, this is done only once for a 4 x 4 x 512 output with core features extracted. These features are stored as a new dataset.
2. The custom layer on top is a separate, lightweight custom model layer that performs standard forward and backward propagation. The frozen base is never revisited.

Evaluation metrics applied
* The confusion matrix
* Precision
* Recall
* F1 Score
* K-fold Cross Validation

