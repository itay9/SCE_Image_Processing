
Dependencies:
opencv-contrib-python==4.5.4.58
scikit-image==0.18.3
scikit-learn==1.0.2
tabulate==0.8.9

This script classifies hebrew hand write images by gender.
Using HHD_gender Dataset*, the script takes 3 arguments:
train set, validation set and test consists of male and female hand written images.
Extracting LBP features using skimage library, and then training the model to select the best SVM parameters,
such as kernel type, and hyperparameters such as gamma and c.
When finishing training, the model run it on validation set for tuning and then runs it on test set.
The model outputs "results.txt" file which include the best parameters, model's accuracy and confusion matrix.

How to make it work?
run from your command line:
python classifier.py path_train path_val path_test

"path_train", "path_val" and "path_test" are the directories contain the image sets.

(use this pattern for each path: "[insert_path_here"])

*Rabaev I., Litvak M., Asulin S., Tabibi O.H. (2021) Automatic Gender Classification from
Handwritten Images: A Case Study. In: Computer Analysis of Images and Patterns. CAIP 2021.
Lecture Notes in Computer Science, vol 13053. Springer, Cham.
https://doi.org/10.1007/978-3-030-89131-2_30