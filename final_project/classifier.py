# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 00:38:30 2022

@author: Shmulik && Itay
"""
import os
from sys import argv
import numpy as np
import cv2 as cv
from skimage import feature
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from tabulate import tabulate

path_train= argv[1]
path_val= argv[2]
path_test= argv[3]


# import pathlib
def count_files(path):
    count = 0
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            count += 1
    return count

### loading train&test sets

def load_im(path,length):
    #pbar = tqdm(total=length)
    i_dir=os.path.basename(os.path.normpath(path))
    print("Loading images from "+ i_dir+ " folder: ")
    data, labels = [], []
    for dirname, _, filenames in os.walk(path):
        count=0
        #with alive_bar() as bar:
        for filename in filenames:
            photo_path = os.path.join(dirname, filename).replace("\\","/")
            photo_class = dirname.split('\\')[-1]
            if count==0:
                print("Loading " +photo_class +" class images")
            count+=1
            try:
                img=cv.imread(photo_path)
                data.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
                #plt.imshow(img)
                #plt.show()
                #input_im.append(cv2.resize(read_im, resize))
                #pbar.update(n=1)
                # female == 0
                if photo_class == 'female':
                    labels.append(0)
                # male == 1
                elif photo_class == 'male':
                    labels.append(1)
            except:
                print("Can't load image")
    # return list of images and another list of correponding labels
    print("Finished loading images\n")
    return data, labels

def lbp_features(data, radius, num_points):
    print("Extracting LBP features")
    result = []
    for image in data:
        lbp = feature.local_binary_pattern(image, num_points, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),density=True, bins=range(0, num_points + 3),range=(0, num_points + 2))
        eps = 1e-7
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        result.append(hist)
    print("Finished extraction")
    return result

def svm_selector(trn_data, trn_labels, tst_data, tst_labels,param_grid):
    print("Starts calculating best parameters and kernel")
    clf_grid = GridSearchCV(svm.SVC(probability=True), param_grid, verbose=1)
    clf_grid.fit(trn_data, trn_labels)
    y_pred=clf_grid.predict(tst_data)
    return clf_grid.best_params_, accuracy_score(y_pred, tst_labels)

def svm_classifier(train_data, train_labels, test_data, test_labels,c,gamma,kernel):
    print("creting the best classifier")
    clf = svm.SVC(kernel=kernel,C=c,gamma=gamma)
    # Train classifier
    clf.fit(train_data, train_labels)
    # Make predictions on unseen test data
    clf_predictions = clf.predict(test_data)
    matrix = confusion_matrix(test_labels,clf_predictions, labels=[0,1])
    return {'C':c, 'Gamma':gamma, 'Kernel':kernel},clf.score(test_data, test_labels) * 100,matrix

def max_accuracy():
    print("Starts building the model")
    d1={}
    d2={}
    d1,acc1=svm_selector(trn1, train_labels,val1,valid_labels,param_grid)
    d2,acc2=svm_selector(trn2, train_labels,val2,valid_labels,param_grid)
    print("Now it's time to compare")
    if acc1>acc2:
        tst1= lbp_features(test_data, 1, 8)
        d1,acc1,mat1=svm_classifier(trn1, train_labels, tst1, test_labels,d1['C'],d1['gamma'],d1['kernel'])
        print("Done building the model")
        return d1,acc1,"method1",mat1
    else:
        tst2= lbp_features(test_data, 3, 24)
        d2,acc2,mat2=svm_classifier(trn2, train_labels, tst2, test_labels,d2['C'],d2['gamma'],d2['kernel'])
        print("Done building the model")
        return d2,acc2,"method2",mat2


train_data, train_labels = load_im(path_train, count_files(path_train))
valid_data, valid_labels =load_im(path_val, count_files(path_val))
test_data, test_labels =load_im(path_test, count_files(path_test))

trn1=lbp_features(train_data, 1, 8)
trn2=lbp_features(train_data, 3, 24)

val1= lbp_features(valid_data, 1, 8)
val2= lbp_features(valid_data, 3, 24)

param_grid = {'C': [0.1, 1, 10, 100],
 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10], 'kernel':['rbf','linear']}

d={}
d, a, m, mat=max_accuracy()

table=[["male",mat[1][1],mat[0][1]],["female",mat[1][0],mat[0][0]]]

print("Outpust all the results to 'results.txt'")

with open('results.txt', 'w') as f:
    if m=="method1":
        f.write("Radius: 1, Num Of Points: 8\n")
    else:
        f.write("Radius: 3, Num Of Points: 24\n")
    f.write("Kernel: {}, Gamma: {}, C: {}\n".format(d['Kernel'],d['Gamma'],d['C']))
    f.write("Accuracy: {:.2f}%\n".format(a))
    f.write(tabulate(table, headers=["male","female"]))
    
    
print("Finished")