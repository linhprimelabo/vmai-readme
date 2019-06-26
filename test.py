import numpy as np
import os
import sys
import cv2
import colorsys
import pickle
from shutil import copyfile

from pose_estimation import load_pose_estimation
e = load_pose_estimation("432x368")

import vm_proccesing
from extract_keras_resnet50 import extract_features

from sklearn import preprocessing
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split

from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
import xgboost as xgb

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

averaged_models = AveragingModels(models = (GBoost, KRR, model_lgb ) )


item_labels = ['top', 'skirts', 'outer', 'pants(bottom)', 'OnePiece']
pattern_labels = ['border', 'strip', 'dot', 'plain', 'flaid', 'flower']
gender_labels = ['Ladies', 'Mens']
POSE_POINT = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar']

def get_pose(res):
    for key, val in res["pose_ponts"].items():
        print (POSE_POINT[int(key)], ':', val)
        
    return res["pose_ponts"]

def object_infor(obj):
    print (obj["discription"])
    print (obj["bounding_box"])

    hsv_color = obj["hsv_value"][0][1]
    print (hsv_color)
    rgb = colorsys.hsv_to_rgb(hsv_color[0]/360, hsv_color[1]/100, hsv_color[2]/100)
    rgb = np.uint8([i*255 for i in rgb])
    print (rgb)

def predict_item(vect):
    item_model = "models/items1800_88_model.sav"

    clf = pickle.load(open(item_model, 'rb'))
    testPredict = clf.predict([vect])    

    i = testPredict[0]
    return item_labels[i]

def predict_pattern(vect):
    pattern_model = "models/patterns1000_linear85_model.sav"

    clf = pickle.load(open(pattern_model, 'rb'))
    testPredict = clf.predict([vect])    
    
    i = testPredict[0]
    return pattern_labels[i]

def predict_gender(vect):
    gender_model = "models/gender_model.sav"

    clf = pickle.load(open(gender_model, 'rb'))
    testPredict = clf.predict([vect])
    
    i = testPredict[0]
    return gender_labels[i]

def cal_length(item, gender, top, bottom, point):
    len_model = "models/0.73_fin_leng-model.sav"
    p0 = point[0]
    p1 = point[1]
    p2 = point[2]
    p5 = point[5]
    #print (p0[0])
    #print (p1[1])
    #print (p2[0])
    #print (p5[1])
    i = item_labels.index(item)
    g = gender_labels.index(gender)
    vect = np.array([i, g, top, bottom, p0[0], p0[1], p1[0], p1[1], p2[0], p2[1], p5[0], p5[1]])
    vect = np.float64(vect).flatten()

    reg = pickle.load(open(len_model, 'rb'))
    predict_len = reg.predict([vect])[0]
    return predict_len
    
def main(image_path):
    res = vm_proccesing.vm_unit_proccess(image_path,e, False)
    print (res["type"])
    points = get_pose(res)

    objects = res["objects"]
    if len(objects)==1:
        vect = extract_features(image_path).flatten()
        item = predict_item(vect)
        print (item)
        print (predict_pattern(vect))
        gender = predict_gender(vect) 
        print (gender)
        print (cal_length(item, gender, objects[0]["bounding_box"]["top"], objects[0]["bounding_box"]["bottom"], points))
    for obj in res["objects"]:
        object_infor(obj)    

    img_inp = cv2.imread(image_path)
    cv2.imshow("Input image", img_inp)
    cv2.waitKey(0)
    
if __name__=="__main__":
    #print (item_labels.index('top'))
    main('501199194U2K.jpg')
