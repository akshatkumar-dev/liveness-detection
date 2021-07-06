import numpy as np
from sklearn.svm import SVC
import joblib
from metric import metric

def load_feature_label(file_name):
    feature_label = np.load(file_name) #loading the features from the npy file
    return feature_label[:,:-1],feature_label[:,-1].astype(np.uint8) #returning the features

def train():
    train_feature,train_label = load_feature_label("train_feature.npy") #loading the extracted features of the training videos
    model = SVC(kernel='rbf', C=1e3, gamma=0.5, class_weight='balanced', probability=True) #using SVC classifier to classify video as fake or real
    model.fit(train_feature, train_label) #fitting the classifier into the model
    joblib.dump(model, "model.m") #saving the model for later use
    predict_proba = model.predict_proba(train_feature) # predicting the using the model
    predict = model.predict(train_feature)
    acc,eer,hter = metric(predict_proba,train_label)
    print("train acc is:%f eer is:%f hter is:%f"%(acc,eer,hter))
    
def test():
    test_feature,test_label = load_feature_label("test_feature.npy") #loading the extracted features of the testing videos
    model = joblib.load("model.m") #loading the saved model
    predict_proba = model.predict_proba(test_feature) #predicting using the saved model
    predict = model.predict(test_feature)
    acc,eer,hter = metric(predict_proba,test_label)
    print("test acc is:%f eer is:%f hter is:%f"%(acc,eer,hter))
if __name__ == "__main__":
    #train()
    test()
