from ensembled_model import EnsembledModel
from pathlib import Path
import random
from PIL import Image
from sklearn.metrics import confusion_matrix
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import pickle

params = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0.25,0.25,0.25,0.25],[0.2,0.3,0.3,0.2],[0.15,0.35,0.35,0.15],[0.1,0.4,0.4,0.1]]

count = 0

for i in params:
    model = EnsembledModel(i)

    data_train_path = Path('data/train')
    data_test_path = Path('data/test')

    image_train_list  = list(data_train_path.glob('*/*.jpg'))
    image_test_list = list(data_test_path.glob('*/*.jpg'))
    random.shuffle(image_train_list)
    random.shuffle(image_test_list)

    image_train_list = image_train_list[:200]
    image_test_list = image_test_list[:200]

    train_acc = 0

    for i in image_train_list:
            real_label = int(str(i).split('\\')[-2])
            img = Image.open(i)
            label_dict = model.forward(img)
            if label_dict['label'] == real_label:
                train_acc+=1
    train_acc = train_acc/200

    print('Train accuracy of the model : ',train_acc)

    #________________________________________________________________

    test_acc = 0

    for i in image_test_list:
            real_label = int(str(i).split('\\')[-2])
            img = Image.open(i)
            label_dict = model.forward(img)
            if label_dict['label'] == real_label:
                test_acc+=1
    test_acc = test_acc/200

    print('Test accuracy of the model : ',test_acc)

    #________________________________________________________________

    data_train_path = Path("data/train")  

    random.shuffle(image_test_list)

    real_labels = []
    pred_labels = []

    for i in image_train_list:
        real_label = int(str(i).split(os.sep)[-2])  
        img = Image.open(i)
        label_dict = model.forward(img)
        real_labels.append(real_label)
        pred_labels.append(label_dict['label'])

    real_labels = np.array(real_labels)
    pred_labels = np.array(pred_labels)

    cm_train = confusion_matrix(real_labels, pred_labels)
    n_classes = cm_train.shape[0]

    # Calculate recall for each class
    recall_per_class_train = []
    for i in range(n_classes):
        TP = cm_train[i, i]  # True Positives for class i
        FN = np.sum(cm_train[i, :]) - TP  # False Negatives for class i
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        recall_per_class_train.append(recall)

    print("Recall for each class for train data :", recall_per_class_train)

    # ________________________________________________________________

    data_test_path = Path("data/test")  

    random.shuffle(image_test_list)

    real_labels = []
    pred_labels = []

    for i in image_test_list:
        real_label = int(str(i).split(os.sep)[-2])  
        img = Image.open(i)
        label_dict = model.forward(img)
        real_labels.append(real_label)
        pred_labels.append(label_dict['label'])

    real_labels = np.array(real_labels)
    pred_labels = np.array(pred_labels)

    cm_test = confusion_matrix(real_labels, pred_labels)
    n_classes = cm_test.shape[0]

    recall_per_class_test = []
    for i in range(n_classes):
        TP = cm_test[i, i]  # True Positives for class i
        FN = np.sum(cm_test[i, :]) - TP  # False Negatives for class i
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        recall_per_class_test.append(recall)

    print("Recall for each class for test data :", recall_per_class_test)

    #________________________________________________________________

    data_train_path = Path("data/train")  

    random.shuffle(image_train_list)

    real_labels = []
    pred_labels = []

    for i in image_train_list:
        real_label = int(str(i).split(os.sep)[-2])  
        img = Image.open(i)
        label_dict = model.forward(img)
        real_labels.append(real_label)
        pred_labels.append(label_dict['label'])

    real_labels = np.array(real_labels)
    pred_labels = np.array(pred_labels)

    cm_train = confusion_matrix(real_labels, pred_labels)
    n_classes = cm_train.shape[0]

    precision_per_class_train = []
    for i in range(n_classes):
        TP = cm_train[i, i]  
        FP = np.sum(cm_train[:, i]) - TP  
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        precision_per_class_train.append(precision)

    print("Precision for each class for train data :", precision_per_class_train)
    #________________________________________________________________

    data_test_path = Path("data/test")  

    random.shuffle(image_test_list)

    real_labels = []
    pred_labels = []

    for i in image_test_list:
        real_label = int(str(i).split(os.sep)[-2])  
        img = Image.open(i)
        label_dict = model.forward(img)
        real_labels.append(real_label)
        pred_labels.append(label_dict['label'])

    real_labels = np.array(real_labels)
    pred_labels = np.array(pred_labels)

    cm_test = confusion_matrix(real_labels, pred_labels)
    n_classes = cm_test.shape[0]

    precision_per_class_test = []
    for i in range(n_classes):
        TP = cm_test[i, i]  
        FP = np.sum(cm_test[:, i]) - TP  
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        precision_per_class_test.append(precision)

    print("Precision for each class for test data :", precision_per_class_test)

    #________________________________________________________________

    recall_per_class_train = np.array(recall_per_class_train)
    recall_per_class_test = np.array(recall_per_class_test)
    precision_per_class_train = np.array(precision_per_class_train)
    precision_per_class_test = np.array(precision_per_class_test)

    f1_score_train = (2 * recall_per_class_train * precision_per_class_train)/(recall_per_class_train+precision_per_class_train)

    print('F1 score for each class train : ', f1_score_train.tolist())

    #________________________________________________________________

    f1_score_test = (2*recall_per_class_test*precision_per_class_test)/(recall_per_class_test+precision_per_class_test)

    print('F1 score for each class test : ',f1_score_test.tolist())

    #________________________________________________________________
    data_train_path = Path("data/train")  

    random.shuffle(image_train_list)

    real_labels = []
    pred_probs = []  # Store predicted probabilities

    for i in image_train_list:
        real_label = int(str(i).split(os.sep)[-2])  # Use os.sep for cross-platform compatibility
        img = Image.open(i)
        probabilities = model.forward(img)  # Assuming this returns a list of probabilities
        real_labels.append(real_label)
        pred_probs.append(probabilities['probs'][0])

    real_labels = np.array(real_labels)
    pred_probs = np.array(pred_probs)

    # Binarize the true labels
    n_classes = len(np.unique(real_labels))
    y_true_binarized = label_binarize(real_labels, classes=np.arange(8))

    # Compute ROC Curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc_train = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], pred_probs[:, i])
        roc_auc_train[i] = auc(fpr[i], tpr[i])

    print('ROC and AUC curve of training : ',roc_auc_train)
    #________________________________________________________________

    data_test_path = Path("data/test")  

    random.shuffle(image_test_list)

    real_labels = []
    pred_probs = []  

    for i in image_test_list:
        real_label = int(str(i).split(os.sep)[-2])  
        img = Image.open(i)
        probabilities = model.forward(img)  
        real_labels.append(real_label)
        pred_probs.append(probabilities['probs'][0])

    real_labels = np.array(real_labels)
    pred_probs = np.array(pred_probs)

    # Binarize the true labels
    n_classes = len(np.unique(real_labels))
    y_true_binarized = label_binarize(real_labels, classes=np.arange(8))

    # Compute ROC Curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc_test = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], pred_probs[:, i])
        roc_auc_test[i] = auc(fpr[i], tpr[i])

    print('ROC and AUC curve of test : ',roc_auc_test)

    #________________________________________________________________

    metrics_dict = {
        'train_acc' : train_acc,
        'test_acc' : test_acc,
        'train_recall' : recall_per_class_train,
        'test_recall' : recall_per_class_test,
        'train_precision' : precision_per_class_train,
        'test_precision' : precision_per_class_test,
        'train_f1' : f1_score_train,
        'test_f1' : f1_score_test,
        'train_roc_auc' : roc_auc_train,
        'test_roc_auc' : roc_auc_test
    }
    count+=1
    with open('metrics/final_ensembled_metrics_{}.pkl'.format(count),'wb') as f:
        pickle.dump(metrics_dict,f)