# system level
import sys
import json
import os
import codecs

# arrays
import numpy as np
from numpy import random
from scipy import interp

# keras
from keras.models import model_from_json
from keras.utils import np_utils
from keras import backend as K
from keras import models

# sklearn (for machine learning)
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import brier_score_loss
from scipy.stats import sem


#model plotting
import pydotplus
import keras.utils

# plotting
from matplotlib import pyplot as plt
import pylab as pl
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import matplotlib.patches as mpatches
from vis.visualization import visualize_cam
from vis.utils import utils
from sklearn.metrics import precision_recall_curve

# reporting
from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, FlushLeft, MediumText
from pylatex import Plot, Figure, Matrix, Alignat, MultiColumn, Command, SubFigure, NoEscape, HorizontalSpace
from pylatex.utils import italic, bold

##############################################################
##############################################################





# rewrite to plot all 4 images TP,TN/FP,FN

def examples_plot(images, nrows, ncols):#WORKS
# ------------------------------------------------------------------------------
# Funciton plots images given in examples
# ------------------------------------------------------------------------------
    fig1=plt.figure(figsize=(5,5))
    for i, image in enumerate(images):
        plt.subplot(nrows, ncols, i + 1)
        plt.axis("off")
        plt.imshow(image, aspect='auto', cmap='viridis', norm=LogNorm())
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()
    plt.savefig('images/examples1.pdf')
    return



def load_plot_model(json_file):#WORKS
# ------------------------------------------------------------------------------
# Funciton loads model architecture and plotsit
# ------------------------------------------------------------------------------
    json_file = open(json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    keras.utils.vis_utils.pydot = pydotplus
    keras.utils.plot_model(loaded_model, to_file='images/model.pdf', show_shapes=True)
    return



def true_pred (y_test, y_prob):#WORKS!
# ------------------------------------------------------------------------------
# Function plots true labels vs predictions for train, validaiton and test set
# ------------------------------------------------------------------------------
    fig, axis1 = plt.subplots(figsize=(8,8))
    plt.scatter(y_test, y_pred, label='test')
    plt.plot([0,1], [0,1], 'k--', label="1-1")
    plt.xlabel("Truth")
    plt.ylabel("Prediction")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('images/true_pred.pdf')
    return


def loss_acc_plot_novalidation(loss, acc):#WORKS!
# ------------------------------------------------------------------------------
# Funciton plots a combined loss and accuracy plot for training and validation set
# ------------------------------------------------------------------------------
    epochs_list = list(range(len(loss)))

    figsize=(6,4)
    fig, axis = plt.subplots(figsize=figsize)
    plot_acc = axis.plot(epochs_list, acc, 'navy', label='accuracy')

    plot_loss = axis.plot(epochs_list, loss, 'red', label='loss')

    axis.set_xlabel('Epoch')
    axis.set_ylabel('Loss/Accuracy')
    plt.tight_layout()
    axis.legend(loc='center right')
    plt.savefig('images/loss_acc.pdf')
    return


def loss_acc_plot(loss, val_loss, acc, val_acc):#WORKS!
# ------------------------------------------------------------------------------
# Funciton plots a combined loss and accuracy plot for training and validation set
# ------------------------------------------------------------------------------
    epochs_list = list(range(len(loss)))

    figsize=(6,4)
    fig, axis = plt.subplots(figsize=figsize)
    plot_lacc = axis.plot(epochs_list, acc, 'navy', label='accuracy')
    plot_val_lacc = axis.plot(epochs_list, val_acc, 'deepskyblue', label="validation accuracy")

    plot_loss = axis.plot(epochs_list, loss, 'red', label='loss')
    plot_val_loss = axis.plot(epochs_list, val_loss, 'lightsalmon', label="validation loss")

    axis.set_xlabel('Epoch')
    axis.set_ylabel('Loss/Accuracy')
    plt.tight_layout()
    axis.legend(loc='center right')
    plt.savefig('images/loss_acc.pdf')
    return



def prec_recall_plot(y_test, probability):#WORKS!
# ------------------------------------------------------------------------------
# Funciton plots a combined precision and recall plot for training and validation set
# ------------------------------------------------------------------------------
    precision, recall, thresholds = precision_recall_curve(y_test, probability)
    auc = auc(recall, precision)

    figsize=(6,4)
    plt.plot(precision, recall, 'r')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.tight_layout()
    plt.savefig('images/prec_recall.pdf')
    return auc


def conf_matrix(y_true, y_pred, normalize=False, cmap=plt.cm.Blues):#WORKS!
# ------------------------------------------------------------------------------
# Outputs the confusion matrix for arbitrary number of classes
# ------------------------------------------------------------------------------
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #if not title:
    if normalize:
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)


    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('images/conf.pdf')
    return



def CM_ROC(x_test,y_test, loaded_model):#WORKS but not needed (I need to add ROC separately)
# ------------------------------------------------------------------------------
# Evaluate classificaiton
# Outputs the confusion matrix and ROC curve vith its AUC 
# ------------------------------------------------------------------------------
    # predict
    prob = loaded_model.predict(x_test)
    pred =  (prob > 0.5).astype('int32') 

    # measure confusion
    labels=[0, 1]
    cm = metrics.confusion_matrix(y_test, pred, labels=labels)
    cm = cm.astype('float')
    cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]
    print("cm", cm)
    print("cm_norm", cm_norm)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, prob, pos_label=1)
    auc = metrics.roc_auc_score(y_test, prob)
    np.save('images/auc.npy',auc)
    print("AUC:", auc)

    #plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix',y=1.08)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    fmt = '.2f'
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, format(cm_norm[i, j], fmt),
            ha="center", va="center",
            color="white" if cm_norm[i, j] < thresh else "black")
    pl.savefig('images/conf.pdf')
    plt.show()
    
    #ROC
    figsize=(5,5)
    fig, axis1 = plt.subplots(figsize=figsize)
    x_onetoone = y_onetoone = [0, 1]

    plt.plot(fpr, tpr, 'r-')
    plt.plot(x_onetoone, y_onetoone, 'k--',  label="1-1")
    plt.legend(loc=0)
    plt.title("Receiver Operator Characteristic (ROC)")
    plt.xlabel("False Positive (1 - Specificity)")
    plt.ylabel("True Positive (Selectivity)")
    plt.tight_layout()
    pl.savefig('images/ROC.pdf')
    return prob, pred, auc



def scores(y_test, probability, prediction):#WORKS
# ------------------------------------------------------------------------------
# Evaluate classificaiton
# Outputs the accuracy, precision, recall, f1 score, brier score of classificaiton
# ------------------------------------------------------------------------------
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, prediction)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, prediction)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, prediction)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, prediction)
    print('F1 score: %f' % f1)
    #brier score
    br = brier_score_loss(y_test, probability)
    print('Brier score is: %f' % br)
    scoring = np.array([accuracy, precision, recall, f1, br])
    np.save('images/scoring.npy',scoring)
    return accuracy, precision, recall, f1, br



def histogram(outputs, num_class, bin_num):#WORKS
# ------------------------------------------------------------------------------
# Funciton plots a nice histogram for 3 merger subsamples
# ------------------------------------------------------------------------------
    bins = bin_num
    for i in range(num_class):
        plt.hist(outputs[i], bins, alpha=0.9, label='class_'+str(i+1))
        plt.xlabel("CNN Output")
        plt.ylabel("Frequency in test set")
        plt.savefig('images/histogram.pdf')
    plt.show()
    return



def twoD_histogram (prob_all, prob_TP_TN, variable_all, variable_TP_TN):#WORKS
# ------------------------------------------------------------------------------
# Plot 2D histogram of the distribution of all entire positive/negative class vs TP/TN
# vs one object parameter
# ------------------------------------------------------------------------------
    sns.set_style("white")
    plt.ylabel('CNN Output')
    plt.xlabel('Stellar Mass')
    plt.xlim(9.4, 11.8)
    plt.xticks([9.5, 10, 10.5, 11, 11.5])
    sns.kdeplot(variable_TP_TN, prob_TP_TN, cmap="RdGy",  n_levels=10)
    sns.kdeplot(variable_all_P_N, prob_all_P_N, cmap="coolwarm", n_levels=10)

    r = sns.color_palette("RdGy")[0]
    b = sns.color_palette("coolwarm")[0]

    red_patch = mpatches.Patch(color=r, label='TP/TN')
    blue_patch = mpatches.Patch(color=b, label='all positives/negatives')
    plt.legend(handles=[red_patch,blue_patch],loc='lower right')
    plt.show()
    plt.savefig('images/2Dhistogram.pdf')
    return


def bootstraping_scores (y_test, probability, prediction, boot_num, rand_seed):#WORKS
# ------------------------------------------------------------------------------
# Produces bootstraped errors for all scoring methods
# auc, accuracy, precision, recall, f1 score, brier score
# ------------------------------------------------------------------------------    

    y_pred = probability
    y_true = y_test
    pred = prediction


    n_bootstraps = boot_num     #number of bootstrap samples we want
    rng_seed = rand_seed        # control reproducibility
    bootstrapped_roc_auc = []
    bootstrapped_accuracy = []
    bootstrapped_precision = []
    bootstrapped_recall = []
    bootstrapped_f1 = []
    bootstrapped_brier = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        roc_auc = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_roc_auc.append(roc_auc)
   
        score_acc = accuracy_score(y_true[indices], pred[indices])
        bootstrapped_accuracy.append(score_acc)
    
        score_precision = precision_score(y_true[indices], pred[indices])
        bootstrapped_precision.append(score_precision)
    
        score_recall = recall_score(y_true[indices], pred[indices])
        bootstrapped_recall.append(score_recall)
    
        score_f1 = f1_score(y_true[indices], pred[indices])
        bootstrapped_f1.append(score_f1)
    
        score_brier = brier_score_loss(y_true[indices], y_pred[indices])
        bootstrapped_brier.append(score_brier)
        
        
    return bootstrapped_roc_auc,bootstrapped_accuracy, bootstrapped_precision,bootstrapped_recall,bootstrapped_f1,bootstrapped_brier



def confidence_interval (bootstrapped_score,score):#WORKS
# ------------------------------------------------------------------------------
# Calculates 95% CI from the bootstraped values
# ------------------------------------------------------------------------------
    sorted_scores = np.array(bootstrapped_score)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]

    confidence_lower1 = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper1 = sorted_scores[int(0.975 * len(sorted_scores))]

    print("Original ROC area: {:0.3f}".format(score))

    print("90% Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
        confidence_lower, confidence_upper))
    print("95% Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
        confidence_lower1, confidence_upper1))
    print("95% Errors are: [{:0.3f} , {:0.3}]".format(
        confidence_lower1-score, confidence_upper1-score),"\n")
    return


#WORKS
def bootstraped_hist (bootstrapped_roc_auc, bootstrapped_accuracy, bootstrapped_precision, bootstrapped_recall, bootstrapped_f1, bootstrapped_brier):
# ------------------------------------------------------------------------------
# Plot all bootstrap histograms
# ------------------------------------------------------------------------------

    plt.hist(bootstrapped_roc_auc, bins=50)
    plt.title('Histogram of the bootstrapped ROC AUC scores')
    plt.show()
    
    plt.hist(bootstrapped_accuracy, bins=50)
    plt.title('Histogram of the bootstrapped accuraciy scores')
    plt.show()

    plt.hist(bootstrapped_precision, bins=50)
    plt.title('Histogram of the bootstrapped precision scores')
    plt.show()

    plt.hist(bootstrapped_recall, bins=50)
    plt.title('Histogram of the bootstrapped recall scores')
    plt.show()

    plt.hist(bootstrapped_f1, bins=50)
    plt.title('Histogram of the bootstrapped F1 scores')
    plt.show()
    return


def roc_with_CI (y_test, prob, num_boot, rand_seed):#WORKS
# ------------------------------------------------------------------------------
# Bootstrap the ROC curve in y directon
# ------------------------------------------------------------------------------

    y_pred = prob
    y_true = y_test

    fpr, tpr, thresholds = metrics.roc_curve(y_test, prob, pos_label=1)
    
    n_bootstraps = num_boot   #number of bootstrap samples we want
    rng_seed = rand_seed      # control reproducibility
    bootstrapped_fpr = []
    bootstrapped_tpr = []


    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        fpr_score, tpr_score, thresholds_score = metrics.roc_curve(y_true[indices], y_pred[indices], pos_label=1)

        bootstrapped_fpr.append(fpr_score)
        bootstrapped_tpr.append(tpr_score)

        
    tprs = []
    base_fpr = np.linspace(0, 1, 1001)   
    for i in range(len(bootstrapped_fpr)):    
        tpr1 = interp(base_fpr, bootstrapped_fpr[i], bootstrapped_tpr[i])
        tpr1[0] = 0.0
        tprs.append(tpr1)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    #https://dfrieds.com/math/confidence-intervals   for 95%CI
    tprs_upper_95 = mean_tprs - 1.96*std 
    tprs_lower_95 = mean_tprs + 1.96*std
    
        
    #plot
    plt.figure(figsize=(6, 6))
    x_onetoone = y_onetoone = [0, 1]
    prist = plt.plot(fpr, tpr, 'navy', linewidth=2, label='pristine images')
    prist1 = plt.fill_between(base_fpr, tprs_lower_95, tprs_upper_95, color='deepskyblue', alpha=0.9, label='95%CI pristine images')
    line11 = plt.plot(x_onetoone, y_onetoone, 'k--',linewidth=1, label="1-1")

    plt.legend(loc='lower right')

    plt.xlabel("False Positive (1 - Specificity)")
    plt.ylabel("True Positive (Selectivity)")
    plt.tight_layout()
    plt.savefig('images/ROC_CI.pdf')
    return


def report():
    import numpy as np
#if __name__ == '__main__':
    #image_filename = os.path.join(os.path.dirname(__file__), 'histogram.jpg')
    example_TP = 'images/examples1.pdf'
    example_FP = 'images/examples2.pdf'
    example_TN = 'images/examples1.pdf'
    example_FN = 'images/examples2.pdf'
    
    model = 'images/model.pdf'
    
    lo_acc = 'images/loss_acc.pdf'
    pr_rec = 'images/prec_recall.pdf'
    
    hist = 'images/histogram.pdf'
    hist_2d = 'images/2Dhistogram.pdf'
    
    conf_matrix = 'images/conf.pdf'
    roc = 'images/ROC.pdf'
    #roc_ci = 'images/ROC_CI.pdf'
    
    true_pred = 'images/true_pred.pdf'
    
    scoring = np.load('images/scoring.npy')
    auc_save = np.load('images/auc.npy')
    auc_pr = np.load('images/auc_pr.npy')
    auc = auc_save
    acc = scoring[0]
    precision = scoring[1]
    recall = scoring[2]
    f1 = scoring[3]
    brier = scoring[4]
    
    
    geometry_options = {"tmargin": "1.5cm", "lmargin": "2.5cm"}
    doc = Document(geometry_options=geometry_options)

    with doc.create(Section('CLASSIFICATION REPORT', numbering=0)):
        
        
        # CNN architecture
        with doc.create(Subsection('Architecture of the Neural Network', numbering=0)):
            with doc.create(Figure(position='h!')) as loss_acc:
                loss_acc.add_image(model, width='200px')   
                
        
        # plot some example images
        with doc.create(Subsection('TP/FP/TN/FN Test Set Examples', numbering=0)):
            doc.append('TP - true positives, TN - true negatives, FP - false positives, FN - False negaties')
            with doc.create(Figure(position='h!')) as imagesRow1:
                doc.append(Command('centering'))  
                with doc.create(
                    SubFigure(position='c',  width=NoEscape(r'0.33\linewidth'))) as left_image:
                    left_image.add_image(example_TP, width=NoEscape(r'0.95\linewidth'))
                    left_image.add_caption("Examples of TP")
                
                with doc.create(
                    SubFigure(position='c', width=NoEscape(r'0.33\linewidth'))) as right_image:
                    right_image.add_image(example_FP, width=NoEscape(r'0.95\linewidth'))
                    right_image.add_caption("Examples of FP")
                
            with doc.create(Figure(position='h!')) as imagesRow2:
                doc.append(Command('centering'))  
                with doc.create(
                    SubFigure(position='c',  width=NoEscape(r'0.33\linewidth'))) as left_image:
                    left_image.add_image(example_TN, width=NoEscape(r'0.95\linewidth'))
                    left_image.add_caption("Examples of TN")
                
                with doc.create(
                    SubFigure(position='c', width=NoEscape(r'0.33\linewidth'))) as right_image:
                    right_image.add_image(example_FN, width=NoEscape(r'0.95\linewidth'))
                    right_image.add_caption("Examples of FN")
                
                
         # True values VS predicted output values
        with doc.create(Subsection('Comparison of True Labes and Output Values for Test Set:', numbering=0)):
            with doc.create(Figure(position='h!')) as tr_pr:
                tr_pr.add_image(true_pred, width='200px') 
                
        # Training loss / accuracy
        with doc.create(Subsection('Training and Validation Loss and Accuracy:', numbering=0)):
            with doc.create(Figure(position='h!')) as loss_acc:
                loss_acc.add_image(lo_acc, width='260px')        
                
        # Training precision / recall
        with doc.create(Subsection('Test Set Precission and Recall:', numbering=0)):
            doc.append('''Precision/Recall curve shows the trade-off between returning accurate results (high precision), as well as returning a majority of all positive results (high recall) for different tresholds. It should be used when class imbalance problem occurs. A model with perfect classification skill is depicted as a point at (1,1). Area Under the Curve (AUC) for the perfect classifier will be 1.''')
            with doc.create(Figure(position='h!')) as pre_recall:
                pre_recall.add_image(pr_rec, width='220px')
                doc.append(HorizontalSpace("2cm"))
                doc.append(MediumText('AUC = '+ str(auc_pr)))
                
                
        # plot confusion matrix
        with doc.create(Subsection('Test Set Confusion Matrix', numbering=0)):
            with doc.create(Figure(position='h!')) as conf:
                conf.add_image(conf_matrix, width='210px')
                
                
        # all scorin matrics
        with doc.create(Subsection('Classification Scoring for Test Set', numbering=0)):
            doc.append('TP - true positives, TN - true negatives, FP - false positives, FN - False negaties \n\n')
            doc.append('The performance of a classifier can be described by:\n')
            doc.append(bold('Accuracy '))
            doc.append(' - (TP+TN)/(TP+TN+FP+FN) \n')
            doc.append(bold('Precision '))
            doc.append(' (Purity, Positive Predictive Value) - TP/(TP+FP) \n')
            doc.append(bold('Recall '))
            doc.append(' (Completeness, True Positive Rate - TP/(TP+FN) \n ')
            doc.append(bold('F1 Score '))
            doc.append(' = 2 (Precision * Recall)/(Precision + Recall).\n')
            doc.append(bold('Brier Score '))
            doc.append(''' - mean squared error (MSE) between predicted probabilities (between 0 and 1) and the expected values (0 or 1). Brier score summarizes the magnitude of the forecasting error and takes a value between 0 and 1 (with better models having score close to 0).\n\n''')
            with doc.create(Tabular('|l|l|')) as table:
                table.add_hline()
                table.add_row((bold('Metric'), bold('Score')))
                table.add_hline()
                table.add_row(('Accuracy', auc))
                table.add_row(('Precision', precision))
                table.add_row(('Recall', recall))
                table.add_row(('F1 Score',f1))
                table.add_row(('Brier Score', brier))
                table.add_hline()
            doc.append('\n\n')
       

         # plot ROC and AUC
        with doc.create(Subsection('Reciever Operating Characteristic (ROC) and Area Under the Curve (AUC) for Test Set', numbering=0)):
            doc.append('''The ROC curve graphically shows the trade-off between between true-positive rate and false-positive rate.The AUC summarizes the ROC curve - where the AUC is close to unity, classification is successful, while an AUC of 0.5 indicates the model is performs as well as a random guess.''')
            with doc.create(Figure(position='h!')) as roc_curve: 
                roc_curve.add_image(roc, width='220px')  
                doc.append(HorizontalSpace("2cm"))
                doc.append(MediumText('AUC = '+ str(auc)+'\n\n\n\n\n\n\n\n'))
                
                
        # plot histogram of output values
        with doc.create(Subsection('Histogram of the Output Probabilities for Test Set', numbering=0)):
            with doc.create(Figure(position='h!')) as histogram:
                histogram.add_image(hist, width='230px')
                
    
        # plot 2D histogram of output values and some object parameter
        with doc.create(Subsection('2D Histogram of the Output vs One Object Feature for Test Set', numbering=0)):
            with doc.create(Figure(position='h!')) as histogram_2d:
                histogram_2d.add_image(hist_2d, width='230px')
                
                
    doc.generate_pdf('report', clean_tex=False)
    return

report()