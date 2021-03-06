# -*- coding: utf-8 -*-
"""Diagnostics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19bf30ZZofd3zsCNPyRQEg4y3ZhInwWy-
"""


import os
import argparse

import seaborn as sn
import numpy as np
import pandas as pd
import sklearn
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils.multiclass import unique_labels


class Diagnostics(object):
    """
    Class that creates plots for regression or classification problems in machine learning problems

    Attributes:
        actual - list or array, the true labels inputted into model
        predicted - list or array, the predicted labels outputted by your model
        acc - list or array, in format: [acc_train_per_epoch, acc_validation_per_epoch]
        loss - list or array, in format: [loss_train_per_epoch, loss_validation_per_epoch]
        auc - list or array, in format: [auc_train_per_epoch, auc_validation_per_epoch]
        feature_list - list, features used to train model
        cross_val
        data_batch - array, batch of image pixels
        labels_batch - array, labels matching data_batch images
    """

    # Plot axes formatting
    plt.rcParams['axes.linewidth']=3
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['xtick.minor.width'] = 2
    plt.rcParams['ytick.minor.width'] = 2
    plt.rc('xtick.major', size=8, pad=8)
    plt.rc('xtick.minor', size=6, pad=5)
    plt.rc('ytick.major', size=8, pad=8)
    plt.rc('ytick.minor', size=6, pad=5)

    def __init__(self, config, path, predicted, actual, acc=[0,0], loss=[0.0], auc=[0,0], feature_list=0, cross_val=0, data_batch=0, labels_batch=0):
        # Mandatory lists for diagnostics
        self.actual = actual
        self.predicted = predicted
        self.path = path + '/plots/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        plot_config = config['plot_config']
        self.show = plot_config['show']
        self.save = plot_config['save']

        # User-added lists for diagnostics
        self.acc=acc
        self.loss=loss
        self.auc = auc
        self.feature_lists=feature_list
        self.cross_val=cross_val
        self.data = data_batch
        self.labels_batch = labels_batch

    def plot_metrics_per_epoch(self, figsize = (15, 4), name_plot=[0,1,2], figname="metrics.png", show=True, save_individual=False):
        """
        Plots accuracy, loss, and auc curves per epoch

        Input:
        - figsize: tuple, the size of the metric curve
        - name_plot: list, whether you want '0' (loss), '1' (accuracy), and/or '2' (auc) plots, default plots all three
        - figname: string, what you want the combined plot to be saved as
        - show: boolean, whether you want to plt.show() your figure or just save it to your computer
        - save_individual: boolean, whether you want to plot and save individual metrics or if you want one large plot
                           with subplots for each metric.
        """
        num_graphs = len(name_plot)
        plot_name = ["Loss", "Accuracy", "AUC"]
        try:
            metrics = np.array([(self.loss[0], self.loss[1]), (self.acc[0], self.acc[1]), (self.auc[0], self.auc[1])])
        except:
            raise Exception("Inputs for any or all of loss, acc, auc are not properly formatted. Please input lists of with [train, test]")

        # if you want to plot individual metrics
        if save_individual:
            for i in name_plot:
                print(i)
                plt.figure(figsize=figsize)
                metric_train = metrics[i][0]
                metric_val =metrics[i][1]

                # check_for_array(metric_train, plot_name[i] + " Train"); check_for_array(metric_val, plot_name[i] + " Validation")

                plt.plot(metric_train, '-', color='seagreen', label='Training')
                plt.plot(metric_val, '--', color='blue', label='Validation')

                plt.title(plot_name[i] + " Per Epoch", fontsize=20)
                plt.xlabel("Epoch", fontsize=14)
                plt.ylabel(plot_name[i], fontsize=14)
                plt.savefig(plot_name[i] + ".png", bbox_inches='tight')
                if show: plt.show()
                plt.close()

        # if you want to plot metrics as subplots in one figure
        else:
            fig, axes = plt.subplots(nrows=1, ncols=num_graphs, figsize=figsize)
            for i in range(num_graphs):
                metric_train = metrics[i][0]
                metric_val = metrics[i][1]

                check_for_array(metric_train, plot_name[i] + " Train"); check_for_array(metric_val, plot_name[i] + " Validation")

                axes[i].plot(metric_train, '-', color='seagreen', label='Training')
                axes[i].plot(metric_val, '--', color='blue', label='Validation')

                axes[i].set_title(plot_name[i] + " Per Epoch", fontsize=20)

                axes[i].set_xlabel("Epoch", fontsize=16)
                axes[i].set_ylabel(plot_name[i], fontsize=16)
                axes[i].tick_params(labelsize=16)
                axes[i].legend(loc='best')

                plt.subplots_adjust(wspace=0.35, hspace=0.35)
                fig.savefig(figname, bbox_inches='tight')

                if (show): fig.show()
                plt.close()

    # need to fix this function
    def plot_cross_validation(self, figsize = (6, 4), show=True):
        fn = self.plot_path("K_fold_Cross_Validation.png")
        plt.close()

    # need to fix this function
    def plot_cross_validation(self, figsize = (6, 4), figname="metrics.png", show=True):
        """
        Plots Cross validation

        Input:
        - figsize: tuple, the size of the metric curve
        - figname: string, what you want the combined plot to be saved as
        - show: boolean, whether you want to plt.show() your figure or just save it to your computer
        """
        file_name = "K_fold_Cross_Validation.png"
        # check_for_array(self.cross_val, "Cross_Val")
        plt.figure(figsize=figsize)
        plt.tile("K-fold Cross Validation", fontsize=14)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.ylabel("Folds", fontsize=14)
        plt.xlabel("Accuracy", fontsize=14)
        # plt.plot(self.loss)
        plt.plot(self.cross_val)
        plt.savefig(file_name)
        if show: plt.show()
        plt.close()

    def ROC_plot(self, figsize = (6, 4), show=True):
        """
        Plots the ROC curve between the predicted and actual labels
        Note: "actual" labels must be a 1D array

        Input:
          - figsize: tuple, the desired size of the image
          - show: boolean, whether you want to plt.show() your figure or just save it to your computer
        """
        actual_lbl = convert_to_index(self.actual)
        #actual_lbl = [np.argmax(array_temp) for array_temp in self.actual]
        skplt.metrics.plot_roc(actual_lbl, self.predicted, figsize=figsize)
        if show: plt.show()
        plt.close()

    def residual_dist_by_feature(self, figsize = (6,8), target='Target', hex_bin=False, show=True):
        """
        Plots residual by feature

        Input:
        - figsize: tuple, the size of the metric curve
        - target: ???
        - hex_bin: use hex bin?
        - show: boolean, whether you want to plt.show() your figure or just save it to your computer
        """
        file_name = '{}_errors_by_feature.pdf'.format(target)
        #Calculate residuals as fractional error.
        error=2*(self.predicted-self.actual)/(abs(self.actual)+abs(self.predicted))
        num_features=len(self.feature_list.columns); figure_width, figure_height = figsize
        fig=plt.figure(figsize=(figure_width, figure_height*num_features))
        for i in range(0, num_features):
            ax = fig.add_subplot(num_features, 1, i+1)
            #Plot the errors vs. feature.
            if hex_bin==True:
                ax.hexbin(feature_list[feature_list.columns[i]],error, bins='log')
            else:
                ax.plot(feature_list[feature_list.columns[i]],error, '.', alpha=0.2)
            ax.set_xlabel(feature_list.columns[i], fontsize=14)
            ax.set_ylabel('Fractional Error', fontsize=14)
            plt.rc('xtick',labelsize=14)
            plt.rc('ytick',labelsize=14)
            ax.set_title('Fractional Error as a function of {}'.format(feature_list.columns[i]), fontsize=14)
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            plt.savefig(file_name, bbox_inches=extent)
        if show: fig.show()
        fig.close()


    def residual_dist_by_feature(self, figsize = (6,8), target='Target', hex_bin=False, show=True, save_individual=False):
        """
        Plots fractional residual as a function of each feature
        There is one plot for each feature
        Inputs:
            - hex_bin = False - makes a scatter plot
            - hex_bin = True - makes a hexbin plot with a colormap based on number points
            - Target - name of the quantity that you're trying to predict
            - save_individual - can save feature graphs as multiple images or as single image
        """
        check_for_dataframe(self.feature_lists, "Feature List")
        num_features=len(self.feature_list.columns); figure_width, figure_height = figsize
        error=2*(self.predicted-self.actual)/(abs(self.actual)+abs(self.predicted)) #Fractional error constructed to deal with the case of actual being 0
        plt.rc('xtick',labelsize=14)
        plt.rc('ytick',labelsize=14)

        if save_invdividual:
            for i in range(0, num_features):
                plt.figure(figsize=figsize)
                plt.set_xlabel(feature_list.columns[i], fontsize=14)
                plt.set_ylabel('Fractional Error', fontsize=14)
                plt.set_title('Fractional Error as a function of {}'.format(feature_list.columns[i]), fontsize=14)
                if hex_bin==True:
                    plt.hexbin(feature_list[feature_list.columns[i]],error, bins='log')
                else:
                    plt.plot(feature_list[feature_list.columns[i]],error, '.', alpha=0.2)
                plt.savefig('Fractional_Error_as_a_function_of_{}.png'.format(feature_list.columns[i]))
                if (show): plt.show()
                plt.close()

        if not save_individual:
            file_name = '{}_errors_by_feature.pdf'.format(target)
            fig=plt.figure(figsize=(figure_width, figure_height*num_features))
            for i in range(0, num_features):
                ax = fig.add_subplot(num_features, 1, i+1)
                #Plot the errors vs. feature.
                if hex_bin==True:
                    ax.hexbin(feature_list[feature_list.columns[i]],error, bins='log')
                else:
                    ax.plot(feature_list[feature_list.columns[i]],error, '.', alpha=0.2)
                ax.set_xlabel(feature_list.columns[i], fontsize=14)
                ax.set_ylabel('Fractional Error', fontsize=14)
                ax.set_title('Fractional Error as a function of {}'.format(feature_list.columns[i]), fontsize=14)
            if (show): fig.show()
            plt.close()

    def one_to_one_plot(self, target_name='Target', axis_scale='linear', show=True):
        """
        Plots actual vs predicted values
        Inputs:
            - axis_scale = default is linear, but can choose any sort of axis scale
        """

        file_name = '{}_One_to_One.pdf'.format(target_name)
        plt.plot(self.actual, self.predicted, '.')
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.xlabel('True {}'.format(target_name), fontsize=14)
        plt.ylabel('Predicted {}'.format(target_name), fontsize=14)
        plt.title('One to one plot showing predicted vs. true {}'.format(target_name), fontsize=14)
        plt.xscale(axis_scale)
        plt.yscale(axis_scale)
        line_x, line_y = np.arange(min(self.actual),1.1*max(self.actual),(max(self.actual)-min(self.actual))/10), np.arange(min(self.actual),1.1*max(self.actual),(max(self.actual)-min(self.actual))/10)
        plt.plot(line_x,line_y,'r--')
        plt.savefig(file_name)
        if (show): plt.show()
        plt.close()

    def target_distributions(self, figsize=(6, 4), target='Target', x_scale='linear', y_scale='linear', show=True):
        """
        Plots a histogram of two distributions, one for the predcited values and one for the actual values
        Inputs:
            - Target - name of the quantity that you're trying to predict
            - x_scale - default is linear, but can choose any sort of axis scale
            - y_scale - default is linear, but can choose any sort of axis scale
        """

        file_name = '{}_distributions.pdf'.format(target)
        # Assign colors for each group and the names
        colors = ['#E69F00', '#56B4E9']
        names = ['True {}'.format(target), 'Predicted {}'.format(target)]

        actual_lbls = convert_to_index(self.actual)
        pred_lbls = convert_to_index(self.predicted)

        plt.figure(figsize=figsize)
        plt.hist([actual_lbls, pred_lbls], color=colors, label=names)
        plt.yscale(y_scale)
        plt.xscale(x_scale)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        # Plot formatting
        plt.legend()
        plt.xlabel(target, fontsize=14)
        plt.title('{} distributions for True and Predicted'.format(target), fontsize=14)
        plt.savefig(file_name)
        if (show): plt.show()
        plt.close()

    def plot_sample_img(self, figsize=(10, 10), filename="Image_Sample.png", show=True):
        """
        Plots data where each row of the plot consists of the same image in different channels (bands)

        Input:
          - data: array, an array of shape [batch_size, channels, height, width] OR [batch_size, height, width, channels]
          - labels: array, a 1D array of labels that match to the corresponding data
          - figsize: tuple, the figure size of the main plot
          - filename: string, saved filename
          - show: boolean, whether you want to plt.show() your figure or just save it to your computer
        """

        check_for_array(self.data, "Data batch"); check_for_array(self.labels_batch, "Label batch")

        plt.figure(figsize=figsize)

        num_imgs = len(self.data)
        counter = 1
        labels = convert_to_index(self.labels_batch)

        # if the image data is in the format [batch_size, channels, height, width]
        if (self.data.shape)[1] < (self.data.shape)[3]:
            num_bands = (self.data.shape)[1]
        # if the image data is in the format [batch_size, height, width, channels]
        else:
            num_bands = (self.data.shape)[3]

        for i in range(len(self.data)):
            for j in range(num_bands):
                #format plot horizontally
                if num_bands == 1:
                    plt.subplot(num_bands, num_imgs, counter)
                #format plot vertically
                else:
                    plt.subplot(num_imgs, num_bands, counter)
                # if data format in shape [batch_size, channels, height, width]
                if (self.data.shape)[1] < (self.data.shape)[3]:
                    plt.imshow(self.data[i][j], cmap='gray')
                # if data format in shape [batch_size, height, width, channels]
                else:
                    plt.imshow(self.data[i, :, :, j], cmap='gray')
                plt.title("Label: "+ str(labels[i]), fontsize=14)
                counter += 1

        plt.subplots_adjust(wspace=.35, hspace=.35)
        plt.savefig(filename)
        if (show): plt.show()


    def output_average_precision(self):
        average_precision = average_precision_score(self.actual, self.predicted)
        print('Average precision-recall score: {0:0.2f}'.format(average_precision))


    def precision_recall_plot(self, show=True):
        plt.figure()
        actual_lbl = convert_to_index(self.actual)
        pred_lbl = convert_to_index(self.predicted)

        preicision, recall = precision_recall_curve(actual_lbl, pred_lbl)
        tep_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
        plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
        if (show): plt.show(); plt.close()

    def run_diagnostics(self):
        """
        Runs through all of the available plotting methods and displays results
        """
        self.plot_sample_img()
        self.plot_metrics_per_epoch()
        self.plot_cm()
        self.ROC_plot()
        self.target_distributions()
        self.output_average_precision()

    def plot_path(self, plot_name):
        return self.path + plot_name


def check_for_array(arr, name):
    if (not hasattr(arr, '__len__') and (not isinstance(arr, str))):
        raise Exception(
            name + " is not in the appropriate format. Should be an array")

def check_for_same_length(arr1, name1, arr2, name2):
    if (not (len(arr1) == len(arr2))):
        raise Exception(name1 + " and " + name2 +
                        " are not the same length")

def check_for_dataframe(arr, name):
        if (not isinstance(arr, pd.DataFrame)):
            raise Exception(
                name + " is not in the appropriate format. Should be a dataframe")


def convert_to_index(array_categorical):
    array_index = [np.argmax(array_temp) for array_temp in array_categorical]
    return array_index

# Plots confusion matrix. If norm is set, values are between 0-1. Shows figure if show is set
def plot_cm(predicted, actual, figsize = (6, 4), norm=True, show=True, save_path=None, config=None, epoch=None):
    """
    Creates a confusion matrix for the predicted and actual labels for your model

    Input:
       - predicted
       - actual
       - figsize: tuple, the figure size of the desired plot
       - norm: boolean, whether or not you want your confusion matrix normalized (between 0-1)
       - show: boolean, whether you want to plt.show() your figure or just save it to your computer
       - save
    """
    if actual == None or predicted == None:
        raise NameError("Missing either actual predicted labels")

    if (len(actual) != len(predicted)):
        raise NameError("Predicted and actual labels must be same shape")

    if config:
        plot_config = config.get('plot_config')
        show = plot_config.get('show')

    #converting raw labels into a 1D list
    # actual_lbl = convert_to_index(self.actual)
    pred_lbl = convert_to_index(predicted)

    cm = confusion_matrix(actual, pred_lbl)
    plt.figure(figsize=figsize)
    labels = np.unique(pred_lbl).tolist()
    np.set_printoptions(precision=2)

    if norm:
        a = cm.sum(axis=1)[:, np.newaxis]
        b = cm.astype("float")

        # print(f'cm sum: {a}')
        # print(f'cm.astype: {b}')

        heatmap_value = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if epoch:
            file_name = f"Confusion_Matrix_Norm_{epoch}.png"
        else:
            file_name = "Confusion_Matrix_Norm.png"
        plt.title("Normalized Confusion Matrix", fontsize=18)
    else:
        heatmap_value = cm.astype('float')
        file_name = "Confusion_Matrix.png"
        plt.title("Confusion Matrix", fontsize=14)

    sn.heatmap(heatmap_value, annot=True, xticklabels=labels, yticklabels=labels,
               cmap="Blues", annot_kws={"size": 10}, fmt='.2f')

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
    if save_path:
        full_path = save_path + file_name
        plt.savefig(full_path, bbox_inches='tight')

    if show: plt.show()
    plt.close()
