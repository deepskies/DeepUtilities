

def report():
    import numpy as np
    from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, FlushLeft, MediumText
    from pylatex import Plot, Figure, Matrix, Alignat, MultiColumn, Command, SubFigure, NoEscape, HorizontalSpace
    from pylatex.utils import italic, bold
    
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
