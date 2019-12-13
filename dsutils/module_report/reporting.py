def report (model=False, 
            examples=False, 
            tr_pr=False,  
            lo_acc=False, 
            pr_rec=False, 
            score=False, 
            conf_matrix=False, 
            roc_auc=False, 
            auc_pr=False, 
            hist=False, 
            hist2D=False):
    
    import numpy as np
    import os
    from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, FlushLeft, MediumText
    from pylatex import Plot, Figure, Matrix, Alignat, MultiColumn, Command, SubFigure, NoEscape, HorizontalSpace
    from pylatex.utils import italic, bold

    geometry_options = {"tmargin": "1.5cm", "lmargin": "2.5cm"}
    doc = Document(geometry_options=geometry_options)
    
    with doc.create(Section('CLASSIFICATION REPORT', numbering=0)):
        
        
        # CNN architecture
        if model==True:
            if os.path.exists('images/model.pdf'):
                model = 'images/model.pdf'
                with doc.create(Subsection('Architecture of the Neural Network', numbering=0)):
                    with doc.create(Figure(position='h!')) as loss_acc:
                        loss_acc.add_image(model, width='200px')   
            else:
                print("Model architecture image not found! Skipping.")
                          
        
        # plot some example images
        if examples==True:
            if (os.path.exists('images/examples1.pdf') and os.path.exists('images/examples2.pdf') 
            and os.path.exists('images/examples3.pdf') and os.path.exists('images/examples4.pdf')):
                example_TP = 'images/examples1.pdf'
                example_FP = 'images/examples2.pdf'
                example_TN = 'images/examples3.pdf'
                example_FN = 'images/examples4.pdf'
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
            else:
                print("Example images not found! Skipping.")   
           
        
        # True values VS predicted output values        
        if tr_pr==True:
            if os.path.exists('images/true_pred.pdf'):
                true_pred = 'images/true_pred.pdf'
                with doc.create(Subsection('Comparison of True Labes and Output Values for Test Set:', numbering=0)):
                    with doc.create(Figure(position='h!')) as tr_pr:
                        tr_pr.add_image(true_pred, width='200px')         
            else:
                print("Image comparing true labes and output values not found! Skipping.")
                        
        
        # True values VS predicted output values
        if lo_acc==True:
            if os.path.exists('images/loss_acc.pdf'):
                lo_acc = 'images/loss_acc.pdf'
                with doc.create(Subsection('Training and Validation Loss and Accuracy:', numbering=0)):
                    with doc.create(Figure(position='h!')) as loss_acc:
                        loss_acc.add_image(lo_acc, width='260px')            
            else:
                print("Loss Accuracy plot not found! Skipping.")          
               
            
        # Training precision / recall        
        if pr_rec==True:
            if os.path.exists('images/prec_recall.pdf'):
                pr_rec = 'images/prec_recall.pdf'
                with doc.create(Subsection('Test Set Precission and Recall:', numbering=0)):
                    doc.append('''Precision/Recall curve shows the trade-off between returning accurate results (high precision), as well as returning a majority of all positive results (high recall) for different tresholds. It should be used when class imbalance problem occurs. A model with perfect classification skill is depicted as a point at (1,1). Area Under the Curve (AUC) for the perfect classifier will be 1.''')
                    with doc.create(Figure(position='h!')) as pre_recall:
                        pre_recall.add_image(pr_rec, width='220px')
                        doc.append(HorizontalSpace("2cm"))
                        doc.append(MediumText('AUC = '+ str(auc_pr)))   
            else:
                print("Precision Recall plot not found! Skipping.")
        
                
        # plot confusion matrix       
        if conf_matrix==True: 
            if os.path.exists('images/conf.pdf'):
                conf_matrix = 'images/conf.pdf'
                with doc.create(Subsection('Test Set Confusion Matrix', numbering=0)):
                    with doc.create(Figure(position='h!')) as conf:
                        conf.add_image(conf_matrix, width='210px')
            else:
                print("Confusion matrix not found! Skipping.")
                        
                
        # all scoring matrics       
        if score==True:
            if (os.path.exists('images/scoring.npy') and os.path.exists('images/auc.npy')):
                scoring = np.load('images/scoring.npy')
                auc_save = np.load('images/auc.npy')
                auc = auc_save
                acc = scoring[0]
                precision = scoring[1]
                recall = scoring[2]
                f1 = scoring[3]
                brier = scoring[4]
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
            else:
                print("Scoring file not found! Skipping.")       

                
        # plot ROC and AUC        
        if roc_auc==True:
            if (os.path.exists('images/ROC.pdf') and os.path.exists('images/auc.npy')):
                roc = 'images/ROC.pdf'
                auc_save = np.load('images/auc.npy')
                auc = auc_save
                with doc.create(Subsection('Reciever Operating Characteristic (ROC) and Area Under the Curve (AUC) for Test Set', numbering=0)):
                    doc.append('''The ROC curve graphically shows the trade-off between between 
                    true-positive rate and false-positive rate.The AUC summarizes the ROC curve - where the 
                    AUC is close to unity, classification is successful, while an AUC of 0.5 indicates the 
                    model is performs as well as a random guess.''')
                    with doc.create(Figure(position='h!')) as roc_curve: 
                        roc_curve.add_image(roc, width='220px')  
                        doc.append(HorizontalSpace("2cm"))
                        doc.append(MediumText('AUC = '+ str(auc)+'\n\n\n\n\n\n\n\n'))
            else:
                print("Roc curve image and/or AUC not found! Skipping.")       
       
    
        # plot histogram of output values        
        if hist==True:
            if os.path.exists('images/histogram.pdf'):
                hist = 'images/histogram.pdf'
                with doc.create(Subsection('Histogram of the Output Probabilities for Test Set', numbering=0)):
                    with doc.create(Figure(position='h!')) as histogram:
                        histogram.add_image(hist, width='230px')
            else:
                print("Histogram image not found! Skipping.")       
               
            
        # plot 2D histogram of output values and some object parameter
        if hist2D==True:         
            if os.path.exists('images/2Dhistogram.pdf'):
                hist_2d = 'images/2Dhistogram.pdf'
                with doc.create(Subsection('2D Histogram of the Output vs One Object Feature for Test Set', numbering=0)):
                    with doc.create(Figure(position='h!')) as histogram_2d:
                        histogram_2d.add_image(hist_2d, width='230px')
            else:
                print("2D Histogram image not found! Skipping.")  
                
                
    doc.generate_pdf('report', clean_tex=False)
    return
