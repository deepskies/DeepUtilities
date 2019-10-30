'''from https://github.com/deepskies/stronglenses/blob/master/engine/view_curves.py'''

def get_ROC(model_nm, nb_thresh=1000):
    ''' Return (selectivity, sensitivity)
        Each array has length (nb_thresh+1):
    '''
    preds_by_class = get_preds_by_class(model_nm)
    threshs = np.arange(-1.0/nb_thresh, 1.0+2.0/nb_thresh, 1.0/nb_thresh)
    threshs = {0: 1-threshs, 1:threshs}
    recalls_by_class = {c: (np.searchsorted(preds, threshs[c])
                           .astype(float) / len(preds)) * -1.0 + 1.0
                        for c, preds in preds_by_class.items()}
    return tuple(recalls_by_class[c] for c in classes)
    
def get_cums(model_nm, bins=1000, eps=1e-6):
    ''' Return domain [0.5, 1.0] of conf, cumulative counts of correct
        predictions on sets of given min conf, and cumulative counts of
        all predictions on those sets.
    '''
    preds = get_prediction(model_nm)
    confidences = np.arange(0.5, 1.0 + 1.0/(2*bins), 1.0/(2*bins))
    correct_histogram = np.zeros(bins + 1) + eps
    total_histogram = np.zeros(bins + 1) + eps
    for p, label in zip(preds, y):
        c = abs(p-0.5)
        index = int(c*bins) 
        total_histogram[index] += 1.0
        if round(p) != label: continue
        correct_histogram[index] += 1.0
    correct_cum = np.cumsum(correct_histogram[::-1])[::-1]
    total_cum = np.cumsum(total_histogram[::-1])[::-1]
    correct_cum = blur(correct_cum, sigma=bins/40)
    total_cum = blur(total_cum, sigma=bins/40)
    return confidences, correct_cum, total_cum 
    
def auc_from_roc(sensitivities, selectivities):
    ''' Computes `area under the curve` by linear interpolation
    '''
    r_old, d_old = 1.0, 0.0 # at 0 threshold, perfect recall
    auc = 0.0
    for r, d in list(zip(sensitivities, selectivities)) + [(0.0, 1.0)]:
        auc += ((r+r_old)/2) * (d-d_old)
        r_old, d_old = r, d
    return auc

def roc_curve(model_nm):
    ''' A `receiver-operating characteristic curve`
        plots sensitivity vs selectivity.
    '''
    ds, rs = get_ROC(model_nm)
    return '%s has auc=%.3f' % (model_nm, auc_from_roc(rs, ds)), \
           ds, rs
           
def logroc_curve(model_nm, eps=1e-6):
    ''' A `logscale receiver-operating characteristic curve`
        plots f(sensitivity) vs f(selectivity), where
        f(x) = -log(1-x). We use a numerical fudge factor.
    '''
    f = lambda x: -np.log(1-x + eps)
    ds, rs = get_ROC(model_nm)
    return '%s has auc=%.3f' % (model_nm, auc_from_roc(rs, ds)), \
           f(ds), f(rs)

def conf_curve(model_nm):
    ''' A `confidence curve` plots accuracy vs min conf '''
    confs, correct_cum, total_cum = get_cums(model_nm)
    return model_nm, confs, (correct_cum/total_cum)
YIELDS = [0.8]
def str_round(tup, sigfigs=3):
    ''' Return string representation of tuple of floats. 
        TODO: put in utils.terminal.
    '''
    return str(tuple(round(val, sigfigs) for val in tup))
 
def yield_curve(model_nm):
    ''' A `yield curve` plots accuracy vs yield.
        The blurring done in `get_cums` potentially makes
        `total_cum` non-monotonic; we remedy this by chopping
        at the argmax.
    '''
    confs, correct_cum, total_cum = get_cums(model_nm)
    i = np.argmax(total_cum)
    yields = (total_cum/max(total_cum))[i:]
    accs = (correct_cum/total_cum)[i:]
    ACCS = [get_acc_at_yield(yields, accs, Y) for Y in YIELDS]
    return model_nm if not YIELDS else '%s has acc %s at yield %s' % \
           (model_nm, str_round(ACCS, 3), str_round(YIELDS, 2)), \
           yields, accs 

curve_getters_by_mode = {
    'yield': yield_curve,
    'conf': conf_curve,
    'roc': roc_curve,
    'logroc': logroc_curve
}


def compute_curve(model_nm, mode): 
    return curve_getters_by_mode[mode](model_nm)

def view_curves():
    ''' Interactively save and display yield or confidence curves
        of specified model.
        TODO: merge with history viewing, and refactor all
    '''
    global YIELDS

    mode = 'yield'  
    for command in user_input_iterator():
        words = command.split()
        if not words:
            continue

        if words[0] in curve_getters_by_mode:
            mode = words[0]
            if mode=='yield':
               YIELDS = map(float, words[1:])
            print(colorize('{BLUE}switched to `%s` mode{GREEN}' % mode))
            continue

        min_y = float('inf') 
        model_nms = words
        for nm in model_nms:
            try:
                color = get('MODEL.%s.PLOTCOLOR' % nm)
            except KeyError:
                print(colorize('{RED}Oops! I do not see model %s!{GREEN}' % nm))
                continue
            label, xvals, yvals = compute_curve(nm, mode)
            min_y = min(min_y, np.amin(yvals))
            plt.plot(xvals, yvals, label=label,
                     color=color, ls='-', lw=2.0)

        if mode=='yield':
            for Y in YIELDS:
                plt.plot([Y, Y], [min_y, 1.0], 
                          color='k', ls='-', lw=1.0)
            plt.title('Yield curves of %s' % ', '.join(model_nms))
            plt.gca().set_xlabel('Fraction F of data')
            plt.gca().set_ylabel('Accuracy on top F of data, by confidence')
            plt.xlim([0.0, 1.0])
            plt.ylim([min_y, 1.0])
        elif mode=='conf':
            plt.plot(xvals, xvals, label='truth',
                     color='k', ls='-', lw=1.0)
            plt.title('Confidence curves of %s' % ', '.join(model_nms))
            plt.gca().set_xlabel('Confidence threshold C')
            plt.gca().set_ylabel('Accuracy on data on which model is at least C confident')
            plt.xlim([0.5, 1.0])
            plt.ylim([0.5, 1.0])
        elif mode=='roc':
            plt.title('(Reflected) ROC curves of %s' % ', '.join(model_nms)) 
            plt.gca().set_xlabel('Selectivity p(guess = - | truth = -)')
            plt.gca().set_ylabel('Sensitivity p(guess = + | truth = +)')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
        elif mode=='logroc':
            plt.title('(Reflected and Distorted) ROC curves of %s' % ', '.join(model_nms)) 
            plt.gca().set_xlabel('Selectivity log(1.0/p(guess = + | truth = -))')
            plt.gca().set_ylabel('Sensitivity log(1.0/p(guess = - | truth = +))')
            plt.xlim([0.0, 14.0])
            plt.ylim([0.0, 14.0])
        else:
            assert(False)
        plt.
legend(loc='best')
        fig_nm = '_vs_'.join(model_nms) + '.%s.png' % mode
        plt.savefig(get('TRAIN.FIGURE_DIR') + '/' + fig_nm)
        plt.show()

        
''' from  https://github.com/deepskies/deeplensing-old/blob/master/OLD/Plotting.py'''
# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def make_mosaic(imgs, nrows, ncols, border=1):
    """Make a nice mosaic.
    
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols;
    intended for use with activation layer
    """
    # set up data
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    mosaic = np.ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                           ncols * imshape[1] + (ncols - 1) * border),
                           dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic


# ------------------------------------------------------------------------------
def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Image plot."""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax,
                   interpolation='nearest', cmap=cmap)
    plt.colorbar(im, cax=cax)


# ------------------------------------------------------------------------------
def plot_roc(dir_analysis_figure, fpr, tpr):
    """Plot the ROC."""
    # plot

    sns.set_style("white")
    sns.set_style("ticks")

    fig, axis1 = plt.subplots(figsize=(10,8))
    sns.set_palette("coolwarm", n_colors=6)
    sns.set_color_codes()
    lw = 3
    legend_label_size = "xx-large"

    plt.plot(fpr, tpr, 'r-', lw=lw, label="ROC curve")
    plt.plot([0, 1], [0, 1], 'k--', lw=lw, label="1-1")
    plt.xlabel("False positive", fontsize=25)
    plt.ylabel("True positive", fontsize=25)

    axis1.grid(False)
    axis1.tick_params(labeltop='off', labelright='off')
    axis1.xaxis.set_ticks_position('bottom')
    axis1.yaxis.set_ticks_position('left')

    legend1 = plt.legend(loc='lower right', shadow=True, prop={'size':25})
    for label in legend1.get_texts():
            label.set_fontsize(legend_label_size)
    for label in legend1.get_lines():
        label.set_linewidth(lw)  # the legend line width

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        top='off',
        labelsize=20)         # ticks along the top edge are off
    plt.tick_params(
        axis='y',          
        which='both',      
        labelsize=20)      

    # save to file
    # make output directory if necessary
    make_dir(dir_analysis_figure)
    file_save = dir_analysis_figure + "roc.png"
    plt.savefig(file_save)

    # auc = metrics.roc_auc_score(ytrue, yscore)
    # print "auc:", auc

    return file_save


# ------------------------------------------------------------------------------
def plot_confusion_matrix(dir_analysis_figure, cm, label_quadrant=True, labels=[0,1]):
    """Plot the confusion matrix."""
    # check shape
    assert cm.shape == (2,2), "Not the right shape for input cm"

    sns.set_style("white")
    sns.set_style("ticks")
    n_colors=300
    # cmap = sns.cubehelix_palette(n_colors, start=2.8, rot=.1, as_cmap=True) #, start=1, rot=1.2, as_cmap=True)
    cmap = sns.color_palette("coolwarm", n_colors=n_colors)#, color_codes=True)
    cmap = ListedColormap(cmap)

    # plot
    plt.figure()
    plt.imshow(np.transpose(cm[::-1,::-1]), 
               interpolation='nearest', cmap=cmap, extent=[1.5,-0.5,-0.5,1.5])
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted', fontsize=20)
    plt.ylabel('True', fontsize=20)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20) 
    plt.tight_layout()

    if label_quadrant:
        tp_x, tp_y = 1, 1
        tn_x, tn_y = 0, 0
        fn_x, fn_y = 0, 1
        fp_x, fp_y = 1, 0

        verticalalignment = 'center'
        myformat = '{:04.3f}'
        tp_str = "TP:\n" + myformat.format(cm[tp_y,tp_x])
        tn_str = "TN:\n" + myformat.format(cm[tn_y,tn_x])
        fp_str = "FP:\n" + myformat.format(cm[fp_y,fp_x])
        fn_str = "FN:\n" + myformat.format(cm[fn_y,fn_x])
        plt.text(tn_x, tn_y, tn_str,
                 fontsize=25, color='white', fontweight='bold',
                 verticalalignment=verticalalignment, 
                 horizontalalignment='center')
        plt.text(tp_x, tp_y, tp_str,
                 fontsize=25, color='white', fontweight='bold',
                 verticalalignment=verticalalignment,
                 horizontalalignment='center')
        plt.text(fp_x, fp_y, fp_str,
                 fontsize=25, color='white', fontweight='bold',
                 verticalalignment=verticalalignment,
                 horizontalalignment='center')
        plt.text(fn_x, fn_y, fn_str,
                 fontsize=25, color='white', fontweight='bold',
                 verticalalignment=verticalalignment,
                 horizontalalignment='center')

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        top='off',
        labelsize=20)         # ticks along the top edge are off
    plt.tick_params(
        axis='y',          
        which='both',      
        labelsize=20)      

    label_dict = {0: "Non", 1: "Lens"}
    labels_lab = [ label_dict[labels[0]], label_dict[labels[1]] ]  
    plt.xticks(labels, labels_lab)
    plt.yticks(labels, labels_lab)

    make_dir(dir_analysis_figure)
    file_save = dir_analysis_figure + "confusion.png"
    plt.savefig(file_save)

    return file_save


# ------------------------------------------------------------------------------
def plot_history(dir_analysis_figure, history):
    """Plot history as a function of epoch."""
    # get variables

    sns.set_style("white")
    sns.set_style("ticks")

    loss = history["loss"]
    val_loss = history["val_loss"]
    frac_loss = [val_l_tmp / l_tmp for l_tmp, val_l_tmp in zip(loss, val_loss)]
    # diff_loss = [val_l_tmp - l_tmp for l_tmp, val_l_tmp in zip(loss, val_loss)]
    epochs = np.arange(len(loss))

    #plot
    # cmap = sns.color_palette("coolwarm")
    # cmap = ListedColormap(cmap)
    sns.set_palette("coolwarm", n_colors=6)#, color_codes=True)
    sns.set_color_codes()
    lw = 3

    fig, axis1 = plt.subplots()
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        top='off',
        labelsize=20)         # ticks along the top edge are off
    plt.tick_params(
        axis='y',          
        which='both',      
        labelsize=20)      
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    axis1.plot(epochs, loss, 'b', lw=lw, label='loss')
    axis1.plot(epochs, val_loss, 'r', lw=lw, label="val loss")
    legend1 = plt.legend(loc='upper left', shadow=True)

    axis2 = axis1.twinx()
    legend_label_size = "large"
    axis2.plot(epochs, frac_loss, 'k--', lw=lw, label="frac: loss / val loss")
    legend2 = plt.legend(loc='upper right', shadow=True)
    for label in legend1.get_texts():
        label.set_fontsize(legend_label_size)

    for label in legend1.get_lines():
        label.set_linewidth(lw)  # the legend line width

    axis2.grid(False)
    axis1.grid(False)
    axis1.tick_params(labeltop='off', labelright='off')
    axis2.tick_params(labeltop='off')
    axis1.xaxis.set_ticks_position('bottom')
    axis2.xaxis.set_ticks_position('bottom')
    # axis1.yaxis.set_ticks_position('left')
    # axis1.xaxis.set_ticks_position('bottom')

    axis1.set_xlabel('Epoch', fontsize=20)
    plt.title("Loss", fontsize=15)
    axis1.set_ylabel('Loss', fontsize=20)
    axis2.set_ylabel('Fractional Loss', fontsize=20)
    plt.tight_layout()
    for label in legend2.get_texts():
        label.set_fontsize(legend_label_size)

    for label in legend2.get_lines():
        label.set_linewidth(lw)  # the legend line width

    epochs_new = [ep + 1 for ep in epochs]
    epochs_lab = [str(ep) for ep in epochs_new]
    plt.xticks(epochs, epochs_lab)

    sns.set(font_scale=1)
    make_dir(dir_analysis_figure)
    file_save = dir_analysis_figure + "loss.png"
    plt.savefig(file_save)
    print file_save

    return file_save

    
# ------------------------------------------------------------------------------
def plot_probability_distribution(dir_analysis_figure, y_prob_set):
    """Plot history as a function of epoch."""
    sns.set_style("white")
    sns.set_style("ticks")

    fig, axis1 = plt.subplots(figsize=(10,8))

    n_colors = 4
    sns.cubehelix_palette(n_colors, start=2.8, rot=.1) #, start=1, rot=1.2, as_cmap=True)
    sns.set_palette("coolwarm", n_colors=n_colors )
    sns.set_color_codes()
    lw = 5
    legend_label_size = "xx-large"
    nbins = 7


    plt.title("Probability Distributions")
    plt.xlabel('probability', fontsize=20)
        

    axis1.grid(False)
    axis1.tick_params(labeltop='off', labelright='off')
    axis1.xaxis.set_ticks_position('bottom')
    axis1.yaxis.set_ticks_position('left')


    y_prob_all = y_prob_set['all']
    y_prob_tp = y_prob_set['tp']
    y_prob_fp = y_prob_set['fp']
    y_prob_tn = y_prob_set['tn']
    y_prob_fn = y_prob_set['fn']

    hist_all, bins_all = np.histogram(y_prob_all, bins=nbins)
    center_all = (bins_all[:-1] + bins_all[1:]) / 2

    hist_tp, bins_tp = np.histogram(y_prob_tp, bins=nbins)
    center_tp = (bins_tp[:-1] + bins_tp[1:]) / 2

    hist_fp, bins_fp = np.histogram(y_prob_fp, bins=nbins)
    center_fp = (bins_fp[:-1] + bins_fp[1:]) / 2

    hist_tn, bins_tn = np.histogram(y_prob_tn, bins=nbins)
    center_tn = (bins_tn[:-1] + bins_tn[1:]) / 2

    hist_fn, bins_fn = np.histogram(y_prob_fn, bins=nbins)
    center_fn = (bins_fn[:-1] + bins_fn[1:]) / 2

    plt.yscale('log')
    #plt.xscale('log')
    axis1.plot(center_all, hist_all,  lw=lw, color='k', ls=":", label="All")
    axis1.plot(center_tp, hist_tp,  lw=lw, ls='-', label="TP")
    axis1.plot(center_fp, hist_fp,  lw=lw, ls='-', label="FP")
    axis1.plot(center_tn, hist_tn, lw=lw, ls='--', label="TN")
    axis1.plot(center_fn, hist_fn,  lw=lw, ls='--', label="FN")

    legend1 = plt.legend(loc='upper left', shadow=True, prop={'size':25})
    for label in legend1.get_texts():
        label.set_fontsize(legend_label_size)
    for label in legend1.get_lines():
        label.set_linewidth(lw)  # the legend line width

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        top='off',
        labelsize=20)         # ticks along the top edge are off
    plt.tick_params(
        axis='y',          
        which='both',      
        labelsize=20)       

    make_dir(dir_analysis_figure)
    file_save = dir_analysis_figure + "probability_distribution.png"
    plt.savefig(file_save)

    return file_save

# ------------------------------------------------------------------------------
def plot_layer_panel(dir_analysis_figure,
                     layer_object,
                     n_img_y=6,
                     n_img_x=6,
                     cmap=cm.jet):
    """Plot: result of a layer; all filters."""
    c1 = np.squeeze(layer_object)
  
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)

    # plot
    plt.figure(figsize=(12, 12))
    plt.title('Layer')
    nice_imshow(plt.gca(), make_mosaic(c1, n_img_x, n_img_y), cmap=cmap)

    # save
    make_dir(dir_analysis_figure)
    file_save = dir_analysis_figure + "layer_panel.png"
    plt.savefig(file_save)

    return file_save


# ------------------------------------------------------------------------------
def plot_layer_confusion_panel(dir_analysis_figure,
                               layer_matrix,
                               hspace=0.1, 
                               wspace=0.1,
                               name="test",
                               label_list=None,
                               figsize=(10,10),
                               make_tick_labels=False,
                               make_axis_grid=False,
                               universal_palette=True):
    """Make Layer Matrix panel."""
    fig = plt.figure(0, figsize=figsize)


    n_grid_x = 4

    n_grid_y = layer_matrix[:, 0].shape[0]
    print "n_grid_y", n_grid_y
    
    gs = gridspec.GridSpec(n_grid_y,
                           n_grid_x,
                           hspace=hspace,
                           wspace=wspace)


    column_label = ["TP", "FP", "TN", "FN"]
    icol = 0
    col_count = 1.
    for col in range(4):
        x = layer_matrix[:, col]

        # light = 3./col_count
        for row in range(4):
            axis = fig.add_subplot(gs[col,row])
           
            if universal_palette: 
                cmap = sns.cubehelix_palette(8,
                        gamma=0.6, rot=0.0, start=0.0,
                        dark=0.0, light=1.1, as_cmap=True)
                palette = "univ_palette"
            else:
                cmap = sns.color_palette("RdBu_r", int(1e3))
                cmap = ListedColormap(cmap)
                palette = "warm_palette"
            if axis.is_last_row():
                mult_fac = -1.0
            else:
                mult_fac = 1.0

            axis.imshow(mult_fac*x[row],
                        aspect='equal', alpha=1, 
                        origin=None, cmap=cmap, 
                        interpolation="kaiser")
                        # vmin=vmin, vmax=vmax)
                        # norm=LogNorm())

            if axis.is_first_row():
                if label_list is not None:
                    label_value = ": " + str(label_list[icol])
                else:
                    label_value = ""
                axis.set_xlabel(column_label[icol] + label_value, fontsize = 14)
                axis.xaxis.set_label_position('top') 
                icol += 1

            if not make_tick_labels:
                axis.grid(False)

            if not make_tick_labels:
                axis.set_xticklabels([])
                axis.set_yticklabels([])

                for tl in axis.xaxis.get_major_ticks():
                    tl.tick1On = tl.tick2On = False
                for tl in axis.yaxis.get_major_ticks():
                    tl.tick1On = tl.tick2On = False
        col_count += 1
        
    f_io = dir_analysis_figure + "layer_confusion_panel_" + name + "_" + palette + ".png"
    plt.savefig(f_io)
    plt.close()
    print 'Output (multipanel figure):', f_io
    
    
    
''' REPORT from https://github.com/deepskies/deeplensing-old/blob/master/deeplensing/plotting/deepreporting.py'''    
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def create_report(params, params_model, dirs, figure_dict, data_analysis):
    """Create report."""
    # report filename
    filename_report = dirs['analysis'] + "report_" + str(params["id_model"]).zfill(3) + "_" + str(params["id_analysis"]).zfill(3)
    

    # Create Document
    # geometry_options = {"tmargin": "-2.875in", "oddsidemargin": "-.875in", "evensidemargin":"-.875in", 
    #                    "textwidth":"0.75in", "textheight": "2.75in"}
    #geometry_options = {"tmargin": "-.875in", "lmargin": "1.5cm", "footskip":"30pt", "textheight":""}
    #geometry_options = {"margin": "-0.5in"}

    # create document
    doc = Document() #geometry_options=geometry_options)

    doc.preamble.append(NoEscape(r'\usepackage{float}'))
    #doc.preamble.append(NoEscape(r'\extrafloats{100}'))

    # Display basic analysis parameters
    with doc.create(Section('Analysis Parameters')):
        with doc.create(Tabular('c|c|c')) as table:
            table.add_row(("name", "id number", 'number of objects'))
            table.add_hline()
            table.add_hline()
            table.add_row(("id_model", params["id_model"], '-'))
            table.add_hline()
            table.add_row(("id_analysis", params["id_analysis"], '-'))
            table.add_hline()
            table.add_row(("id_data_train", params["id_data_train"], params_model["nb_train"]))
            table.add_hline()
            table.add_row(("id_data_valid", params["id_data_valid"], params_model["nb_valid"]))
            table.add_hline()
            table.add_row(("id_data_test", params["id_data_test"], params["nb_test"]))
            table.add_hline()


    with doc.create(Section('Confusion Diagnostics')):
        with doc.create(Tabular('c|c')) as table:
            table.add_hline()
            table.add_hline()
            table.add_row(('AUC', data_analysis['roc']['auc']))
            table.add_hline()
            table.add_hline()
            table.add_row(('Quality', data_analysis['quality']))
            table.add_hline()
            table.add_hline()
            table.add_row(('Confusion Elements', ''))
            table.add_hline()
            table.add_hline()
            for key, value in data_analysis['cm_elem'].items():
                table.add_row((key,value))
                table.add_hline()

    # model parameters
    with doc.create(Section('Training Parameters')):
        with doc.create(Tabular('c|c')) as table:
            table.add_hline()
            table.add_hline()
            metadata = dm.load_metadata_log(dirs['learnlist'], id_model=params["id_model"])
            #print 'metadata type', isinstance(metadata, dict)
            for key, val in metadata.items():
                table.add_row((key,val))
                table.add_hline()

    doc.append(NewPage()) 

    # model parameters
    with doc.create(Section('Model Summary')):
        file_model_layers = dirs["model"] + "model_summary.txt"
        f = open(file_model_layers, 'r')
        x = f.read()
        f.close()
        doc.append(x)

    doc.append(NewPage()) 
    # Diagnostics Figures
    with doc.create(Section('Diagnostic Figures')):
        doc.append(NewPage()) 
        ifig = 0
        print "figures"
        for key, val in figure_dict.items():
            try:
                print ifig, key, val
                with doc.create(Figure(position='')) as temp_object:
                    temp_object.add_image(figure_dict[key]['filename'], width=figure_dict[key]['width'])
                    temp_object.add_caption(key)
                ifig+=1
            except:
                pass

            if ifig % 10 == 0:
                 doc.append(NoEscape(r'\clearpage'))
            #    doc.append(NewPage())
   


    # generate pdf
    doc.generate_pdf(filename_report, clean_tex=False)
    return filename_report + ".pdf"

''' entire https://github.com/deepskies/deeplensing-old/blob/master/deeplensing/ImagePanel.py   could be useful '''
''' also https://github.com/deepskies/deeplensing-old/blob/master/deeplensing/deeprun.py '''

''' notebook With a REPORT example https://github.com/deepskies/deeplensing-old/blob/master/notebooks/.ipynb_checkpoints/analysis_safe-checkpoint.ipynb '''
''' https://github.com/deepskies/deeplensing-old/blob/master/notebooks/.ipynb_checkpoints/plot_layers-checkpoint.ipynb''' 


''' Not for CNN but still a nice plot '''
''' https://github.com/deepskies/tlens/blob/master/tsne_mnist.ipynb '''
def plot_mnist(X, y, X_embedded, name, min_dist=10.0):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(frameon=False)
    plt.title("\\textbf{MNIST dataset} -- Two-dimensional "
          "embedding of 70,000 handwritten digits with %s" % name)
    plt.setp(ax, xticks=(), yticks=())
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                    wspace=0.0, hspace=0.0)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, marker="x")
    print y.shape
    print y[0]
    print len(y)
    print y.ndim
    print type(y)
    if min_dist is not None:
        from matplotlib import offsetbox
        shown_images = np.array([[15., 15.]])
        indices = np.arange(X_embedded.shape[0])
        np.random.shuffle(indices)
        for i in indices[:5000]:
            dist = np.sum((X_embedded[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist:
                continue
            shown_images = np.r_[shown_images, [X_embedded[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(X[i].reshape(28, 28),
                                      cmap=plt.cm.gray_r), X_embedded[i])
            ax.add_artist(imagebox)
            
    plt.show()
    
   
''' SALIENCY MAPS: https://github.com/deepskies/deepmhd/blob/master/PB19_saliency_maps.ipynb'''
