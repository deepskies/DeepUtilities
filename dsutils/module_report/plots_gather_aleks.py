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

''' https://github.com/deepskies/hepedge/blob/master/LArTPCmodelFiles/LArTPCmodel_build_train_evaluate.py  '''
def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()
    
    
 ''' https://github.com/deepskies/OptClusterSpec/blob/master/FullyConnectedV2.ipynb '''
    
  opt = Adam(lr = .0001)
model.compile(loss = 'mse', optimizer = opt, metrics = ['mse', 'mae', 'mape', 'cosine']) #lower learning rate 
    
    print(history.history.keys())

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
'''#zoom in
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylim(bottom = 0, top = .1)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()'''

# summarize history for mean_squared_error
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('mean_squared_error')
plt.ylabel('mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# summarize history for mean_absolute_error
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('mean_absolute_error')
plt.ylabel('mean_absolute_error')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# summarize history for mean_absolute_percentage_error
plt.plot(history.history['mean_absolute_percentage_error'])
plt.plot(history.history['val_mean_absolute_percentage_error'])
plt.title('mean_absolute_percentage_error')
plt.ylabel('mean_absolute_percentage_error')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# summarize history for cosine
plt.plot(history.history['cosine_proximity'])
plt.plot(history.history['val_cosine_proximity'])
plt.title('cosine_proximity')
plt.ylabel('cosine_proximity')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

dict_keys(['val_loss', 'val_mean_squared_error', 'val_mean_absolute_error', 'val_mean_absolute_percentage_error', 'val_cosine_proximity', 'loss', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'])

''' https://github.com/deepskies/OptClusterSpec/blob/master/MLOpticalClusters_Mass_Calibration-RF.ipynb '''


'''https://github.com/deepskies/OptClusterSpec/blob/master/MLOpticalClusters_Halo_Model-RF.ipynb'''

#Plot actual Ngal vs. predicted Ngal.
plt.hexbin(val_y, val_preds, bins='log')
plt.ylabel('Predicted Central Galaxy Magnitude')
plt.xlabel('MICECAT Central Galaxy Magnitude')
#plt.xscale('log')
#plt.yscale('log')
line_x, line_y = np.arange(-23,-14,1), np.arange(-23,-14,1)
plt.plot(line_x,line_y,'r--')
plt.show()

''' https://github.com/deepskies/deeperCMB   a lot of fancy code '''

''' https://github.com/deepskies/astroencoder/blob/master/analysis/analysis.py '''
 def plot_losses(self):
        try:    
            losses = np.loadtxt(self.modeldir + '/log.csv', delimiter=',',
                                skiprows=1, unpack=True)
            assert(len(losses) == 2*len(self.y_true)+4)
        except IOError:
            print('File log.csv with loss data not found. Loss not plotted.')
            return None
        except AssertionError:
            print('File log.csv does not have the expected format. Loss not plotted.')
            return None
        
        epochs = range(1, len(losses[0]) + 1) #start at 1
        n = len(self.output_names)
        
        for i in range(n):
            fig, axis1 = plt.subplots(figsize=(10,6))
            plt.plot(epochs, losses[i+1], c = black, label='loss')
            plt.plot(epochs, losses[n+i+3], c = orange, label="val loss")
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.ylim((min(min(losses[i+1]), min(losses[n+i+3]))-.1,
                      1.2*max(losses[i+1,0],losses[n+i+3,0])))
            plt.title("Loss History for %s" %self.output_names[i])
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(self.modeldir+'/loss %s.png'%self.output_names[i])
            plt.show()
            
        fig, axis1 = plt.subplots(figsize=(10,6))
        plt.plot(epochs, losses[n+1], c = black, label = 'loss')
        plt.plot(epochs, losses[2*n+3], c = orange, label = "val loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim((min(min(losses[n+1]), min(losses[2*n+3]))-.1,
                  1.2*max(losses[n+1,0],losses[2*n+3,0])))
        plt.title("Total Loss History")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(self.modeldir+'/loss.png')
        plt.show()
        plt.clf()
        
    def plot_random_samples(self, howmany):
        samples = np.random.randint(self.y_true[0].shape[0], size = howmany)
        
        for ii in samples:
            self.plot_sample(ii, savefig = True)
            
    def plot_sample(self, ii, savefig = False, colors = 'hot'):
        n_in = self.x_in.shape[1]
        n_out = len(self.y_true)
        
        plt.figure()
            
        for jj in range(n_in):
            ax = plt.subplot(n_in,3,3*jj+1)
            plt.imshow(self.x_in[ii,jj], cmap=colors)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title('Input %s map'%self.input_names[jj])
        for jj in range(n_out):
            ax = plt.subplot(n_out,3,3*jj+2)
            plt.imshow(self.y_true[jj][ii,0], cmap=colors)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title('True %s map'%self.output_names[jj])
            ax = plt.subplot(n_out,3,3*jj+3)
            plt.imshow(self.y_pred[jj][ii,0], cmap=colors)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title('Predicted %s map'%self.output_names[jj])
        
        if savefig:
            plt.savefig(self.modeldir+'/samples %s.png'%ii)
        plt.show()
        plt.clf()
        
        # utility function for showing images
def show_imgs(inputs, outputs, predicted=None, indices=range(10),filename=None,
              npix=npix_default, output_as_list=True):
    ins=inputs.shape[1]
    if output_as_list: 
        outs=len(outputs)
    else:
        outs=outputs.shape[1]
    
    if predicted is not None:
        npics=ins+2*outs
    else:
        npics=ins+outs

    n=len(indices) #you can provide a set of indices you'd like to see, or just see the first 10

    plt.figure(figsize=(2*n, npics))

    for i in range(n):
        for j in range(ins):
            ax = plt.subplot(npics, n, i+1+j*n)
            plt.imshow(inputs[indices[i],j])
            plt.gray(); ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)

        for j in range(outs):
            ax = plt.subplot(npics, n, i+1+(ins+j)*n)
            if output_as_list:
                plt.imshow(outputs[j][indices[i],0])
            else:
                plt.imshow(outputs[indices[i],j])
            plt.gray(); ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)

        if predicted is not None:
            for j in range(outs):
                ax = plt.subplot(npics, n, i+1+(ins+outs+j)*n)
                if output_as_list:
                    plt.imshow(predicted[j][indices[i],0])
                else:
                    plt.imshow(predicted[indices[i],j])
                plt.gray(); ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)

    if filename is not None:
        plt.savefig(filename+".png")
    plt.show()
    plt.clf()

''' https://github.com/deepskies/deepmerge/blob/master/DeepMerge.ipynb '''

''' https://github.com/deepskies/stronglensbnns/blob/master/src/notebooks/Bayesian_CNN.ipynb''''
#@title
from scipy.stats import norm
import matplotlib.pylab as pl

def visualize_norm_dist(means, stds, labels, title, ax=None, target=None):
  if ax is None:
    fig, ax = plt.subplots(figsize=(15,6))
  palette = pl.cm.jet(np.linspace(0, 1, len(means))) 
  mx = 0
  for idx, mean in enumerate(means):
    std = stds[idx]
    x = np.linspace(mean - 4*std, mean + 4*std, 1000)
    y = norm.pdf(x, mean, std)
    mx = max(np.max(y), mx)
    ax.plot(x, y, label=labels[idx], color=palette[idx])
  if target:  
    lab_x = [target, target]
    lab_y = [-0.1*mx, 1.1*mx]
    ax.plot(lab_x, lab_y, 'k-', label='target')  
  plt.title(title)
  plt.xlabel('x')
  plt.ylabel('pdf(x)')
  plt.legend()


def plot_standard_deviations(stds, epochs, ylabel=None, xlabel=None):
  fig = plt.figure(figsize=(15, 6))
  plt.plot(epochs, stds)
  if xlabel:
    plt.xlabel(xlabel)
  if ylabel:
    plt.ylabel(ylabel)
  plt.title('Standard Deviation Change across across epochs')
  
def visualize_weight_posterior_distribution(layer_idx, nets, epochs, tensor_idx=[0,0,0,0]):
  layers = [net.layers[layer_idx] for net in nets]
  qw_means = []
  qw_stds = []
  
  for net in nets:
    layer = net.layers[layer_idx]
    ndim = len(layer.qw_mean.shape)
    qw_mean = layer.qw_mean[tensor_idx[0], tensor_idx[1], tensor_idx[2], tensor_idx[3]].item() if ndim == 4 else layer.qw_mean[tensor_idx[0], tensor_idx[1]].item()
    qw_means.append(qw_mean)
    
    alpha = torch.exp(layer.log_alpha).item()
    qw_std = np.sqrt(1e-8 + alpha * (qw_mean**2))
    qw_stds.append(qw_std)
    
  labels = ['epoch {}'.format(epoch) for epoch in epochs]
  
  qw_title = 'Posterior Distributions For Leyer {} Weight {} across epochs'.format(layer_idx, tensor_idx)
  visualize_norm_dist(qw_means, qw_stds, labels, qw_title )
  plot_standard_deviations(qw_stds, epochs)
  
  
def visualize_output_distribution(layer_idx, nets, epochs, tensor_idx=[0,0,0,0]):
  layers = [net.layers[layer_idx] for net in nets]
  output_means = []
  output_stds = []
  
  for net in nets:
    layer = net.layers[layer_idx]
    alpha = torch.exp(layer.log_alpha).item()
    output_mean = layer.conv_qw_mean if hasattr(layer, 'conv_qw_mean') else layer.fc_qw_mean
    ndim = len(output_mean.shape)
    output_mean = output_mean[tensor_idx[0], tensor_idx[1], tensor_idx[2], tensor_idx[3]].item() if ndim == 4 else output_mean[tensor_idx[0], tensor_idx[1]].item()
    output_means.append(output_mean)
    
    output_std = np.sqrt(1e-8 + alpha * (output_mean**2))
    output_stds.append(output_std)
    
  labels = ['epoch {}'.format(epoch) for epoch in epochs]
  
  output_title = 'Distributions For Leyer {} Output {} across epochs'.format(layer_idx, tensor_idx)
  visualize_norm_dist(output_means, output_stds, labels, output_title)
  plot_standard_deviations(output_stds, epochs)


'''https://github.com/deepskies/deepsz/blob/master/Analysis/Analysis_caldeira_newdata.ipynb '''
'''how F1 changes when probability treshold changes....we can do simillar or different scoring methods'''

def get_F1(df, mass_thresh='2e14', xlim=None, method='cnn'):
    if xlim is None:
        xlim = (0.8, 0.999) if method == 'cnn' else (3., 30.)
    col = f'score_wdust (trained>{mass_thresh})' if method == 'cnn' else 'mf_peaksig'
    Fscore = lambda x: _get_Fbeta(df[f'Truth(>{mass_thresh})'], (df[col]>x).astype(int))
    
    f = plt.figure(figsize=(12, 4))
    x = np.linspace(xlim[0], xlim[1])
    y = [Fscore(xx) for xx in x]
    plt.scatter(x, y)
    plt.xlim(xlim)
    plt.xlabel('%s thres'%("CNN prob" if method == 'cnn' else "MF S/N"), fontsize='xx-large')
    plt.ylabel('F1_score', fontsize='xx-large')

    plt.show(block=True)

    '''preciion recall curves'''
    def pr_curve_for_mf(min_mf=0, max_mf=100, mass_thresh='2e14', xlim=[-0.05,1.05]):
    cut_df = merged_df[(merged_df['mf_peaksig'] > min_mf) & (merged_df['mf_peaksig'] < max_mf)].copy()
    
    cut_df.sort_values(by=['mf_peaksig'], inplace=True, ascending=False)
    cut_df.reset_index(inplace=True, drop=True)
    
    index = cut_df.index.get_level_values(None)
    n = len(cut_df)
    n_true = np.sum(cut_df[f'Truth(>{mass_thresh})'])
    tps = cut_df[f'Truth(>{mass_thresh})'].cumsum()
    
    precision = tps/(index+1)
    recall = tps/n_true
    proportion_tagged = (index+1)/n 

    plt.plot(proportion_tagged, precision, label='Precision')
    plt.plot(proportion_tagged, recall, label='Recall')

    plt.xlabel('Proportion of cutouts tagged',fontsize='xx-large')
    plt.xlim(xlim)
    plt.legend()
    plt.show()
    
    
    ''' https://github.com/deepskies/deepsz/blob/master/utils/utils2.py '''
    def eval(models, get_test_func, model_weight_paths=None, pred_only=False):
    y_prob_avg = None
    y_probs = []
    x_test, y_test = get_test_func()
    num_nets = len(models)
    for i in range(num_nets):
        model = models[i]
        if model_weight_paths is not None:
            model.load_weights(model_weight_paths[i])
        y_prob = model.predict(x_test)
        y_probs.append(y_prob.squeeze())
        y_prob_avg = y_prob if y_prob_avg is None else y_prob + y_prob_avg
    y_probs = np.stack(y_probs, 0)
    y_prob_avg /= float(num_nets)
    y_pred = (y_prob_avg > 0.5).astype('int32').squeeze() # binary classification
    if pred_only:
        return y_prob_avg
    return summary_results_class(y_probs, y_test), y_pred, y_prob_avg, y_test, x_test, models

def summary_results_class(y_probs, y_test, threshold=0.5, log_roc=False, show_f1=True):
    """
        y_probs: a list of independent predictions
        y_test: true label
        threshold: predict the image to be positive when the prediction > threshold
    """
    # measure confusion matrix
    if show_f1:
        threshold, maxf1 = get_F1(y_probs.mean(0),y_test)
        threshold = threshold - 1e-7

    cm = pd.DataFrame(0, index=['pred0','pred1'], columns=['actual0','actual1'])
    cm_std = pd.DataFrame(0, index=['pred0', 'pred1'], columns=['actual0', 'actual1'])
    #memorizing the number of samples in each case (true positive, false positive, etc.)
    tp_rate, tn_rate = np.zeros(len(y_probs)), np.zeros(len(y_probs))
    for actual_label in range(2):
        for pred_label in range(2):
            cnt = np.zeros(len(y_probs))
            for i in range(len(y_probs)):
                cnt[i] = np.sum(np.logical_and(y_test == actual_label, (y_probs[i] > threshold) == pred_label))
            cm.loc["pred%d"%pred_label,"actual%d"%actual_label] = cnt.mean()
            cm_std.loc["pred%d" % pred_label, "actual%d" % actual_label] = cnt.std()

    print("Confusion matrix (cnts)",cm)
    print("Confusion matrix (stdev of cnts)", cm_std)

    #Measuring the true positive and negative rates, 
    #since the false positive/negative rates are always 1 minus these, 
    #they are not printed and have the same standard deviation
    for i in range(len(y_probs)):
        pred_i = y_probs[i] > threshold
        tp_rate[i] = np.sum(np.logical_and(y_test==1, pred_i==1)) / np.sum(pred_i==1)
        tn_rate[i] = np.sum(np.logical_and(y_test==0, pred_i==0)) / np.sum(pred_i == 0)
    print("True Positive (rate): {0:0.4f} ({1:0.4f})".format(tp_rate.mean(), tp_rate.std()))
    print("True Negative (rate): {0:0.4f} ({1:0.4f})".format(tn_rate.mean(), tn_rate.std()))
    
 def vertical_averaging_help(xs, ys, xlen=101):
        """
            Interpolate the ROC curves to the same grid on x-axis
        """
        numnets = len(xs)
        xvals = np.linspace(0,1,xlen)
        yinterp = np.zeros((len(ys),len(xvals)))
        for i in range(numnets):
            yinterp[i,:] = np.interp(xvals, xs[i], ys[i])
        return xvals, yinterp
    fprs, tprs = [], []
    for i in range(len(y_probs)):
        fpr, tpr, _ = metrics.roc_curve(y_test, y_probs[i], pos_label=1)
        fprs.append(fpr)
        tprs.append(tpr)
    new_fprs, new_tprs = vertical_averaging_help(fprs, tprs)

    # measure Area Under Curve (AUC)
    y_prob_mean = y_probs.mean(0)
    auc = metrics.roc_auc_score(y_test, y_prob_mean)
    try:
        auc = metrics.roc_auc_score(y_test, y_prob_mean)
        print()
        print("AUC:", auc)
    except Exception as err:
        print(err)
        auc = np.nan
        
    #Take the percentiles for of the ROC curves at each point
    new_tpr_mean, new_tpr_5, new_tpr_95 = new_tprs.mean(0), np.percentile(new_tprs, 95, 0), np.percentile(new_tprs, 5, 0)
    # plot ROC curve
    plt.figure(figsize=[12,8])
    lw = 2
    plt.plot(new_fprs, new_tpr_mean, color='darkorange',
                lw=lw, label='ROC curve (area = %0.4f)' % metrics.auc(new_fprs, new_tpr_mean))
    if len(y_probs) > 1:
        plt.plot(new_fprs, new_tpr_95, color='yellow',
                    lw=lw, label='ROC curve 5%s (area = %0.4f)' % ("%", metrics.auc(new_fprs, new_tpr_95)))
        plt.plot(new_fprs, new_tpr_5, color='yellow',
                    lw=lw, label='ROC curve 95%s (area = %0.4f)' % ("%", metrics.auc(new_fprs, new_tpr_5)))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right", fontsize=16)
    plt.grid()
    plt.show()

    #If log flag is set, plot also the log of the ROC curves within some reasonable range
    if log_roc:
        # plot ROC curve
        plt.figure(figsize=[12,8])
        lw = 2
        plt.plot(np.log(new_fprs), np.log(new_tpr_mean), color='darkorange',
                    lw=lw, label='ROC curve (area = %0.4f)' % metrics.auc(new_fprs, new_tpr_mean))
        if len(y_probs) > 1:
            plt.plot(np.log(new_fprs), np.log(new_tpr_95), color='yellow',
                        lw=lw, label='ROC curve 5%s (area = %0.4f)' % ("%", metrics.auc(new_fprs, new_tpr_95)))
            plt.plot(np.log(new_fprs), np.log(new_tpr_5), color='yellow',
                        lw=lw, label='ROC curve 95%s (area = %0.4f)' % ("%", metrics.auc(new_fprs, new_tpr_5)))
        #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([-5, -3])
        plt.ylim([-1, 0.2])
        plt.xlabel('Log False Positive Rate', fontsize=16)
        plt.ylabel('Log True Positive Rate', fontsize=16)
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right", fontsize=16)
        plt.grid()
        plt.show()
    return (auc,maxf1) if show_f1 else auc, (tp_rate.mean(),tn_rate.mean()), new_fprs, new_tprs


def get_F1(y_pred, y, xlim=None, method='cnn', mass_thresh='5e13'):
    if xlim is None:
        xlim = (0, 0.997)
    col = f'score_wdust (trained>{mass_thresh})' if method == 'cnn' else 'mf_peaksig'
    Fscore = lambda x: _get_Fbeta(y, (y_pred > x).astype(int))

    f = plt.figure(figsize=(12, 4))
    x = np.linspace(xlim[0], xlim[1])
    y = np.asarray([Fscore(xx) for xx in x])
    plt.scatter(x, y)
    plt.xlim(xlim)
    plt.xlabel('%s thres' % ("CNN prob" if method == 'cnn' else "MF S/N"), fontsize='xx-large')
    plt.ylabel('F1_score', fontsize='xx-large')

    plt.show(block=True)

    return x[np.argmax(y)], np.max(y)

def get_F1_CNN_and_MF(vdf, col_cnn='score_wdust (trained>%s)', col_mf ='mf_peaksig', col_label='Truth(>%s)', mass_thresh='5e13'):
    import itertools
    if mass_thresh == '5e13':
        cnn_range = (0, 0.997)
        mf_range = (3, 15)
    else:
        cnn_range = (0.4, 0.997)
        mf_range = (3, 40)
    cnn_range = np.linspace(cnn_range[0], cnn_range[1])
    mf_range = np.linspace(mf_range[0], mf_range[1])
    #criteria = itertools.product(cnn_range, mf_range)
    criteria = [(c,m) for c in cnn_range for m in mf_range]
    col_cnn, col_label = col_cnn%mass_thresh, col_label%mass_thresh
    Fscore = lambda cc, mc: _get_Fbeta(vdf[col_label], (vdf[col_cnn]> cc).astype(int) * (vdf[col_mf]> mc).astype(int))
    cnn_x = np.asarray([c[0] for c in criteria])
    mf_y = np.asarray([c[1] for c in criteria])
    vals = np.asarray([Fscore(cc,mc) for cc,mc in criteria])

    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(cnn_x, mf_y, c=vals, cmap=cm)
    plt.colorbar(sc)
    plt.xlabel("CNN_threshold")
    plt.ylabel("MF_threshold")
    return criteria[np.argmax(vals)], np.max(vals)


''' https://github.com/deepskies/HEPEdgeCloud/blob/master/Azure_ResNet50/utils.py  '''
############ PLOTTING FUNCTIONS ###############
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function modified to plots the ConfusionMatrix object.
    Normalization can be applied by setting `normalize=True`.
    
    Code Reference : 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This script is derived from PyCM repository: https://github.com/sepandhaghighi/pycm
    
    """
    import matplotlib.pyplot as plt
    import itertools
    import numpy as np
    
    plt_cm = []
    for i in cm.classes:
        row=[]
        for j in cm.classes:
            row.append(cm.table[i][j])
        plt_cm.append(row)
    plt_cm = np.array(plt_cm)
    if normalize:
        plt_cm = plt_cm.astype('float') / plt_cm.sum(axis=1)[:, np.newaxis]     
    plt.imshow(plt_cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(cm.classes))
    plt.xticks(tick_marks, cm.classes, rotation=45)
    plt.yticks(tick_marks, cm.classes)

    fmt = '.2f' if normalize else 'd'
    thresh = plt_cm.max() / 2.
    for i, j in itertools.product(range(plt_cm.shape[0]), range(plt_cm.shape[1])):
        plt.text(j, i, format(plt_cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if plt_cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual', fontsize=16)
    plt.xlabel('Predict', fontsize=16)

def plot_acc_loss(figsize, num_plots, results, subplot_title_list, filename):
    """
    """
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    import numpy as np
    data_sizes = list(results.keys())

    format_plot()
    plt.figure(figsize=figsize)
    plt.subplots_adjust(wspace=0.35)
    y_axis = ["Loss", "Accuracy", "Auc"]
    colors=['forestgreen', 'royalblue', 'crimson', 'darkorchid']
    
    color=iter(cm.tab10(np.linspace(0,1,len(results))))
    j=0

    for keys, values in results.items():
        c=next(color)

        #grabbing accuracy, loss, and auc info
        for i in range(num_plots):
            plt.subplot(1, 3, i+1)
            # plotting training and validation per epoch for either accuracy, loss, or auc data
            # dashed line = validation info
            # solid line = training data
            plt.plot(values[i],'-', color='seagreen', label='Training') #label=str(keys), color=colors[j])
            plt.plot(values[i+3], '--', color='blue', label='Validation') #color=colors[j], label='Validation')
            plt.title(subplot_title_list[i], fontsize=26)
            plt.xlabel("Epoch", fontsize=20)
            plt.ylabel(y_axis[i], fontsize=20)
            plt.xticks(fontsize=16); plt.yticks(fontsize=16)
            plt.legend()
        j+=1

    #plt.colorbar(data_sizes, axis=1, cmap=color, wspace=5)
    plt.savefig(filename, bbox_inches='tight', transparent=True)
    plt.show()
    
def plot_sample_img(data, labels, figsize, filename="Image_Sample.png", show=True):
    """
    Plots data where each row of the plot consists of the same image in different channels (bands)
    Input:
    - data: array, an array of shape [batch_size, channels, height, width] OR [batch_size, height, width, channels]
    - labels: array, a 1D array of labels that match to the corresponding data
    - figsize: tuple, the figure size of the main plot
    - filename: string, saved filename
    - show: boolean, whether you want to plt.show() your figure or just save it to your computer  
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=figsize)

    counter = 1
    num_imgs = len(data)

    # if the image data is in the format [batch_size, channels, height, width]
    if (data.shape)[1] < (data.shape)[3]:
        num_bands = (data.shape)[1]
    # if the image data is in the format [batch_size, height, width, channels]
    else:
        num_bands = (data.shape)[3]

    for i in range(len(data)):
        for j in range(num_bands):
            #format plot horizontally
            if num_bands == 1:
                plt.subplot(num_bands, num_imgs, counter)
            #format plot vertically
            else:
                plt.subplot(num_imgs, num_bands, counter)                
            # if data format in shape [batch_size, channels, height, width]
            if (data.shape)[1] < (data.shape)[3]:
                plt.imshow(data[i][j], cmap='gray')
            # if data format in shape [batch_size, height, width, channels]
            else:
                plt.imshow(data[i, :, :, j], cmap='gray')
                
            plt.title("Label: "+ str(labels[i]), fontsize=14)
            counter += 1

    plt.subplots_adjust(wspace=.35, hspace=.35)
    plt.savefig(filename)
    if (show): plt.show()    
    

    
    ''' https://github.com/deepskies/deepmerge/blob/master/DeepMerge.ipynb '''
    
    

# plot histogram
bins = 50
plt.hist(non, bins, alpha=0.9, label='non-mergers', color='red')
plt.hist(past, bins, alpha=1, label='past mergers', color='deepskyblue')
plt.hist(future, bins, alpha=1, label='future mergers', color='navy')
plt.legend(loc='upper center')
plt.xticks(np.arange(0.1, 1, step=0.1))
plt.yticks(np.arange(0, 310, step=50))
plt.xlabel("CNN Output")
plt.show()




# Plot 2D histogram of the distribution of all future mergers vs TP future mergers 
#(stellar mass and the output probability)
sns.set_style("white")
plt.ylabel('CNN Output')
plt.xlabel('Stellar Mass')
plt.xlim(9.4, 11.8)
plt.xticks([9.5, 10, 10.5, 11, 11.5])
sns.kdeplot(logM_TP_future, prob_TP_future[:,0], cmap="RdGy",  n_levels=10)
sns.kdeplot(logM_future, future[:,0], cmap="coolwarm", n_levels=10)

r = sns.color_palette("RdGy")[0]
b = sns.color_palette("coolwarm")[0]

red_patch = mpatches.Patch(color=r, label='TP future mergers')
blue_patch = mpatches.Patch(color=b, label='all future mergers')
plt.legend(handles=[red_patch,blue_patch],loc='lower right')
plt.show()
