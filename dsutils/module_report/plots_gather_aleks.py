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
