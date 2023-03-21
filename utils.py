#  +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+
#  |u| |t| |i| |l| |i| |t| |i| |e| |s|
#  +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+


def performance_report(cm, mode='macro', printing=False):
    """
    Generate a performance report of the model.

    Args:
    - cm (dictionary): Confusion matrix containing true and predicted labels for each class.
    - mode (str, optional): Type of average calculation to use. Can be 'macro' or 'weighted'. Default is 'macro'.
    - printing (bool, optional): Whether to print the classification report. Default is False.

    Returns:
    - list: List of precision, recall, F1-score and support for each class, calculated using the specified average mode.
    """
    col = len(cm)
    labels = list(cm.keys())

    # col=number of class
    arr = []
    for key, value in cm.items():
        arr.append(value)

    cr = dict()
    support_sum = 0

    # Calculate macro and weighted averages of precision, recall and F1-score.
    # Macro average is the unweighted mean of each class' metric.
    # Weighted average is the mean of each class' metric weighted by the number of instances in that class.
    macro = [0] * 3  
    weighted = [0] * 3
    for i in range(col):
        # Calculate precision, recall and F1-score for each class.
        vertical_sum = sum([arr[j][i] for j in range(col)])
        horizontal_sum = sum(arr[i])
        p = arr[i][i] / vertical_sum
        r = arr[i][i] / horizontal_sum
        f = (2 * p * r) / (p + r)
        s = horizontal_sum
        row = [p,r,f,s]

        # Add support to the sum.
        support_sum += s

        # Add macro and weighted averages.
        for j in range(3):
            macro[j] += row[j]
            weighted[j] += row[j]*s

        # Save metrics for each class to cr dictionary.
        cr[i] = row

    # Add accuracy parameters.
    truepos=0
    total=0
    for i in range(col):
        truepos += arr[i][i]
        total += sum(arr[i])
    cr['Accuracy'] = ["", "", truepos/total, support_sum]

    # Add macro-weight features.
    macro_avg = [Sum/col for Sum in macro]
    macro_avg.append(support_sum)
    cr['Macro_avg'] = macro_avg

    # Add weighted_avg
    weighted_avg = [Sum/support_sum for Sum in weighted]
    weighted_avg.append(support_sum)
    cr['Weighted_avg'] = weighted_avg

    # Print the classification report.
    if printing:
        stop=0
        max_key = max(len(str(x)) for x in list(cr.keys())) + 15
        print("Performance report of the model is :")
        print(f"%{max_key}s %9s %9s %9s %9s\n" % (" ", "Precision", "Recall", "F1-Score", "Support"))
        for i, (key, value) in enumerate(cr.items()):
            if stop<col:
                stop += 1
                print(f"%{max_key}s %9.2f %9.2f %9.2f %9d" % (labels[key] if isinstance(key, int) else key, value[0], value[1], value[2], value[3]))
            elif stop == col:
                stop += 1
                print(f"\n%{max_key}s %9s %9s %9.2f %9d" % (labels[key] if isinstance(key, int) else key, value[0], value[1], value[2], value[3]))
            else:
                print(f"%{max_key}s %9.2f %9.2f %9.2f %9d" % (labels[key] if isinstance(key, int) else key, value[0], value[1], value[2], value[3]))
    if mode == 'macro':
        return cr['Macro_avg']
    else:
        return cr['Weighted_avg']


def cm_to_dict(cm, labels):
    """
    Convert a confusion matrix in numpy array format to a dictionary.

    Args:
        cm (numpy.ndarray): Confusion matrix in numpy array format.
        labels (list): List of labels corresponding to the classes.

    Returns:
        dict: Dictionary where keys are the labels and values are the row of the confusion matrix corresponding to that label.

    """
    cm_dict = dict()
    for i, row in enumerate(cm):
        # The index i corresponds to the ith class label
        # The row of the confusion matrix corresponding to the label is added to the dictionary
        cm_dict[labels[i]] = row
    return cm_dict