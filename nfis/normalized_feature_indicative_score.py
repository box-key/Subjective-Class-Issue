from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import MaxAbsScaler

def _getChi2(X, y):
    # Map categorical values into numerical values
    lb = LabelBinarizer()
    Y = lb.fit_transform(y)
    # For Debugging
    for i in range(Y.shape[1]):
        print('Column', i, ':', Y[:,i].sum())
    print(lb.classes_)
    # Calculate Chi2 score for each class label
    observed = safe_sparse_dot(Y.T, X)
    feature_count = X.sum(axis=0).reshape(1, -1)
    class_prob = Y.mean(axis=0).reshape(1, -1)
    expected = np.dot(class_prob.T, feature_count)
    observed = np.asarray(observed, dtype=np.float64)
    chisq = observed
    chisq -= expected
    chisq **= 2
    # np.errstate determines how to handle an exception
    with np.errstate(invalid="ignore"):
        chisq /= expected
    return chisq

def getK_NFIS(X, y, k):
    # Obtain matrix of chi2 score
    # Each column corresponds to each class label
    chi2 = _getChi2(X, y).T
    sorted_chi2 = []
    for i in range(chi2.shape[1]):
        # Sort a column in descending order
        sorted = chi2[chi2[:,i].argsort()[::-1]]
        # Normalize chi2 scores and append K elements into array
        sorted_chi2.append(MaxAbsScaler().fit_transform(sorted[:,i][:k].reshape(k, 1)))
    return sorted_chi2

def plot_scores(score, row, col, bins):
    fig, axs = plt.subplots(row,col,sharey=True, tight_layout=True)
    cap = len(score)
    for r in range(row):
        for c in range(col):
            if r*col+c >= cap:
                break
            axs[r, c].hist(sorted_chi2[r*col+c], bins=bins)
            axs[r, c].set_xlim(0,1)
