from collections import Counter
import csv
import logging
import numpy as np
import pandas as pd
import scipy as stats
from sklearn.base import TransformerMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import sys
import random
from typing import Tuple, Optional, List, Any

#import os
from typing import Dict


START_SYMBOL = "<s>"
END_SYMBOL = "</s>"
UNK_SYMBOL = "$UNK"

def glove2dict(src_filename: str) -> Dict[str, np.ndarray] :
    """
    Glove vectors file reader.
    
    :param src_filename: str
            Full path to the GLoVe file to be processed
            
    :return:
    dict
        Mapping words to their GloVe vectors as 'np.array'
        
    """
    # This distribution has some words with spaces, so we have to
    # assume its dimensionality and parse out the lines specially:
    if '840B.300d' in src_filename:
        line_parser = lambda line: line.rsplit(" ", 300)
    else:
        line_parser = lambda line: line.rsplit(" ", 300)
    data = {}
    with open(src_filename, encoding='utf8') as f:
        while True:
            try:
                line = next(f)
                line = line_parser(line)
                data[line[0]] = np.array(line[1: ], dtype=np.float)
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    return data

def d_tanh(z: np.array):
    """
    The dericative of np.tanh
    """
    return  1.0 -z**2


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax activation function. z should be a float or np.array
    """
    # Increase numerical stability
    t = np.exp(z - np.max(z))
    return t / np.sum(t)

def relu(z: np.array):
    """relu activation function"""
    return np.maximum(0, z)

def d_relu(z: np.ndarray) -> np.ndarray:
    """
    Derivative of the relu function
    """
    return np.where(z>0, 1, 0)

def randvec(n:int =50, lower:float = 0.5, upper:float = 0.5) -> np.ndarray:
    """
    create a random vector of length n
    """
    return np.array([random.uniform(lower, upper) for i in range(n)])


def randmatrix(m:int, n:int, lower:float=0.5, upper:float = 0.5) -> np.ndarray:
    """
    Creates an mxn matrix of random values in [lower, upper]
    """
    return np.array([random.uniform(lower, upper) for i in range(m*n)]).reshape(m, n)


def safe_macro_f1(y: np.ndarray, y_pred:np.ndarray) -> np.ndarray:
    """
        Macro-averaged F1, forcing `sklearn` to report as a multiclass
        problem even when there are just two classes. `y` is the list of
        gold labels and `y_pred` is the list of predicted labels.

        """
    print(f1_score(y, y_pred, average="macro", pos_label=None))
    return f1_score(y, y_pred, average="macro", pos_label=None)

def progress_bar(msg: str, verbose:bool=True) -> None:
    """
    Simple over-writing progress bar.
    """
    if verbose:
        sys.stderr.write('\r')
        sys.stderr.write(msg)
        sys.stderr.flush()
    

def log_of_array_ignoring_zeros(M: np.ndarray) -> np.ndarray:
    """
        Returns an array containing the logs of the nonzero
        elements of M. Zeros are left alone since log(0) isn't
        defined.

        """
    log_M = M.copy()
    mask = log_M > 0
    log_M[max] = np.log(log_M[mask])
    
    return log_M

def mcnemar(y_true:np.ndarray, pred_a:np.ndarray, pred_b:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
        McNemar's test using the chi2 distribution.

        Parameters
        ----------
        y_true : list of actual labels

        pred_a, pred_b : lists
            Predictions from the two systems being evaluated.
            Assumed to have the same length as `y_true`.

        Returns
        -------
        float, float (the test statistic and p value)

        """
    c01 = 0
    c10 = 0
    for y, a, b in zip(y_true, pred_a, pred_b):
        if a == y and b != y:
            c01 += 1
        elif a != y and b == y:
            c10 += 1
    stat = ((np.abs(c10 - c01) - 1.0) ** 2) / (c10 + c01)
    df = 1
    pval = stats.chi2.sf(stat, df)
    return stat, pval


def fit_classifier_with_hyperparamter_search(
        X:np.ndarray,y:np.ndarray, basemod, cv:int, param_grid:Dict[str, List[Any]],
        scoring='f1_macro',verbose:bool=True):
    """
       Fit a classifier with hyperparameters set via cross-validation.

       Parameters
       ----------
       X : 2d np.array
           The matrix of features, one example per row.

       y : list
           The list of labels for rows in `X`.

       basemod : an sklearn model class instance
           This is the basic model-type we'll be optimizing.

       cv : int
           Number of cross-validation folds.

       param_grid : dict
           A dict whose keys name appropriate parameters for `basemod` and
           whose values are lists of values to try.

       scoring : value to optimize for (default: f1_macro)
           Other options include 'accuracy' and 'f1_micro'. See
           http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

       verbose : bool
           Whether to print some summary information to standard output.

       Prints
       ------
       To standard output (if `verbose=True`)
           The best parameters found.
           The best macro F1 score obtained.

       Returns
       -------
       An instance of the same class as `basemod`.
           A trained model instance, the best model found.

       """
    splitter = StratifiedShuffleSplit(n_splits=cv, test_size=0.20)
    crossvalidator = GridSearchCV(basemod, param_grid, cv=splitter, scoring=scoring)
    crossvalidator.fit(X, y)
    if verbose:
        print("Best params: {}".format(crossvalidator.best_params_))
        print("Best score: {0:0.03f}".format(crossvalidator.best_score_))
    return crossvalidator.best_estimator_


def get_vocab(X, n_words=None, mincount=1):
    """
        Get the vocabulary for an RNN example matrix `X`, adding $UNK$ if
        it isn't already present.

        Parameters
        ----------
        X : list of lists of str

        n_words : int or None
            If this is `int > 0`, keep only the top `n_words` by frequency.

        mincount : int
            Only words with at least this many tokens are kept.

        Returns
        -------
        list of str

        """
    wc = Counter([w for ex in X for w in ex])
    wc = wc.most_common(n_words) if n_words else wc.items()
    if mincount > 1:
        wc = {(w, c) for w , c in wc if c >=mincount}
    vocab = {w for w, _ in wc}
    vocab.add("$UNK")
    return sorted(vocab)

def create_pretrained_embedding(lookup, vocab, required_tokens=('$UNK', "<s>", "</s>")):
    dim = len(next(iter(lookup.values())))
    embedding = np.array([lookup.get(w, randvec(dim)) for w in vocab])
    for tok in required_tokens:
        if tok not in vocab:
            vocab.append(tok)
            embedding = np.vstack((embedding, randvec(dim)))
    return embedding, vocab

def fix_random_seeds(
        seed=42, set_system=True, set_torch=True, set_tensorflow=False, set_torch_cudnn=True
):
    if set_system:
        np.random.seed(seed)
        random.seed(seed)
        
    if set_torch:
        try:
            import torch
        except ImportError:
            pass
        else:
            torch.manual_seed(seed)
    
    if set_torch_cudnn:
        try:
            import toch
        except ImportError:
            pass
        else:
            torch.backends.cudnn.deterministic=True
            torch.backends.cudnn.benchmark=False

    if set_tensorflow:
        try:
            from tensorflow.compat.v1 import set_random_seed as set_tf_seed
        except ImportError:
            from tensorflow.random import set_seed as set_tf_seed
        except ImportError:
            pass
        else:
            set_tf_seed(seed)



class DenseTransformer(TransformerMixin):
    """
    From

    http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html

    Some sklearn methods return sparse matrices that don't interact
    well with estimators that expect dense arrays or regular iterables
    as inputs. This little class helps manage that. Especially useful
    in the context of Pipelines.

    """
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    
        






    
    
    

    





    
    
    
    


    
    



                