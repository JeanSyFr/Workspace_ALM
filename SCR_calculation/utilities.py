'''
General utility functions
'''

import os
import gzip
import pandas
import numpy
from scipy.sparse import issparse
from sklearn.utils.extmath import row_norms, safe_sparse_dot
from sklearn.utils import check_X_y
from scipy import stats
import dill


def save_file(the_object, filename, protocol=-1, do_zip=False):
    """
        Save an object to a file
    """

    # If the directory doesn't exist, create it
    directory = os.path.dirname(filename)
    if directory != '' and not os.path.exists(directory):
        print('Creating a new directory', directory)
        os.makedirs(directory)

    if do_zip:
        try:
            file_handle = gzip.GzipFile(filename, 'wb')
        except IOError:
            print('Error opening the file', filename)
            raise
    else:
        try:
            file_handle = open(filename, 'wb')
        except IOError:
            print('Error opening the file', filename)
            raise

    dill.dump(the_object, file_handle, protocol)
    file_handle.close()


def append_file(the_object, filename, protocol=-1, do_zip=False):
    """
       Save an object to file, appending
    """

    if do_zip:
        try:
            file_handle = gzip.GzipFile(filename, 'ab')
        except IOError:
            print('Error opening the file', filename)
            raise
    else:
        try:
            file_handle = open(filename, 'wb')
        except IOError:
            print('Error opening the file', filename)
            raise

    dill.dump(the_object, file_handle, protocol)
    file_handle.close()


def load_file(filename, do_zip=False):
    """
        Loads a file from disk
    """

    if do_zip:
        try:
            file_handle = gzip.GzipFile(filename, 'rb')
        except IOError:
            print('Error opening the file', filename)
            raise
    else:
        try:
            file_handle = open(filename, 'rb')
            # with open(filename, "rb") as fh:
            #         the_object = pickle.load(fh)
                    
        except IOError:
            print('Error opening the file', filename)
            raise
    the_object = dill.load(file_handle)
    file_handle.close()

    return the_object


def check_shape_vs_index(df):
    '''
        Given a dataframe with a MultiIndex, make sure that the actual size matches with the
        size of the index. This can fail if there are values in the index which are unused,
        which can be confusing, so we check and eliminate this condition.
    '''

    assert isinstance(df, pandas.DataFrame) or isinstance(df, pandas.Series)
    assert isinstance(df.index, pandas.MultiIndex)

    # Check that the indices match the actual length
    len_from_shape = df.shape[0]
    level_shape = [len(level) for level in df.index.levels]
    len_from_index = numpy.prod(level_shape)
    assert len_from_shape == len_from_index, "size from shape does not match size from index. Sizes: %s vs. %s, Shapes: %s vs. %s" % (
        str(len_from_shape), str(len_from_index), str(df.shape), str(level_shape))


def merge_multi_dfs(df1, df2, how='left', on=None, set_index=['Strategy','Scenario','Timestep']):
    '''
        Merge two pandas DataFrames with MultiIndex. If on is set to None,
        tries to automatically determine which indices to merge on.

        Arguments:
            df1: first dataframe (pandas.DataFrame)

            df2: second dataframe (pandas.DataFrame)

            how: how to merge, see pandas.merge

            on: which indices to merge on or None for automatic, also see pandas.merge

        Returns:
            merged dataframe (pandas.DataFrame)
    '''


    assert isinstance(df1, pandas.DataFrame)
    assert isinstance(df2, pandas.DataFrame)
    assert isinstance(df1.index, pandas.MultiIndex)
    assert isinstance(df2.index, pandas.MultiIndex)

    if on is None:
        # Try to figure out the indices to merge on automatically
        if not set(df2.index.names).issubset(df1.index.names):
            raise ValueError('Cannot automatically determine which indices to merge on')

        on = df2.index.names

    return pandas.merge(df1.reset_index(), df2.reset_index(), on=on, how=how).set_index(set_index)


'''
The code for the function f_regression() below is based on
sklearn.feature_selection.f_regression which was under the BSD 3-clause license,
which is a permissive license that allows commercial use and distribution.
See the text of the license below.

Copyright (c) 2017, 1QBit
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of [project] nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''


def f_regression(X, y, center=True):
    """Univariate linear regression tests.
    Quick linear model for testing the effect of a single regressor,
    sequentially for many regressors.
    This is done in 2 steps:
    1. The cross correlation between each regressor and the target is computed,
       that is, ((X[:, i] - mean(X[:, i])) * (y - mean_y)) / (std(X[:, i]) *
       std(y)).
    2. It is converted to an F score then to a p-value.
    Read more in the :ref:`User Guide <univariate_feature_selection>`.
    Parameters
    ----------
    X : {array-like, sparse matrix}  shape = (n_samples, n_features)
        The set of regressors that will be tested sequentially.
    y : array of shape(n_samples).
        The data matrix
    center : True, bool,
        If true, X and y will be centered.
    Returns
    -------
    F : array, shape=(n_features,)
        F values of features.
    pval : array, shape=(n_features,)
        p-values of F-scores.
    See also
    --------
    f_classif: ANOVA F-value between label/feature for classification tasks.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    """
    if issparse(X) and center:
        raise ValueError("center=True only allowed for dense data")
    X, y = check_X_y(X, y, ['csr', 'csc', 'coo'], dtype=numpy.float64)

    if center:
        y = y - numpy.mean(y)
        X = X.copy('F')  # faster in fortran
        X -= X.mean(axis=0)

    # compute the correlation
    corr = safe_sparse_dot(y, X)
    normX = row_norms(X.T)
    normY = numpy.linalg.norm(y)
    for id in range(len(corr)):
        if (normX[id] != 0):
            corr[id] /= normX[id]
    if normY != 0:
        corr /= normY

    # Convert to p-value
    degrees_of_freedom = y.size - (2 if center else 1)
    F = corr ** 2 / (1 - corr ** 2) * degrees_of_freedom
    pv = stats.f.sf(F, 1, degrees_of_freedom)

    return F, pv
