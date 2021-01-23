import numpy as np
from tqdm import tqdm
from collections import deque

from biolearn.utils.misc import _check_activation
from biolearn.utils.weights import BaseWeights
from biolearn.utils.activations import Activations
from biolearn.utils.optimizer import Optimizer

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

__author__  = ['Nico Curti', 'SimoneGasperini']
__email__ = ['nico.curit2@unibo.it', 'simone.gasperini2@studio.unibo.it']


class Base (BaseEstimator, TransformerMixin):

  '''
  Parameters
  ----------
    inputs : int (default=None)
      Number of input units

    outputs : int (default=100)
      Number of hidden units

    num_epochs : int (default=100)
      Maximum number of epochs for model convergency

    batch_size : int (default=100)
      Size of the minibatch

    weights_init : BaseWeights (default=BaseWeights)
      Weights initialization strategy object

    activation : Activations (default=Activations)
      Activation function object

    optimizer : Optimizer (default=Optimizer)
      Optimizer object

    precision : float (default=1e-30)
      Parameter that controls numerical precision of the weight updates

    epochs_for_convergency : int (default=None)
      Number of stable epochs requested for the convergency.
      If None the training proceeds up to the maximum number of epochs (num_epochs)

    convergency_atol : float (default=0.01)
      Absolute tolerance requested for the convergency

    random_state : int (default=None)
      Random seed for batch subdivisions

    verbose : bool (default=True)
      Turn on/off the verbosity
  '''

  def __init__ (self, inputs=None, outputs=100, num_epochs=100, batch_size=100,
      weights_init=BaseWeights, activation=Activations, optimizer=Optimizer,
      precision=1e-30, epochs_for_convergency=None, convergency_atol=0.01,
      random_state=None, verbose=True):

    _, activation = _check_activation(self, activation)

    self.inputs = inputs
    self.outputs = outputs
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.weights_init = weights_init
    self.activation = activation
    self.optimizer = optimizer
    self.precision = precision
    self.epochs_for_convergency = epochs_for_convergency if epochs_for_convergency is not None else num_epochs
    self.epochs_for_convergency = max(self.epochs_for_convergency, 1)
    self.convergency_atol = convergency_atol
    self.random_state = random_state
    self.verbose = verbose

    if inputs is not None:
      #self.weights = np.random.normal(loc=self.mu, scale=self.sigma, size=(outputs, inputs))
      self.weights = self.weights_init.get(size=(outputs, inputs))

  def _weights_update (self, X, output):
    '''
    Compute the weights update using the given learning rule.

    Parameters
    ----------
      X : array-like (2D)
        Input array of data

      output : array-like (2D)
        Output of the model estimated by the predict function

    Returns
    -------
      weight_update : array-like (2D)
        Weight updates matrix to apply

      theta : array-like
        Array of learning progress
    '''

    raise NotImplementedError

  def _weights_orthogonalization (self):
    '''
    Apply the orthogonalization algorithm to the weights.
    '''

    norms = np.linalg.norm(self.weights, axis=1)

    U, _, Vt = np.linalg.svd(self.weights, full_matrices=False)

    #self.weights = np.diag(norms) @ U @ Vt
    self.weights = np.einsum('ij, jk, kl -> il', np.diag(norms), U, Vt, optimize=True)

  def _lebesque_norm (self):
    '''
    Apply the Lebesgue norm to the weights.
    '''

    if self.p != 2:
      sign = np.sign(self.weights)
      self.weights = sign * np.absolute(self.weights)**(self.p - 1)

  def _fit_step (self, X, norm=False):
    '''
    Core function of fit step (forward + backward + updates).
    We divide the step into a function to allow an easier visualization
    of the weight matrix (if necessary).

    Parameters
    ----------
      X : array-like (2D)
        Input array of data

      norm : bool (default=False)
        Switch on/off the weights normalization using Lebesque norm

    Returns
    -------
      theta : array-like
        Array of learning progress
    '''

    if norm:
      self._lebesque_norm()

    # predict the encoded values
    output = self._predict(X)

    # update weights
    w_update, theta = self._weights_update(X, output)

    #self.weights[:] += epsilon * w_update
    self.weights, = self.optimizer.update(params=[self.weights], gradients=[-w_update]) # -update for compatibility with optimizers

    return theta

  @property
  def _check_convergency (self):
    '''
    Check if the current training has reached the convergency.
    The convergency is estimated by the stability or not of the
    learning parameter in a fixed number of epochs for all the outputs.

    Returns
    -------
      check : bool
        Check if the learning history of the model is stable.
    '''

    if len(self.history) < self.epochs_for_convergency:
      return False

    last = np.full_like(self.history, fill_value=self.history[-1])

    return np.allclose(self.history, last, atol=self.convergency_atol)#, rtol=self.convergency_atol)

  def _fit (self, X, norm=False, ortho=False):
    '''
    Core function for the fit member

    Parameters
    ----------
      X : array-like (2D)
        Input array of data

      norm : bool (default=False)
        Switch on/off the weights normalization using Lebesque norm

      ortho : bool (default=False)
        Switch on/off the synaptic weights orthogonalization algorithm

    Returns
    -------
      self
    '''

    num_samples, _ = X.shape
    indices = np.arange(0, num_samples).astype('int64')
    num_batches = num_samples // self.batch_size

    for epoch in range(self.num_epochs):

      if self.verbose:
        print('Epoch {:d}/{:d}'.format(epoch + 1, self.num_epochs))

      previous_weights = np.copy(self.weights)

      # random shuffle the input
      np.random.shuffle(indices)

      batches = np.lib.stride_tricks.as_strided(indices, shape=(num_batches, self.batch_size), strides=(self.batch_size * 8, 8))

      for batch in tqdm(batches, disable=(not self.verbose)):

        batch_data = X[batch, ...]

        theta = self._fit_step(X=batch_data, norm=norm)

      if ortho:
        self._weights_orthogonalization()

      self.history.append(theta)
      self._thetas.append(theta[:,0])
      self._weights_meandiff.append(np.mean(np.abs(self.weights - previous_weights)))

      # check if the model has reached the convergency (early stopping criteria)
      if self._check_convergency:
        if self.verbose:
          print('Early stopping: the training has reached the convergency criteria')
        break

    return self

  def fit (self, X, y=None):
    '''
    Fit the biolearn model weights.

    Parameters
    ----------
      X : array-like of shape (n_samples, n_features)
        The training input samples

      y : array-like, default=None
        The array of labels

    Returns
    -------
      self : object
        Return self
    '''

    X = check_array(X)
    np.random.seed(self.random_state)
    num_samples, num_features = X.shape

    if self.inputs is not None and self.inputs != num_features:
      raise ValueError('Incorrect number of data features. It must match the number of inputs units. '
                       'Given {:d} for {:d} inputs'.format(num_features, self.inputs))

    if self.batch_size > num_samples:
      raise ValueError('Incorrect batch_size. It must be less or equal to the number of samples. '
                       'Given {:d} for {:d} samples'.format(self.batch_size, num_samples))

    if self.inputs is None:
      self.inputs = num_features
      #self.weights = np.random.normal(loc=self.mu, scale=self.sigma, size=(self.outputs, self.inputs))
      self.weights = self.weights_init.get(size=(self.outputs, self.inputs))

    self.history = deque(maxlen=self.epochs_for_convergency)
    self._thetas = []
    self._weights_meandiff = []

    self._fit(X)

    return self

  def _predict (self, X):
    '''
    Core function for the predict member
    '''

    raise NotImplementedError

  def predict (self, X, y=None):
    '''
    Reduce X applying the biolearn encoding.

    Parameters
    ----------
      X : array of shape (n_samples, n_features)
        The input samples

      y : array-like, default=None
        The array of labels

    Returns
    -------
      Xnew : array of shape (n_values, n_samples)
        The encoded features
    '''

    check_is_fitted(self, 'weights')
    X = check_array(X)

    return self._predict(X)

  def transform (self, X):
    '''
    Apply the data reduction according to the features in the best signature found.

    Parameters
    ----------
      X : array-like of shape (n_samples, n_features)
        The input samples

    Returns
    -------
      Xnew : array-like of shape (n_samples, encoded_features)
        The data encoded according to the model weights.
    '''

    check_is_fitted(self, 'weights')
    Xnew = self._predict(X)

    return Xnew.transpose()

  def fit_transform (self, X, y=None):
    '''
    Fit the model model meta-transformer and apply the data encoding transformation.

    Parameters
    ----------
      X : array-like of shape (n_samples, n_features)
        The training input samples

      y : array-like, shape (n_samples,)
        The target values

    Returns
    -------
      Xnew : array-like of shape (n_samples, encoded_features)
          The data encoded according to the model weights.
    '''

    self.fit(X, y=y)
    Xnew = self.transform(X)

    return Xnew

  def save_weights (self, filename):
    '''
    Save the current weights to a binary file.

    Parameters
    ----------
      filename : str
        Filename or path

    Returns
    -------
      True if everything is ok
    '''

    check_is_fitted(self, 'weights')
    with open(filename, 'wb') as fp:
      self.weights.tofile(fp, sep='')

    return True

  def load_weights (self, filename):
    '''
    Load the weight matrix from a binary file.

    Parameters
    ----------
      filename : str
        Filename or path

    Returns
    -------
      self : object
        Return self
    '''

    with open(filename, 'rb') as fp:
      self.weights = np.fromfile(fp, dtype=np.float, count=-1)

    return self

  def __repr__(self):
    '''
    Object representation
    '''

    class_name = self.__class__.__qualname__
    params = self.__init__.__code__.co_varnames
    params = set(params) - {'self'}
    args = ', '.join(['{0}={1}'.format(k, str(getattr(self, k)))
                      if not isinstance(getattr(self, k), str) else '{0}="{1}"'.format(k, str(getattr(self, k)))
                      for k in params])

    return '{0}({1})'.format(class_name, args)
