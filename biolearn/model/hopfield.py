import numpy as np

from biolearn.model._base import Base
from biolearn.utils.weights import Normal
from biolearn.utils.activations import Linear
from biolearn.utils.optimizer import SGD

__author__  = ['Nico Curti', 'SimoneGasperini']
__email__ = ['nico.curit2@unibo.it', 'simone.gasperini2@studio.unibo.it']


class Hopfield (Base):

  '''
  Parameters
  ----------
    inputs : int (default=None)
      Number of input units

    outputs : int (default=100)
      Number of hidden units

    num_epochs : int (default=100)
      Maximum number of epochs for model convergency

    batch_size : int (default=10)
      Size of the minibatch

    weights_init : BaseWeights (default=Normal)
      Weights initialization strategy object

    optimizer : Optimizer (default=SGD)
      Optimizer object

    delta : float (default=0.4)
      Strength of the anti-hebbian learning

    p : float (default=2.)
      Lebesgue norm of the weights

    k : int (default=2)
      Ranking parameter, must be integer that is bigger or equal than 2

    precision : float (default=1e-30)
      Parameter that controls numerical precision of the weight updates

    epochs_for_convergency : int (default=None)
      Number of stable epochs requested for the convergency.
      If None the training proceeds up to the maximum number of epochs (num_epochs)

    convergency_atol : float (default=0.01)
      Absolute tolerance requested for the convergency

    random_state : int (default=None)
      Random seed for weights generation

    verbose : bool (default=True)
      Turn on/off the verbosity
  '''

  def __init__(self, inputs=None, outputs=100, num_epochs=100, batch_size=100,
      weights_init=Normal(), optimizer=SGD(), delta=.4, p=2., k=2,
      precision=1e-30, epochs_for_convergency=None, convergency_atol=0.01,
      random_state=None, verbose=True):

    self.delta = delta
    self.p = p
    self.k = k

    super (Hopfield, self).__init__(inputs=inputs, outputs=outputs, num_epochs=num_epochs, batch_size=batch_size,
                                    weights_init=weights_init, activation=Linear(), optimizer=optimizer,
                                    precision=precision, epochs_for_convergency=epochs_for_convergency,
                                    convergency_atol=convergency_atol, random_state=random_state, verbose=verbose)

  def _weights_update (self, X, output):
    '''
    This is the core function of the Hopfield model since it implements the
    Hopfield learning rule using the approximation introduced by Krotov:
    instead of solving the dynamical equations, the currents are used as a
    proxy for ranking the outputs activities and then computing the weights
    update.

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

      theta : array-like (1D)
        Array of learning progress
    '''

    order = np.argsort(output, axis=0)
    yl = np.zeros_like(output, dtype=float)
    yl[order[-1, :], range(self.batch_size)] = 1.
    yl[order[-self.k, :], range(self.batch_size)] = - self.delta

    xx = np.sum(yl * output, axis=1, keepdims=True)
    #ds = yl @ X - xx * self.weights
    ds = np.einsum('ij, jk -> ik', yl, X, optimize=True) - xx * self.weights

    nc = np.max(np.abs(ds))
    nc = 1. / max(nc, self.precision)

    return ds * nc, xx

  def _fit (self, X):
    '''
    Core function for the fit member
    '''

    return super(Hopfield, self)._fit(X=X, norm=True)

  def _predict (self, X):
    '''
    Core function for the predict member
    '''

    # return self.weights @ X
    return np.einsum('ij, kj -> ik', self.weights, X, optimize=True)
