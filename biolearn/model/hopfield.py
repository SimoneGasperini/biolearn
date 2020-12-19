import numpy as np

from biolearn.model._base import BasePlasticity
from biolearn.utils.optimizer import SGD
from biolearn.utils.weights import Normal

__author__  = ['Nico Curti', 'SimoneGasperini']
__email__ = ['nico.curit2@unibo.it', 'simone.gasperini2@studio.unibo.it']


class Hopfield (BasePlasticity):

  '''
  Hopfield and Krotov implementation of the BCM algorithm [1]_.

  Parameters
  ----------
    outputs : int (default=100)
      Number of hidden units

    num_epochs : int (default=100)
      Maximum number of epochs for model convergency

    batch_size : int (default=10)
      Size of the minibatch

    optimizer : Optimizer (default=SGD)
      Optimizer object (derived by the base class Optimizer)

    delta : float (default=0.4)
      Strength of the anti-hebbian learning

    weights_init : BaseWeights object (default="Normal")
      Weights initialization strategy.

    p : float (default=2.)
      Lebesgue norm of the weights

    k : int (default=2)
      Ranking parameter, must be integer that is bigger or equal than 2

    precision : float (default=1e-30)
      Parameter that controls numerical precision of the weight updates

    epochs_for_convergency : int (default=None)
      Number of stable epochs requested for the convergency.
      If None the training proceeds up to the maximum number of epochs (num_epochs).

    convergency_atol : float (default=0.01)
      Absolute tolerance requested for the convergency

    random_state : int (default=None)
      Random seed for weights generation

    verbose : bool (default=True)
      Turn on/off the verbosity

  Examples
  --------
  >>> from sklearn.datasets import fetch_openml
  >>> import pylab as plt
  >>> from plasticity.model import Hopfield
  >>>
  >>> X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)
  >>> X *= 1. / 255
  >>> model = Hopfield(outputs=100, num_epochs=10)
  >>> model.fit(X)
  Hopfield(batch_size=100, outputs=100, num_epochs=10, random_state=42, precision=1e-30)
  >>>
  >>> # view the memorized weights
  >>> w = model.weights[0].reshape(28, 28)
  >>> nc = np.max(np.abs(w))
  >>>
  >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
  >>> im = ax.imshow(w, cmap='bwr', vmin=-nc, vmax=nc)
  >>> fig.colorbar(im, ticks=[np.min(w), 0, np.max(w)])
  >>> ax.axis("off")
  >>> plt.show()

  .. image:: ../../../img/Hopfield_weights.gif

  References
  ----------
  .. [1] Dmitry Krotov, and John J. Hopfield. Unsupervised learning by competing hidden units,
         PNAS, 2019, www.pnas.org/cgi/doi/10.1073/pnas.1820458116
  '''

  def __init__(self, outputs=100, num_epochs=100,
      batch_size=100, delta=.4,
      optimizer=SGD(learning_rate=2e-2),
      weights_init=Normal(mu=0., std=1.),
      p=2., k=2,
      precision=1e-30,
      epochs_for_convergency=None,
      convergency_atol=0.01,
      random_state=None, verbose=True):

    self.delta = delta
    self.p = p
    self.k = k

    super (Hopfield, self).__init__(outputs=outputs, num_epochs=num_epochs,
                                    batch_size=batch_size, activation='Linear',
                                    optimizer=optimizer,
                                    weights_init=weights_init,
                                    precision=precision,
                                    epochs_for_convergency=epochs_for_convergency,
                                    convergency_atol=convergency_atol,
                                    random_state=random_state, verbose=verbose)

  def _weights_update (self, X, output):
    '''
    Approximation introduced by Krotov.
    Instead of solving dynamical equations we use the currents as a proxy
    for ranking of the final activities as suggested in [0].

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

    Notes
    -----
    .. note::
      This is the core function of the Hopfield class since it implements
      the Hopfield learning rule.
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
