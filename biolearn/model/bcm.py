import numpy as np

from biolearn.model._base import Base
from biolearn.utils.optimizer import SGD
from biolearn.utils.weights import Normal

__author__  = ['Nico Curti', 'SimoneGasperini']
__email__ = ['nico.curit2@unibo.it', 'simone.gasperini2@studio.unibo.it']


class BCM (Base):

  '''
  The idea of BCM theory is that for a random sequence of input patterns a synapse
  is learning to differentiate between those stimuli that excite the postsynaptic
  neuron strongly and those stimuli that excite that neuron weakly.
  Learned BCM feature detectors cannot, however, be simply used as the lowest layer
  of a feedforward network so that the entire network is competitive to a network of
  the same size trained with backpropagation algorithm end-to-end.

  Parameters
  ----------
    outputs : int (default=100)
      Number of hidden units

    num_epochs : int (default=100)
      Maximum number of epochs for model convergency

    batch_size : int (default=10)
      Size of the minibatch

    activation : string or Activations object (default='Logistic')
      Activation function to apply

    optimizer : Optimizer (default=SGD)
      Optimizer object (derived by the base class Optimizer)

    weights_init : BaseWeights object (default='Normal')
      Weights initialization strategy.

    orthogonalization : bool (default=False)
      Turn on/off the synaptic weights orthogonalization algorithm

    interaction_strength : float (default=0.)
      Set the lateral interaction strenght between weights

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
  '''

  def __init__(self, outputs=100, num_epochs=100,
      batch_size=100, activation='Logistic',
      optimizer=SGD(learning_rate=2e-2),
      weights_init=Normal(mu=0., std=1.),
      orthogonalization=False,
      interaction_strength=0.,
      precision=1e-30,
      epochs_for_convergency=None,
      convergency_atol=0.01,
      random_state=None, verbose=True):

    self.orthogonalization = orthogonalization
    self._interaction_matrix = self._weights_interaction(interaction_strength, outputs)
    self.interaction_strength = interaction_strength

    super (BCM, self).__init__(outputs=outputs, num_epochs=num_epochs,
                               batch_size=batch_size, activation=activation,
                               optimizer=optimizer,
                               weights_init=weights_init,
                               precision=precision,
                               epochs_for_convergency=epochs_for_convergency,
                               convergency_atol=convergency_atol,
                               random_state=random_state, verbose=verbose)

  def _weights_interaction (self, strength, outputs):
    '''
    Set the interaction matrix between weights' connections

    Parameters
    ----------
      strength : float
        Interaction strength between weights

      outputs : int
        Number of hidden units

    Returns
    -------
      interaction_matrix : array-like
        Matrix of interactions between weights
    '''

    if strength != 0.:
      L = np.full(fill_value=-strength, shape=(outputs, outputs))
      L[np.eye(*L.shape, dtype=bool)] = 1

      return np.linalg.inv(L)

    else:
      return np.eye(M=outputs, N=outputs)

  def _weights_update (self, X, output):
    '''
    Compute the weights update using the BCM learning rule.

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

    theta = np.mean(output**2, axis=1, keepdims=True)
    phi = output * (output - theta) * (1. / (theta + self.precision))

    #dw = phi @ X
    dw = np.einsum('ij, jk -> ik', phi, X, optimize=True)

    nc = np.max(np.abs(dw))
    nc = 1. / max(nc, self.precision)

    return dw * nc, theta

  def _fit (self, X):
    '''
    Core function for the fit member
    '''

    ortho = True if self.orthogonalization else False

    return super(BCM, self)._fit(X=X, norm=False, ortho=ortho)

  def _predict (self, X):
    '''
    Core function for the predict member
    '''

    # return self.activation.activation( self._interaction_matrix @ self.weights @ X.T, copy=True)
    return self.activation.activate(np.einsum('ij, jk, lk -> il', self._interaction_matrix, self.weights, X, optimize=True), copy=True)
