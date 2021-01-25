from __future__ import print_function
from __future__ import division


from biolearn.utils.activations import *
from biolearn.utils.optimizer import *
from biolearn.utils.weights import *

import os
import numpy as np
from biolearn.model.bcm import BCM

from hypothesis import strategies as st
from hypothesis import given, settings, assume


__author__  = ['SimoneGasperini']
__email__   = ['simone.gasperini2@studio.unibo.it']


activations = [AsymmetricLogistic, Elliot, Elu, Hardtan,
               Leaky, Lhtan, Linear, Loggy, Logistic,
               Plse, Ramp, Relie, Relu,
               Selu, SoftPlus, SoftSign, Stair,
               SymmElliot, Tanh]

optimizers = [Adadelta, Adagrad, Adam, Adamax,
              Momentum, NesterovMomentum, RMSprop, SGD]

weights = [GlorotNormal, GlorotUniform, HeNormal, HeUniform,
           LecunUniform, Normal, Ones, Orthogonal,
           TruncatedNormal, Uniform, Zeros]



@given(inputs                 = st.integers(min_value=1, max_value=1000),
       outputs                = st.integers(min_value=1, max_value=1000),
       num_epochs             = st.integers(min_value=1),
       batch_size             = st.integers(min_value=1),
       weights_init           = st.sampled_from(weights),
       activation             = st.sampled_from(activations),
       optimizer              = st.sampled_from(optimizers),
       orthogonalization      = st.booleans(),
       interaction_strength   = st.floats(min_value=-1., max_value=1., exclude_min=True, exclude_max=True),
       precision              = st.floats(),
       epochs_for_convergency = st.integers(min_value=1),
       convergency_atol       = st.floats(),
       random_state           = st.integers(),
       verbose                = st.booleans()
       )
@settings(deadline=None)
def test_constructor (inputs, outputs, num_epochs, batch_size, weights_init, activation, optimizer,
                      orthogonalization, interaction_strength, precision, epochs_for_convergency,
                      convergency_atol, random_state, verbose):
  '''
  Test the BCM object constructor.
  The number of inputs and outputs are bounded in a range of reasonable values
  (also to prevent the test from being too slow).
  The interaction_strength parameter is bounded in the interval ]-1,1[
  (excluding the extremes because in those cases we would have a singular
  and not invertible interaction matrix).
  '''

  params = {'inputs'                 : inputs,
            'outputs'                : outputs,
            'num_epochs'             : num_epochs,
            'batch_size'             : batch_size,
            'weights_init'           : weights_init(),
            'activation'             : activation(),
            'optimizer'              : optimizer(),
            'orthogonalization'      : orthogonalization,
            'interaction_strength'   : interaction_strength,
            'precision'              : precision,
            'epochs_for_convergency' : epochs_for_convergency,
            'convergency_atol'       : convergency_atol,
            'random_state'           : random_state,
            'verbose'                : verbose
            }

  bcm = BCM(**params)
  assert params == bcm.get_params()



@given(samples                = st.integers(min_value=1, max_value=1000),
       inputs                 = st.integers(min_value=1, max_value=20),
       outputs                = st.integers(min_value=1, max_value=5),
       batch_size             = st.integers(min_value=1),
       weights_init           = st.sampled_from(weights),
       activation             = st.sampled_from(activations),
       optimizer              = st.sampled_from(optimizers),
       interaction_strength   = st.floats(min_value=-1., max_value=1., exclude_min=True, exclude_max=True),
       )
@settings(deadline=None)
def test_save_load_weights (samples, inputs, outputs, batch_size,
                            weights_init, activation, optimizer,
                            interaction_strength):
  '''
  Test the save and load weights methods.
  '''

  assume(batch_size <= samples)

  data = np.random.uniform(low=0., high=1., size=(samples,inputs)).astype(float)

  bcm = BCM(inputs=inputs, outputs=outputs, num_epochs=1, batch_size=batch_size,
            weights_init=weights_init(), activation=activation(), optimizer=optimizer(),
            interaction_strength=interaction_strength, verbose=False)

  bcm.fit(X=data)

  assert bcm.save_weights('weights.bin')
  bcm_new = bcm.load_weights('weights.bin')
  os.remove('weights.bin')

  assert np.all(bcm.weights == bcm_new.weights)



@given(samples                = st.integers(min_value=1, max_value=1000),
       inputs                 = st.integers(min_value=1, max_value=20),
       outputs                = st.integers(min_value=1, max_value=5),
       num_epochs             = st.integers(min_value=1, max_value=10),
       batch_size             = st.integers(min_value=1, max_value=1000),
       optimizer              = st.sampled_from(optimizers),
       )
@settings(deadline=None)
def test_fit_negative_weights (samples, inputs, outputs, num_epochs, batch_size, optimizer):
  '''
  Test the model stability in case of negative (or null) weights,
  and Relu activation function (so that f(x)=0 forall x <= 0).
  There are no lateral interactions at all between the neurons and the
  orthogonalization algorithm is disabled.
  '''

  assume(batch_size <= samples)

  data = np.random.uniform(low=0., high=1., size=(samples,inputs)).astype(float)

  bcm = BCM(inputs=inputs, outputs=outputs, num_epochs=num_epochs, batch_size=batch_size,
            weights_init=TruncatedNormal(mu=-2.,std=1.),
            activation=Relu(),
            optimizer=optimizer(),
            verbose=False)

  initial_weights = np.copy(bcm.weights)
  bcm.fit(X=data)
  final_weights = np.copy(bcm.weights)

  assert np.all(initial_weights == final_weights)



@given(samples                = st.integers(min_value=1, max_value=1000),
       inputs                 = st.integers(min_value=1, max_value=50),
       outputs                = st.integers(min_value=1, max_value=10),
       )
@settings(deadline=None)
def test_predict (samples, inputs, outputs):
  '''
  Test the predict method in case of null, positive, and negative input 
  random data and positive or negative weights.
  '''

  zeros = np.zeros(shape=(samples,inputs)).astype(float)
  pos = np.random.uniform(low=0., high=1., size=(samples,inputs)).astype(float)
  neg = np.random.uniform(low=-1., high=0., size=(samples,inputs)).astype(float)

  bcm = BCM(inputs=inputs, outputs=outputs,
                weights_init=TruncatedNormal(mu=2.,std=1.),
                activation=Linear(),
                verbose=False)

  Y1 = bcm.predict(X=zeros)
  assert np.all(Y1 == np.zeros(shape=(outputs,samples)))

  Y2 = bcm.predict(X=pos)
  assert np.all(Y2 >= 0.)

  Y3 = bcm.predict(X=neg)
  assert np.all(Y3 <= 0.)

  bcm.weights = - bcm.weights

  Y4 = bcm.predict(X=zeros)
  assert np.all(Y4 == np.zeros(shape=(outputs,samples)))

  Y5 = bcm.predict(X=pos)
  assert np.all(Y5 <= 0.)

  Y6 = bcm.predict(X=neg)
  assert np.all(Y6 >= 0.)



@given(samples                = st.integers(min_value=1, max_value=1000),
       inputs                 = st.integers(min_value=1, max_value=50),
       outputs                = st.integers(min_value=1, max_value=10),
       batch_size             = st.integers(min_value=1, max_value=1000),
       weights_init           = st.sampled_from(weights),
       activation             = st.sampled_from(activations),
       optimizer              = st.sampled_from(optimizers)
       )
@settings(deadline=None)
def test_weights_orthogonalization (samples, inputs, outputs, batch_size, weights_init, activation, optimizer):
  '''
  Test the weights orthogonalization algorithm checking that the dot product
  between each pair of distinct row of the synaptic weights matrix is 0
  (orthogonal vectors).
  Notice that the dot product of each row by itself is not 1 because the
  vectors norms must be conserved (no normalization).
  It is also assumed that the number of outputs is less or equal to the number
  of inputs (data features), otherwise a set of orthogonal vectors in the
  features space does not exist.
  '''

  def is_orthogonal (A):
    offdiag = ~np.eye(A.shape[0], dtype=bool)
    dot = A @ A.transpose()
    return np.allclose(dot[offdiag], 0.)

  assume(batch_size <= samples)
  assume(outputs <= inputs)

  data = np.random.uniform(low=0., high=1., size=(samples,inputs)).astype(float)

  bcm = BCM(inputs=inputs, outputs=outputs, num_epochs=1, batch_size=batch_size,
            weights_init=weights_init(),
            activation=activation(),
            optimizer=optimizer(),
            orthogonalization=True,
            verbose=False)

  bcm.fit(X=data)

  assert is_orthogonal(bcm.weights)
