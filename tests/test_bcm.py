from __future__ import print_function
from __future__ import division


from biolearn.utils.activations import *
from biolearn.utils.optimizer import *
from biolearn.utils.weights import *

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
       num_epochs             = st.integers(min_value=1, max_value=10),
       batch_size             = st.integers(min_value=1, max_value=1000),
       optimizer              = st.sampled_from(optimizers),
       )
@settings(deadline=None)
def test_positive_weights_with_negative_data (samples, inputs, outputs, num_epochs, batch_size, optimizer):
  '''
  Test the model stability in case of positive (or null) weights,
  negative (or null) input data, and Relu activation function (so that f(x)=0
  forall x <= 0).
  There are no lateral interactions at all between the neurons and the
  orthogonalization algorithm is disabled.
  The data are generated using a uniform distribution U(-1,0) and the weights
  are initialized using a truncated normal distribution N*(2,1).
  '''

  assume(batch_size <= samples)

  data = np.random.uniform(low=-1., high=0., size=(samples,inputs)).astype(float)

  bcm = BCM(inputs=inputs, outputs=outputs, num_epochs=num_epochs, batch_size=batch_size,
            weights_init=TruncatedNormal(mu=2.,std=1.),
            activation=Relu(),
            optimizer=optimizer(), verbose=False)

  initial_weights = np.copy(bcm.weights)
  bcm.fit(X=data)
  final_weights = np.copy(bcm.weights)

  assert np.all(initial_weights == final_weights)



@given(samples                = st.integers(min_value=1, max_value=1000),
       inputs                 = st.integers(min_value=1, max_value=20),
       outputs                = st.integers(min_value=1, max_value=5),
       num_epochs             = st.integers(min_value=1, max_value=10),
       batch_size             = st.integers(min_value=1, max_value=1000),
       optimizer              = st.sampled_from(optimizers),
       )
@settings(deadline=None)
def test_negative_weights_with_positive_data (samples, inputs, outputs, num_epochs, batch_size, optimizer):
  '''
  Test the model stability in case of negative (or null) weights,
  positive (or null) input data, and Relu activation function (so that f(x)=0
  forall x <= 0).
  There are no lateral interactions at all between the neurons and the
  orthogonalization algorithm is disabled.
  The data are generated using a uniform distribution U(0,1) and the weights
  are initialized using a truncated normal distribution N*(-2,1).
  '''

  assume(batch_size <= samples)

  data = np.random.uniform(low=0., high=1., size=(samples,inputs)).astype(float)

  bcm = BCM(inputs=inputs, outputs=outputs, num_epochs=num_epochs, batch_size=batch_size,
            weights_init=TruncatedNormal(mu=-2.,std=1.),
            activation=Relu(),
            optimizer=optimizer(), verbose=False)

  initial_weights = np.copy(bcm.weights)
  bcm.fit(X=data)
  final_weights = np.copy(bcm.weights)

  assert np.all(initial_weights == final_weights)
