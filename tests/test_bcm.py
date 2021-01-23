from __future__ import print_function
from __future__ import division


from biolearn.utils.activations import *
from biolearn.utils.optimizer import *
from biolearn.utils.weights import *

import numpy as np
from biolearn.model.bcm import BCM

from hypothesis import strategies as st
from hypothesis import given, settings


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
  interaction matrix). If incorrect values of interaction_strength are passed,
  a ValueError is raised.
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
