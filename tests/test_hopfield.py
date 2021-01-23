from __future__ import print_function
from __future__ import division


from biolearn.utils.optimizer import *
from biolearn.utils.weights import *

from biolearn.model.hopfield import Hopfield

from hypothesis import strategies as st
from hypothesis import given, settings


__author__  = ['SimoneGasperini']
__email__   = ['simone.gasperini2@studio.unibo.it']


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
       optimizer              = st.sampled_from(optimizers),
       delta                  = st.floats(),
       p                      = st.floats(),
       k                      = st.integers(),
       precision              = st.floats(),
       epochs_for_convergency = st.integers(min_value=1),
       convergency_atol       = st.floats(),
       random_state           = st.integers(),
       verbose                = st.booleans()
       )
@settings(deadline=None)
def test_constructor (inputs, outputs, num_epochs, batch_size, weights_init, optimizer,
                      delta, p, k, precision, epochs_for_convergency,
                      convergency_atol, random_state, verbose):
  '''
  Test the Hopfield object constructor.
  The number of inputs and outputs are bounded in a range of reasonable values
  (also to prevent the test from being too slow).
  '''

  params = {'inputs'                 : inputs,
            'outputs'                : outputs,
            'num_epochs'             : num_epochs,
            'batch_size'             : batch_size,
            'weights_init'           : weights_init(),
            'optimizer'              : optimizer(),
            'delta'                  : delta,
            'p'                      : p,
            'k'                      : k,
            'precision'              : precision,
            'epochs_for_convergency' : epochs_for_convergency,
            'convergency_atol'       : convergency_atol,
            'random_state'           : random_state,
            'verbose'                : verbose
            }

  hopfield = Hopfield(**params)
  assert params == hopfield.get_params()
