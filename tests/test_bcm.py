from __future__ import print_function
from __future__ import division


from biolearn.utils.activations import *
from biolearn.utils.optimizer import *
from biolearn.utils.weights import *

from biolearn.model.bcm import BCM

from hypothesis import strategies as st
from hypothesis import given


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



@given(outputs                = st.integers(min_value=1, max_value=1000),
       num_epochs             = st.integers(min_value=1),
       batch_size             = st.integers(),
       activation             = st.sampled_from(activations),
       optimizer              = st.sampled_from(optimizers),
       weights_init           = st.sampled_from(weights),
       orthogonalization      = st.booleans(),
       interaction_strength   = st.floats(),
       precision              = st.floats(),
       epochs_for_convergency = st.integers(min_value=1),
       convergency_atol       = st.floats(),
       random_state           = st.integers(),
       verbose                = st.booleans()
       )
def test_constructor (outputs, num_epochs, batch_size, activation, optimizer, weights_init,
                      orthogonalization, interaction_strength, precision, epochs_for_convergency,
                      convergency_atol, random_state, verbose):
  '''
  Test the BCM object constructor.
  '''

  params = {'outputs'                : outputs,
            'num_epochs'             : num_epochs,
            'batch_size'             : batch_size,
            'activation'             : activation(),
            'optimizer'              : optimizer(),
            'weights_init'           : weights_init(),
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
