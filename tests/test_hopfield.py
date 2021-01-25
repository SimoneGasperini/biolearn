from __future__ import print_function
from __future__ import division


from biolearn.utils.optimizer import *
from biolearn.utils.weights import *

import os
import numpy as np
from biolearn.model.hopfield import Hopfield

from hypothesis import strategies as st
from hypothesis import given, settings, assume


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
       k                      = st.integers(min_value=1, max_value=5),
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
  The ranking parameter k is assumed to be less or equal to the number of outputs.
  '''

  assume(k <= outputs)

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



@given(samples                = st.integers(min_value=1, max_value=1000),
       inputs                 = st.integers(min_value=1, max_value=20),
       outputs                = st.integers(min_value=1, max_value=5),
       batch_size             = st.integers(min_value=1),
       weights_init           = st.sampled_from(weights),
       optimizer              = st.sampled_from(optimizers),
       k                      = st.integers(min_value=1, max_value=5)
       )
@settings(deadline=None)
def test_save_load_weights (samples, inputs, outputs, batch_size,
                            weights_init, optimizer, k):
  '''
  Test the save and load weights methods.
  '''

  assume(batch_size <= samples)
  assume(k <= outputs)

  data = np.random.uniform(low=0., high=1., size=(samples,inputs)).astype(float)

  hopfield = Hopfield(inputs=inputs, outputs=outputs, num_epochs=1, batch_size=batch_size,
                      weights_init=weights_init(), optimizer=optimizer(), k=k, verbose=False)

  hopfield.fit(X=data)

  assert hopfield.save_weights('weights.bin')
  hopfield_new = hopfield.load_weights('weights.bin')
  os.remove('weights.bin')

  assert np.all(hopfield.weights == hopfield_new.weights)



@given(samples                = st.integers(min_value=1, max_value=1000),
       inputs                 = st.integers(min_value=1, max_value=50),
       outputs                = st.integers(min_value=1, max_value=10),
       k                      = st.integers(min_value=1, max_value=5)
       )
@settings(deadline=None)
def test_predict (samples, inputs, outputs, k):
  '''
  Test the predict method in case of null, positive, and negative input 
  random data and positive or negative weights.
  '''

  assume(k <= outputs)

  zeros = np.zeros(shape=(samples,inputs)).astype(float)
  pos = np.random.uniform(low=0., high=1., size=(samples,inputs)).astype(float)
  neg = np.random.uniform(low=-1., high=0., size=(samples,inputs)).astype(float)

  hopfield = Hopfield(inputs=inputs, outputs=outputs,
                      weights_init=TruncatedNormal(mu=2.,std=1.),
                      k=k, verbose=False)

  Y1 = hopfield.predict(X=zeros)
  assert np.all(Y1 == np.zeros(shape=(outputs,samples)))

  Y2 = hopfield.predict(X=pos)
  assert np.all(Y2 >= 0.)

  Y3 = hopfield.predict(X=neg)
  assert np.all(Y3 <= 0.)

  hopfield.weights = - hopfield.weights

  Y4 = hopfield.predict(X=zeros)
  assert np.all(Y4 == np.zeros(shape=(outputs,samples)))

  Y5 = hopfield.predict(X=pos)
  assert np.all(Y5 <= 0.)

  Y6 = hopfield.predict(X=neg)
  assert np.all(Y6 >= 0.)
