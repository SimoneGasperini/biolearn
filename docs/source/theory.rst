Theory
======

The general idea of BCM theory is that, for a random sequence of input patterns, a synapse is
learning to differentiate between those stimuli that excite the post-synaptic neuron strongly
and those stimuli that excite that neuron weakly. In particular, the BCM algorithm performs
a dynamical adaptation of the weights based on the average post-synaptic activity of each neuron.


BCM with lateral interactions
-----------------------------

Let's consider a single input example (representing the pre-synaptic activities) consisting
in a column vector :math:`\mathbf{x} = (x_1, ..., x_m)` where :math:`m` is the number of data
features. The implemented BCM network has a single dense layer, composed by
laterally-interacting neurons. The output column vector :math:`\mathbf{y} = (y_1, ..., y_k)`
represents the post-synaptic activities and it is given by the feedforward equation:

.. math::
    \mathbf{y} = \sigma \left[ (I - L)^{-1} \cdot W \cdot \mathbf{x} \right]

with :math:`W(k \times m)` the matrix of synapses (weights), :math:`\sigma` the activation
function, and :math:`L(k \times k)` the following symmetric matrix of lateral interactions:

.. math::
    L = \begin{pmatrix} 0 & l_{12} & \cdots & l_{1k} \\
        l_{21} & 0 & & l_{2k} \\
        \vdots & & \ddots & \vdots \\
        l_{k1} & l_{k2} & \cdots & 0 \end{pmatrix}

where the element :math:`l_{ij}` represents the interaction strength between neuron :math:`i`
and neuron :math:`j`.

The computation of the weights update is obtained by numerically integrating the following
differential equations (which describe the BCM learning rule):

.. math::
    \dot W = \mathbf{\Phi}(\mathbf{y}) \cdot \mathbf{x}^T

:math:`\mathbf{\Phi}` is a non-linear function that depends only on the post-synaptic activity
:math:`\mathbf{y}` and it is defined as:

.. math::
    \mathbf{\Phi} (\mathbf{y}) = \mathbf{y} (\mathbf{y} - \mathbf{\theta}) \mathbin{/} \mathbf{\theta}

where :math:`\theta_i` is the average value :math:`E[y_i^2]` on neuron :math:`i` computed over
the input examples.

For low values of the post-synaptic activity :math:`{y_i} < \theta_i`, the function :math:`\Phi_i`
is negative; for higher values :math:`{y_i} > \theta_i`, :math:`\Phi_i` is positive. The rule
stabilizes by allowing the modification threshold :math:`\theta_i` to vary as a super-linear
function of the average activity of the cell. Unlike traditional methods of stabilizing Hebbian
learning, this sliding threshold provides a mechanism for incoming patterns to compete.

Moreover, the synaptic modifications increase when the threshold is small, and decrease as the
threshold increases. The practical result is that the simulation can be run with artificially
high learning rates, and wild oscillations are reduced. For more details about different
implementation of the BCM algorithm see the following *Scholarpedia* page: `BCM theory
<http://www.scholarpedia.org/article/BCM_theory>`__ (Ref. [3]).


Weights orthogonalization
-------------------------

Beyond lateral interactions, it is also implemented an alternative approach to force neurons
selectivity. This technique is based on the orthogonal initialization of the synaptic weights
and their iterative orthogonalization at the end of each training epoch. The orthogonalization
algorithm forces each neuron to become selective to different patterns and it is based on the
singular values decomposition (SVD):

.. math::
    W = U \cdot S \cdot V^T

.. math::
    W' = U \cdot V^T

This transformation is actually performing a weights orthonormalization and so, to restore
the synaptic weights vectors norms, they are also multiplied by their original norm,
computed before the matrix :math:`W` decomposition.

The drawback of this technique is that the receptive fields of the neurons could be sensibly
affected and the convergence rate to a specific pattern is significantly reduced.