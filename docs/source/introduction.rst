Introduction
============

Despite the great success of backpropagation algorithm in deep learning,
a question remains to what extent the computational properties of
artificial neural networks are comparable to the plasticity rules of the
human brain. Indeed, even if the architectures of real and artificial
neural networks are similar, the supervised training based on
backpropagation and the biological learning rules are unrelated.

In the paper by `D. Krotov and J. J.
Hopfield <https://arxiv.org/abs/1806.10181>`__ (Ref. [1]), it is
proposed an unusual learning rule, which has a degree of biological
plausibility, and which is motivated by well known ideas in
neuroplasticity theory:

-  Hebb's rule: changes of the synapse strength depend only on the
   activities of the pre- and post-synaptic neurons and so the learning
   is **physically local** and describable by local mathematics;

-  the core of the learning procedure is **unsupervised** because it is
   believed to be mainly observational, with few or no labels and no
   explicit task.

Starting from these concepts, they were able to design an algorithm
(based on an extension of the *Oja rule*) capable of learning early
feature detectors in a completely unsupervised way, and then they used
them to train a traditional supervised neural network layer.

In their algorithm there is no top–down propagation of information, the
synaptic weights are learned using only bottom–up signals, and the
algorithm is agnostic about the task that the network will have to solve
eventually in the top layer. Despite this lack of knowledge about the
task, the algorithm finds a useful set of weights that leads to a good
generalization performance on the standard classification task, at least
on simple standard datasets like the MNIST and CIFAR-10.

In this project, a parallel approach founded on the same basic concepts
is proposed. In particular, it was developed an algorithm based on the
*BCM theory* (E. Bienenstock, L. Cooper, and P. Munro) with lateral
interactions between neurons. An exhaustive and detailed theoretical
description is provided by the paper by `Castellani et
al. <https://pubmed.ncbi.nlm.nih.gov/10378187/>`__ (Ref. [2]). In
general terms, BCM model proposes a sliding threshold for long-term
potentiation (LTP) or long-term depression (LTD), and states that
synaptic plasticity is stabilized by a dynamic adaptation of the
time-averaged post-synaptic activity.

Lateral interactions between neurons are introduced to change the basins
of attraction associated with different solutions, without affecting the
stability of the possible solutions. When the interaction terms are set
to negative values, for instance, the probabilities of reaching one
specific stable state are different for each neuron. This selective
behaviour is important to make each neuron sensitive to different
patterns of the input data, providing a good features-encoding.