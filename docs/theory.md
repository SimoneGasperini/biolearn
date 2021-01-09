# Overview
Despite the great success of backpropagation algorithm in deep learning, a question remains to what extent the computational properties of artificial neural networks are comparable to the plasticity rules of the human brain.
Indeed, even if the architectures of real and artificial neural networks are similar, the supervised training based on backpropagation and the biological learning rules are unrelated.

In the paper by [D. Krotov and J. J. Hopfield](https://arxiv.org/abs/1806.10181) (Ref. [1]), it is proposed an unusual learning rule, which has a degree of biological plausibility, and which is motivated by well known ideas in neuroplasticity theory:

* Hebb's rule: changes of the synapse strength depend only on the activities of the pre- and post-synaptic neurons and so the learning is **physically local** and describable by local mathematics;

* the core of the learning procedure is **unsupervised** because it is believed to be mainly observational, with few or no labels and no explicit task.

Starting from these concepts, they were able to design an algorithm (based on an extension of the *Oja rule*) capable of learning early feature detectors in a completely unsupervised way, and then they used them to train a traditional supervised neural network layer.

In their algorithm there is no top–down propagation of information, the synaptic
weights are learned using only bottom–up signals, and the algorithm is agnostic about the task that the network will have to solve eventually in the top layer. Despite this lack of knowledge about the task, the algorithm finds a useful set of weights that leads to a good generalization performance on the standard classification task, at least on simple standard datasets like the MNIST and CIFAR-10.

In this project, a parallel approach founded on the same basic concepts is proposed.
In particular, it was developed an algorithm based on the *BCM theory* (E. Bienenstock, L. Cooper, and P. Munro) with lateral interactions between neurons. An exhaustive and detailed theoretical description is provided by the paper by [Castellani et al.](https://pubmed.ncbi.nlm.nih.gov/10378187/) (Ref. [2]).
In general terms, BCM model proposes a sliding threshold for long-term potentiation (LTP) or long-term depression (LTD), and states that synaptic plasticity is stabilized by a dynamic adaptation of the time-averaged post-synaptic activity.

Lateral interactions between neurons are introduced to change the basins of attraction associated with different solutions, without affecting the stability of the possible solutions.
When the interaction terms are set to negative values, for instance, the probabilities of reaching one specific stable state are different for each neuron.
This selective behaviour is important to make each neuron sensitive to different patterns of the input data, providing a good features-encoding.


# BCM theory
The general idea of BCM theory is that, for a random sequence of input patterns, a synapse is learning to differentiate between those stimuli that excite the post-synaptic neuron strongly and those stimuli that excite that neuron weakly.
In particular, the BCM algorithm performs a dynamical adaptation of the weights based on the average post-synaptic activity of each neuron.

## BCM with lateral interactions
Let's consider a single input example (representing the pre-synaptic activities) consisting in a column vector <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{x}&space;=&space;(x_1,&space;...,&space;x_m)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{x}&space;=&space;(x_1,&space;...,&space;x_m)" title="\mathbf{x} = (x_1, ..., x_m)" /></a>  where <a href="https://www.codecogs.com/eqnedit.php?latex=m" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m" title="m" /></a> is the number of data features.
The implemented BCM network has a single dense layer, composed by <a href="https://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k" title="k" /></a> laterally-interacting neurons.
The output column vector <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{y}&space;=&space;(y_1,&space;...,&space;y_k)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{y}&space;=&space;(y_1,&space;...,&space;y_k)" title="\mathbf{y} = (y_1, ..., y_k)" /></a>  represents the post-synaptic activities and it is given by the feedforward equation:

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{y}&space;=&space;\sigma&space;\left[&space;(I&space;-&space;L)^{-1}&space;\cdot&space;W&space;\cdot&space;\mathbf{x}&space;\right]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{y}&space;=&space;\sigma&space;\left[&space;(I&space;-&space;L)^{-1}&space;\cdot&space;W&space;\cdot&space;\mathbf{x}&space;\right]" title="\mathbf{y} = \sigma \left[ (I - L)^{-1} \cdot W \cdot \mathbf{x} \right]" /></a>

with <a href="https://www.codecogs.com/eqnedit.php?latex=W" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W" title="W" /></a> (<a href="https://www.codecogs.com/eqnedit.php?latex=k&space;\times&space;m" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k&space;\times&space;m" title="k \times m" /></a>) the matrix of synapses (weights), <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma" title="\sigma" /></a> the activation function, and <a href="https://www.codecogs.com/eqnedit.php?latex=L" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L" title="L" /></a> (<a href="https://www.codecogs.com/eqnedit.php?latex=k&space;\times&space;k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k&space;\times&space;k" title="k \times k" /></a>) the following symmetric matrix of lateral interactions:

<a href="https://www.codecogs.com/eqnedit.php?latex=L&space;=&space;\begin{pmatrix}&space;0&space;&&space;l_{12}&space;&&space;\cdots&space;&&space;l_{1k}\\&space;l_{21}&space;&&space;0&space;&&space;&&space;l_{2k}\\&space;\vdots&space;&&space;&&space;\ddots&space;&&space;\vdots\\&space;l_{k1}&space;&&space;l_{k2}&space;&&space;\cdots&space;&&space;0&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L&space;=&space;\begin{pmatrix}&space;0&space;&&space;l_{12}&space;&&space;\cdots&space;&&space;l_{1k}\\&space;l_{21}&space;&&space;0&space;&&space;&&space;l_{2k}\\&space;\vdots&space;&&space;&&space;\ddots&space;&&space;\vdots\\&space;l_{k1}&space;&&space;l_{k2}&space;&&space;\cdots&space;&&space;0&space;\end{pmatrix}" title="L = \begin{pmatrix} 0 & l_{12} & \cdots & l_{1k}\\ l_{21} & 0 & & l_{2k}\\ \vdots & & \ddots & \vdots\\ l_{k1} & l_{k2} & \cdots & 0 \end{pmatrix}" /></a>

where the element <a href="https://www.codecogs.com/eqnedit.php?latex=l_{ij}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l_{ij}" title="l_{ij}" /></a> represents the interaction strength between neuron <a href="https://www.codecogs.com/eqnedit.php?latex=i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i" title="i" /></a> and neuron <a href="https://www.codecogs.com/eqnedit.php?latex=j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?j" title="j" /></a>.

The computation of the weights update is obtained by numerically integrating the following differential equations (which describe the BCM learning rule):

<a href="https://www.codecogs.com/eqnedit.php?latex=\dot&space;W&space;=&space;\mathbf{\Phi}(\mathbf{y})&space;\cdot&space;\mathbf{x}^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot&space;W&space;=&space;\mathbf{\Phi}(\mathbf{y})&space;\cdot&space;\mathbf{x}^T" title="\dot W = \mathbf{\Phi}(\mathbf{y}) \cdot \mathbf{x}^T" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{\Phi}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{\Phi}" title="\mathbf{\Phi}" /></a> is a non-linear function that depends only on the post-synaptic activity <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{y}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{y}" title="\mathbf{y}" /></a> (output vector) and it is defined as:

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{\Phi}&space;(\mathbf{y})&space;=&space;\mathbf{y}&space;(\mathbf{y}&space;-&space;\mathbf{\theta})&space;\mathbin{/}&space;\mathbf{\theta}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{\Phi}&space;(\mathbf{y})&space;=&space;\mathbf{y}&space;(\mathbf{y}&space;-&space;\mathbf{\theta})&space;\mathbin{/}&space;\mathbf{\theta}" title="\mathbf{\Phi} (\mathbf{y}) = \mathbf{y} (\mathbf{y} - \mathbf{\theta}) \mathbin{/} \mathbf{\theta}" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\theta_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_i" title="\theta_i" /></a> is an average value <a href="https://www.codecogs.com/eqnedit.php?latex=E[y_i^2]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E[y_i^2]" title="E[y_i^2]" /></a> for neuron <a href="https://www.codecogs.com/eqnedit.php?latex=i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i" title="i" /></a> computed over the input examples.

For low values of the post-synaptic activity <a href="https://www.codecogs.com/eqnedit.php?latex={y_i}&space;<&space;\theta_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{y_i}&space;<&space;\theta_i" title="{y_i} < \theta_i" /></a>, the function <a href="https://www.codecogs.com/eqnedit.php?latex=\Phi_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Phi_i" title="\Phi_i" /></a> is negative; for <a href="https://www.codecogs.com/eqnedit.php?latex={y_i}&space;>&space;\theta_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{y_i}&space;>&space;\theta_i" title="{y_i} > \theta_i" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\Phi_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Phi_i" title="\Phi_i" /></a> is positive.
The rule stabilizes by allowing the modification threshold <a href="https://www.codecogs.com/eqnedit.php?latex=\theta_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_i" title="\theta_i" /></a> to vary as a super-linear function of the average activity of the cell. Unlike traditional methods of stabilizing Hebbian learning, this sliding threshold provides a mechanism for incoming patterns to compete.

Moreover, the synaptic modifications increase when the threshold is small, and decrease as the threshold increases. The practical result is that the simulation can be run with artificially high learning rates, and wild oscillations are reduced.
For more details about different implementation of the BCM algorithm see the following *Scholarpedia* page: [BCM theory](http://www.scholarpedia.org/article/BCM_theory) (Ref. [3]).

## Weights orthogonalization
Beyond lateral interactions, it is also implemented an alternative approach to force neurons selectivity. This technique is based on the orthogonal initialization of the synaptic weights and their iterative orthogonalization at the end of each training epoch.
The orthogonalization algorithm forces each neuron to become selective to different patterns and it is based on the singular values decomposition (SVD):

<a href="https://www.codecogs.com/eqnedit.php?latex=\\W&space;=&space;U&space;\cdot&space;S&space;\cdot&space;V^T&space;\\&space;\\W'&space;=&space;U&space;\cdot&space;V^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\W&space;=&space;U&space;\cdot&space;S&space;\cdot&space;V^T&space;\\&space;\\W'&space;=&space;U&space;\cdot&space;V^T" title="\\W = U \cdot S \cdot V^T \\ \\W' = U \cdot V^T" /></a>

This transformation is actually performing a weights orthonormalization and so, to restore the synaptic weights vectors norms, they are also multiplied by their original norm, computed before the matrix <a href="https://www.codecogs.com/eqnedit.php?latex=W" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W" title="W" /></a> decomposition.

The drawback of this technique is that the receptive fields of the neurons could be sensibly affected and the convergence rate to a specific pattern is significantly reduced.

## References

<blockquote>[1] Dmitry Krotov, John J. Hopfield, Unsupervised learning by competing hidden units, 2019, https://arxiv.org/abs/1806.10181</blockquote>

<blockquote>[2] Castellani G., Intrator N., Shouval H.Z., Cooper L.N., Solutions of the BCM learning rule in a network of lateral interacting nonlinear neurons, 1998, https://pubmed.ncbi.nlm.nih.gov/10378187/</blockquote>

<blockquote>[3] Blais B., Cooper L.N., BCM theory, 2008, http://www.scholarpedia.org/article/BCM_theory</blockquote>
