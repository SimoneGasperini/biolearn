Base
----

**Base class** : 
it is an abstract class integrated with the common Machine Learning tools of `scikit-learn` package
and it provides the standart methods `fit` and `predict`.
Note that the models are unsupervised and so the passed parameter will be just the data
(without any label).
Like other Machine Learning algorithms also the `biolearn` ones depend on many hyper-parameters,
which have to be tuned according to the given problem.


.. automodule:: biolearn.model._base
   :members:
   :show-inheritance: