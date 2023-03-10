.. image:: https://img.shields.io/badge/python-2.7|3.7-blue.svg
   :target: https://www.python.org/
   :alt: Python version
.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/niklastoe/classifier_metric_uncertainty/master?urlpath=%2Fvoila%2Frender%2Finteractive_notebook.ipynb

Classifier Metric Uncertainty Due to Small Sample Sizes
======

Classifier metrics (such as accuracy, sensitivity, specificity, precision...) are highly uncertain if they are calculated from a small sample size. 
Unfortunately, these point estimates are often considered to be exact.
We present a Bayesian method to determine metric uncertainty. 
`Our paper <https://peerj.com/articles/cs-398/>`_ explains the underlying concepts and showcases that many published classifiers have surprisingly large metric uncertainties.
This repository contains the implementation in Python.

Usage
-----
The easiest way to calculate metric uncertainty is via our interactive, browser-based tool.
The site may take a few minutes to load.
It does not install any packages or execute any code on your machine, it needs to start the environment on the host.
This causes the small delay.
`Please follow this link to the browser-based tool. <https://mybinder.org/v2/gh/niklastoe/classifier_metric_uncertainty/master?urlpath=%2Fvoila%2Frender%2Finteractive_notebook.ipynb>`_

If you want to calculate metric uncertainties on a regular basis or even integrate the method into your workflow, feel free to copy this repository.
``tutorial.ipynb`` should give you an idea how to use the most important parts of the code.

Reproducibility
---------------
.. All notebooks to recreate the analysis presented in the paper can be found in ``paper/``.
To ensure that all dependencies work as intended, type ``pip install -r requirements.txt``.

Non-standard Packages & Tools
--------
* `pymc3 <https://docs.pymc.io/>`_ (Gelman-Rubin diagnostics and tests)
* `sympy <https://www.sympy.org/en/index.html>`_ (metric definition)
* `Voila <https://github.com/voila-dashboards/voila>`_ (turns my `Jupyter Notebook <https://github.com/jupyter>`_ into a standalone application)
* `Binder <https://mybinder.org/>`_ (hosts the application)


Citation
--------

.. code-block:: latex

   @article{toetsch2021classifier,
   title={Classifier uncertainty: evidence, potential impact, and probabilistic treatment},
   author = {Tötsch, Niklas and Hoffmann, Daniel},
   journal={PeerJ Computer Science},
   volume={7},
   pages={e398},
   year={2021},
   publisher={PeerJ Inc.}}

Contributing
------------
If you have questions or comments, please create an `issue <https://github.com/niklastoe/classifier_metric_uncertainty/issues>`_.
