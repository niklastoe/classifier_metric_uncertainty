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
The corresponding paper will be submitted soon and explains the underlying concepts.
This repository contains the implementation in Python.

Usage
-----
Please use the `interactive, browser-based tool <https://mybinder.org/v2/gh/niklastoe/classifier_metric_uncertainty/master?urlpath=%2Fvoila%2Frender%2Finteractive_notebook.ipynb>`_.
No programming skills or advanced statistical knowledge needed!
If you want to integrate the method into your workflow, feel free to copy this repository.

Reproducibility
---------------
All notebooks to recreate the analysis presented in the paper can be found in ``paper/``.
To ensure that all dependencies work as intended, type ``pip install -r requirements.txt``.

Non-standard Packages & Tools
--------
* `pymc3 <https://docs.pymc.io/>`_ (Gelman-Rubin diagnostics and tests)
* `sympy <https://www.sympy.org/en/index.html>`_ (metric definition)
* `Voila <https://github.com/voila-dashboards/voila>`_ (turns my `Jupyter Notebook <https://github.com/jupyter>`_ into a standalone application)
* `Binder <https://mybinder.org/>`_ (hosts the application)


Citation
--------

Contributing
------------
If you have questions or comments, please create an `issue <https://github.com/niklastoe/classifier_metric_uncertainty/issues>`_.
