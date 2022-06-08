QuanEstimation
==============

|Dev|

QuanEstimation is a Python-Julia based open-source toolkit for quantum
parameter estimation, which consist in the calculation of the quantum
metrological tools and quantum resources, the optimization of the probe
state, control and measurement in quantum metrology. Futhermore,
QuanEstimation can also perform comprehensive optimization with respect
to the probe state, control and measurement to generate not only optimal
quantum parameter estimation schemes, but also adaptive measurement
schemes.

Documentation
-------------

The documentation of QuanEstimation can be found
`here <https://quanestimation.github.io/QuanEstimation/>`__.

Notes
-----

Welcome to the QuanEstimation community! Feel free to submit issues and
pull requests.

| We are still working hard uploading our package to the PyPI and making
  our docs online. And we are also waiting for our `QuanEstimation.jl
  registration <https://github.com/JuliaRegistries/General/pull/61399#issuecomment-1142241816>`__
  to be auto-merged, which will be much more convenient for our users to
  setup their julia environment.
| So, it’s highly recommended to wait until all these works are
  finished, and follow our documentations to have a better using
  experience.

Nevertheless, if you still want to manually install the toolkit, you can
1. ``git clone`` this repo to local and ``cd QuanEstimaiton``, 2.
``pip install .`` or ``python setup.py install`` to install the python
package, 3. `download julia <https://julialang.org/downloads/>`__ and
install. Or simply via ``pip install jill`` and ``jill install``, 4. set
up julia environment by adding the dependences. Currently this step is
somewhat cumbersome, 1. check `Julia’s
docs <https://docs.julialang.org/en/v1/stdlib/Pkg/>`__ if you are not
familiar with julia’s package management, 2. add the deps
`here <https://github.com/QuanEstimation/QuanEstimation.jl/blob/e1b3b5ab5ac23c01eacd56de5440fcdcf36358d4/Project.toml#L6>`__
to your julia environment **via julia’s REPL** manually 3. and then run
python from command line to set up pyjulia, see `pyJulia’s
documentation <https://pyjulia.readthedocs.io/en/stable/>`__
``python         import julia         julia.install()`` 5.
``import QuanEstimation`` to load the package. 6. then run the examples
in ``quanestimation/examples/`` folder and have fun.

Installation
------------

Run the command in the terminal to install QuanEstimation:

::

   pip install quanestimation

Citation
--------

If you use QuanEstimation in your research, please cite the following
paper:

[1] M. Zhang, H.-M. Yu, H. Yuan, X. Wang, R. Demkowicz-Dobrzański, and
J. Liu, QuanEstimation: an open-source toolkit for quantum parameter
estimation,
`arXiv:2205.15588 <https://doi.org/10.48550/arXiv.2205.15588>`__.

.. |Dev| image:: https://img.shields.io/badge/docs-dev-blue.svg
   :target: https://quanestimation.github.io/QuanEstimation/
