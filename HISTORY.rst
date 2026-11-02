=======
History
=======

0.4.0 (2026-10-20)
------------------

* Stabilize the Python-Julia bridge: cache the loaded module, resolve and
  instantiate development projects, and verify the Julia core version.
* Pin the Julia core to QuanEstimation 0.4.0.
* Add Julia 1.11/1.12/1.13 CI coverage.
* Unify the package version metadata at 0.4.0.
* Add version/pin consistency tests.
* Document the Julia-first thin-shell direction; the QFIM/CFIM delegation
  lands in 0.5.0.

0.5.0 (2026-10-20)
------------------

* Pin the Julia core to QuanEstimation 0.5.0.
* Delegate the core ``QFIM``/``CFIM``/``SLD``/``RLD``/``LLD``/``FIM`` kernels
  to the Julia core; specialized Bloch/Gauss kernels remain for a later
  migration.
* Add a low-temperature thermometry regression test that verifies the
  delegated kernels reproduce the analytic QFI.
* Add Julia 1.11/1.12/1.13 CI coverage.
* Unify the package version metadata at 0.5.0.
* Make the Julia bridge development path (``QuanEstimation_DEV_PATH``) load
  the root package and its subpackages into the juliacall environment, with a
  version check and caching.
* Add tests for version/pin consistency and the Julia bridge.
* Continue the Julia-first thin-shell migration; specialized kernels and
  optimization wrappers remain for a later release.
* Fix multiparameter Bayesian cost regressions.

0.3.0 (2026-07-08)
------------------

* Refactored package structure: split into ``quanestimation.base`` (core) and ``quanestimation.nv`` (NV magnetometer extension).
* Added :class:`NVMagnetometerScheme` class for NV magnetometry support.
* Added control waveform constructors: ``ZeroCTRL``, ``LinearCTRL``, ``SineCTRL``, ``SawCTRL``, ``TriangleCTRL``, ``GaussianCTRL``, ``GaussianEdgeCTRL``.
* Added :func:`error_evaluation` and :func:`error_control` for error analysis.
* Added state utilities: ``BellState``, ``PlusState``, ``MinusState``, ``SigmaX``/``SigmaY``/``SigmaZ``, ``sx``/``sy``/``sz``, :func:`fidelity`.
* Added :func:`QFIM_pure` and :func:`Williamson_form`.
* Added the :func:`QubitDephasing` dynamics constructor.
* Unified Julia-to-Python conversion layer.

0.2.8 (2025-08-03)
------------------

0.2.7 (2025-07-22)
------------------

0.2.6 (2025-06-13)
------------------

0.2.5 (2024-10-16)
------------------

0.2.4 (2024-05-11)
------------------

0.2.3 (2024-03-21)
------------------

0.2.2 (2024-03-19)
------------------

0.2.1 (2024-02-08)
------------------

0.2.0 (2022-09-01)
------------------

0.1.5 (2022-07-07)
------------------

0.1.4 (2022-06-19)
------------------

0.1.3 (2022-06-08)
------------------

0.1.0 (2022-06-04)
------------------

* First release on PyPI.
