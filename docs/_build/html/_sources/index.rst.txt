.. iceshot documentation master file, created by
   sphinx-quickstart on Fri Jun 28 10:49:32 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Incompressible Cell Shapes via Optimal Transport
***************************************************

.. image:: _static/fallingstuff.gif
    :scale: 100% 
    :alt: Falling Soft Spheres
    :align: center

|

Many biological systems such as cell aggregates, tissues or bacterial colonies behave as unconventional systems of particles that are strongly constrained by volume exclusion and shape interactions. Understanding how these constraints lead to macroscopic self-organized structures is a fundamental question in e.g. developmental biology. To this end, various types of computational models have been developed: phase fields, cellular automata, vertex models, level-set, finite element simulations, etc. We introduce a new framework based on optimal transport theory to model particle systems with arbitrary dynamical shapes and deformability. Our method builds upon the pioneering work of Brenier on incompressible fluids and its recent applications to materials science. It lets us specify the shapes of individual cells and supports a wide range of interaction mechanisms, while automatically taking care of the volume exclusion constraint at an affordable numerical cost.

.. rst-class:: center

   **Please check the** `gallery of examples <_auto_examples/index.html>`_ **!**

The project is hosted on `GitHub <https://github.com/antoinediez/ICeShOT>`_, under the permissive `MIT license <https://en.wikipedia.org/wiki/MIT_License>`_.

Citation
=====================

If you use ICeShOT in a research paper, please cite the `arXiv preprint <https://arxiv.org/abs/2402.17086>`_ : ::

   @misc{diez2024optimaltransportmodeldynamical,
   title={An optimal transport model for dynamical shapes, collective motion and cellular aggregates}, 
   author={Antoine Diez and Jean Feydy},
   year={2024},
   eprint={2402.17086},
   archivePrefix={arXiv},
   primaryClass={q-bio.QM},
   url={https://arxiv.org/abs/2402.17086}, 
   }

Diez A., Feydy J., An optimal transport model for dynamical shapes, collective motion and cellular aggregates, arXiv preprint: arXiv2402.17086, 2024

Authors
===========

`Antoine Diez <https://antoinediez.gitlab.io/>`_, Kyoto University Institute for the Advanced Study of Human Biology (ASHBi)

`Jean Feydy <https://www.jeanfeydy.com/>`_, Inria Paris, HeKA team

Table of contents
========================

.. toctree::
   :maxdepth: 2
   :caption: Gallery

   _auto_examples/index


.. toctree::
   :maxdepth: 2
   :caption: API

   api/cells
   api/costs
   api/OT

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
