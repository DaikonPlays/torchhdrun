.. _functional:

torchhd
=========================

.. currentmodule:: torchhd 

The main module contains the basic hypervector generation functions and operations on hypervectors.

Basis-hypervector sets
----------------------------------

.. autosummary:: 
    :toctree: generated/
    :template: function.rst

    identity_hv
    random_hv
    level_hv
    circular_hv


Bundle
--------------------
    
.. autosummary::
    :toctree: generated/
    :template: function.rst

    bundle
    multibundle
    set_bundle_method
    add
    badd
    randsel
    brandsel


Bind
--------------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    bind
    multibind
    set_bind_method
    mul
    bmul


Permute
--------------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    permute
    set_permute_method
    shift


Similarities
------------------------

.. autosummary::
    :toctree: generated/
    :template: function.rst
    
    cos_similarity
    dot_similarity
    ham_similarity
