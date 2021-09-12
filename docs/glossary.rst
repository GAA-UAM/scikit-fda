.. _glossary:

========================
Glossary of Common Terms
========================

This glossary contains concepts and API elements specific to the functional
data analysis setting or for the package ``scikit-fda``. If the term you
are looking is not listed here, it may be a more generally applicable term
listed in the scikit-learn :ref:`sklearn:glossary`.

General Concepts
================

.. glossary::

    codomain
        The set of allowed output values of a function.
        Note that the set of actual output values, called the :term:`image`,
        can be a strict subset of the :term:`codomain`.

    curve
    trajectory
        A :term:`functional data object` whose domain and codomain are both
        the set of real numbers (:math:`\mathbb{R}`).
        Thus, its ``dim_domain`` and ``dim_codomain`` attributes shall both
        be 1.
        
    domain
        The set of possible input values of a function.
        
    domain range
    	The valid range where a function can be evaluated. It is a Python
    	sequence that contains, for each dimension of the domain, a tuple with
    	the minimum and maximum values for that dimension. Usually used in
    	plotting functions and as the domain of integration for this function.
        
    FDA
    Functional Data Analysis
    	The branch of statistics that deals with curves, surfaces or other
    	objects varying over a a continuum (:term:`functional data objects`).

    functional data
    	A collection of :term:`functional data objects`.
    	Represented by a :class:`~skfda.representation.FData` object.

    functional data object
    functional data objects
    functional object
    functional objects
    	An object of study of Functional Data Analysis.
    	It is a function between :math:`\mathbb{R}^p` and :math:`\mathbb{R}^q`.
    	Usually represented by a :class:`~skfda.representation.FData` object of
    	length 1, but in some cases regular Python
    	:term:`callables <sklearn:callable>` are also accepted.
    
    functional observation
    functional observations
        An observed :term:`functional data object`, represented as a
        :class:`~skfda.representation.FData` object of length 1.
    
    image
        The set of actual ouput values that a function takes.
        It must be a (non necessarily strict) subset of the :term:`codomain`.
        
    multivariate functional data
    	Often used for :term:`functional data` where each
    	:term:`functional data object` is a :term:`vector-valued function`.
        
    multivariate object
    multivariate objects
    	An object of study of multivariate statistics.
    	It is a vector of possibly related variables, represented
    	as a :term:`sklearn:1d array`.
    	
    operator
    operators
        Function whose :term:`domain` is a set of functions.
	
    vector-valued function
    	A :term:`functional data object` that outputs vectors, that is, its
    	:term:`codomain` has dimension greater than 1.
