ANFIS
=======

anfis is a Python implementation of an Adaptive Neuro Fuzzy Inference System.

This ANFIS package is essentially a Python refactoring of the R code created
by the team a the BioScience Data Mining Group, the original documentaion of
which can be found here:

http://www.bdmg.com.ar/?page_id=176

As an exmaple of an ANFIS system, this Python code works (install and run the
tests.py script to see it fit the some test data) but there is much left to do
in order to improve the project.  This is very much an early beta version.
My python code is cruddy and NOT at all idiomatic. Documentation and doc strings
need large amounts of work.

All useful contributions to make this a better project will be happily received.


Contributions
============

If you would like to contribute, please see the Issues section for ideas
about what most needs attention.

Features
========

* Currently the implementation will support the use of three types of
membership function:

* gaussmf: Gaussian
* gbellmf: Generalized bell
* sigmf: Sigmoid

This naming is taken from scikit-fuzzy, a fuzzy logic toolkit for SciPy,
which can be found here: https://github.com/scikit-fuzzy/scikit-fuzzy

Each input variable can have an arbitrary number and mix of these membership
functions.

* A user can define the number of epochs that will be run

* The returned ANFIS object can plot training errors, fitted results and
the current shape of its membership functions (pre or post training)


Installation
============

anfis may then be installed by running:

    $ pip install anfis


Dependencies
------------

* Python
* numpy
* scikit-fuzzy
* matplotlib


Quickstart
==========

Install anfis and navigate to the location of anfis/tests.py

From the command line run:
```
python tests.py
```
Alternatively, from the same location launch ipython and run:
```
run tests.py
```

This will set up and fit an ANFIS model based on the data contained
in 'trainingSet.txt', using 10 epochs.  Plots of the fitting errors
and the model predicted output are graphed.



Contact
=======

For other questions, please contact <twmeggs@gmail.com>.