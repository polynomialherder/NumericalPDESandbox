# Numerical PDE Sandbox

This is a sandbox for playing with concepts in numerical solution of partial differential equations as I work through Finite Difference Methods for Ordinary and Partial Differential Equations by Randall J. Leveque.

It consists of the following modules

- *simple_second_order.py* - a set of abstractions for solving a simple linear second order differential equation of the form `f(x) = u''(x)`
- *sandbox.py* - a scratchpad for testing simple ideas

This repository does not currently expose any APIs or UIs. I presently interact with it by doing `python -i script.py` (where `script.py` is one of the modules in this repository) which executes any toplevel code in `script.py` and loads all of the definitions into a Python interpreter session.
