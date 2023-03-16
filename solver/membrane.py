from functools import cached_property

import numpy as np

from scipy.linalg import norm
from solver.ib_utils import interp_to_membrane

class Membrane:

    def __init__(self, X, Y, k, X_ref=None,
                       Y_ref=None, reference_kind="circle", fluid=None, p=2,
                       A=lambda x, t: 0
                 ):
        if reference_kind != "circle":
            raise Exception(f"Non-circular reference configurations aren't supported; got {reference_kind=}")
        self.Z = X + Y*1j
        self.X_ref = X if X_ref is None else X_ref
        self.Y_ref = Y if Y_ref is None else Y_ref
        self.Z_ref = self.X_ref + self.Y_ref*1j
        self.reference_kind = reference_kind
        self.k = k
        self.p = p
        self.fluid = fluid
        self.consistency_check()
        self.A_  = A


    @property
    def t(self):
        if self.fluid.simulation:
            return self.fluid.simulation.t
        return 0


    @property
    def reference_configuration(self):
        return {
            "X": list(self.X_ref),
            "Y": list(self.Y_ref)
        }


    @property
    def X(self):
        return self.Z.real


    @property
    def Y(self):
        return self.Z.imag


    @staticmethod
    def ellipse_area(X, Y):
        a = (X.max() - X.min())/2
        b = (Y.max() - Y.min())/2
        return np.pi*a*b


    def areas(self):
        membrane_area = self.ellipse_area(self.X, self.Y)
        reference_area = self.ellipse_area(self.X_ref, self.Y_ref)
        return membrane_area, reference_area, np.isclose(membrane_area, reference_area, atol=1e-5)


    def circle_consistency_check(self):
        membrane_area, reference_area, match = self.areas()
        if not match:
            print(f"Warning: membrane and reference areas don't match: {membrane_area=}, but {reference_area=}")
        else:
            print(f"Membrane and reference areas match")


    def consistency_check(self):
        if self.reference_kind == "circle":
            self.circle_consistency_check()


    def interp(self, f):
        return interp_to_membrane(f, self.fluid, self)


    def difference_minus(self, vec):
        shifted = np.roll(vec, 1)
        return vec - shifted


    def difference_plus(self, vec):
        shifted = np.roll(vec, -1)
        return shifted - vec


    @cached_property
    def delta_theta_plus(self):
        diff = self.difference_plus(self.Z_ref)
        return np.sqrt(diff.real**2 + diff.imag**2)


    @cached_property
    def delta_theta_minus(self):
        diff = self.difference_minus(self.Z_ref)
        return np.sqrt(diff.real**2 + diff.imag**2)


    @cached_property
    def delta_theta(self):
        return 0.5*(self.delta_theta_plus + self.delta_theta_minus)


    @property
    def delta_plus_Z(self):
        return self.difference_plus(self.Z)/self.delta_theta_plus


    @property
    def delta_minus_Z(self):
        return self.difference_minus(self.Z)/self.delta_theta_minus


    @property
    def norms_delta_plus_Z(self):
        return np.sqrt(self.delta_plus_Z.real**2 + self.delta_plus_Z.imag**2)


    @property
    def norms_delta_minus_Z(self):
        return np.sqrt(self.delta_minus_Z.real**2 + self.delta_minus_Z.imag**2)


    @property
    def tau_plus(self):
        delta_plus_Z = self.delta_plus_Z
        return delta_plus_Z/self.norms_delta_plus_Z


    @property
    def tau_minus(self):
        delta_minus_Z = self.delta_minus_Z
        return delta_minus_Z/self.norms_delta_minus_Z


    @property
    def Ap(self):
        return self.A_(self.delta_minus_Z, self.t)

    @property
    def Am(self):
        return self.A_(self.delta_minus_Z, self.t)


    @property
    def tension_plus(self):
        return self.k*(norm(self.delta_plus_Z) - 1) + self.Ap


    @property
    def tension_minus(self):
        return self.k*(norm(self.delta_minus_Z) - 1) + self.Am


    @property
    def F(self):
        return (self.tension_plus*self.tau_plus - self.tension_minus*self.tau_minus)/self.delta_theta
