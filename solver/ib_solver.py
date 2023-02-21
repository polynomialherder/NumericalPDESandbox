import datetime
import json
import time

from dataclasses import dataclass
from functools import cached_property
from itertools import product
import logging
from math import floor
from os.path import join, isfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import h5py

from scipy.linalg import norm
from solver.ib_utils import spread_to_fluid, interp_to_membrane
from solver.stokes import StokesSolver
from wand.image import Image


class Simulation:

    def __init__(self, fluid, membrane, dt, t=0, id=None, mu=1, save_history=False):
        """ Write a toplevel parameters file for the run
            sizes of arrays
            mu
        """
        self.fluid = fluid
        self.membrane = membrane
        self.dt = dt
        self.mu = mu
        self.t = t
        self.iteration = 1
        self.cache = []
        self.current_step = None
        self.id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") if id is None else id
        self.current_step = None
        self.save_history = save_history
        self.history = []
        self.gif = Image()


    @property
    def logger(self):
        return logging.getLogger(f"Simulation#{self.id}")


    def append_gifimage(self, path):
        if not isfile(path):
            return
        with Image(filename=path) as im:
            self.gif.sequence.append(im)


    def write_gif(self):
        self.gif.type = 'optimize'
        self.gif.save(filename=f"artifacts/{self.id}/MembranePositions.gif")
        self.gif.close()


    def initial_step(self):
        fluid_shape = self.fluid.xv.shape
        fluid_zeros = np.zeros(fluid_shape)
        membrane_zeros = np.zeros(self.membrane.X.shape)
        return SimulationStep(xv=self.fluid.xv, yv=self.fluid.yv, fx=fluid_zeros, fy=fluid_zeros,
                    Fx=membrane_zeros, Fy=membrane_zeros, X=self.membrane.X, Y=self.membrane.Y,
                    X_ref=self.membrane.X_ref, Y_ref=self.membrane.Y_ref, t=self.t, p=fluid_zeros,
                    u=fluid_zeros, v=fluid_zeros, U=membrane_zeros, V=membrane_zeros,
                    simulation_id=self.id, iteration=self.iteration, simulation=self)


    def initial_write(self, data_format="csv", image_format="png"):
        step = self.initial_step()
        self.write_data(step, data_format)
        self.write_plots(step, image_format)
        self.iteration += 1

    def perform_simulation(self, iterations=1000, write_frequency=100, plot_frequency=100, data_format="csv", image_format="png"):
       simulation_start = time.time()
       self.logger.info(f"Beginning simulation with fluid and membrane parameters: {self.parameters}")
       self.logger.info(f"Simulation parameters: {iterations=} {write_frequency=} {plot_frequency=} {data_format=}")
       self.prepare_filesystem()
       self.write_parameters()
       self.initial_write(data_format=data_format, image_format=image_format)
       while self.iteration <= iterations:
          step = self.step()
          if not (self.iteration % write_frequency):
             self.write_data(step, data_format)
          if not (self.iteration % plot_frequency):
             self.write_plots(step, image_format)
          self.iteration += 1
       if self.iteration > 1:
           self.write_data(step, data_format)
           self.write_plots(step, image_format)
           self.write_gif()
       simulation_end = time.time()
       self.logger.info(f"Ending simulation. Simulation took {simulation_end - simulation_start}s")


    def prepare_filesystem(self):
        self.logger.debug(f"Setting up folder structure on the filesystem")
        artifacts = Path(f"artifacts")
        artifacts.mkdir(exist_ok=True)
        simulation = artifacts / Path(self.id)
        simulation.mkdir(exist_ok=True)


    def write_data(self, step, data_format):
        step.iteration_path.mkdir(exist_ok=True)
        self.logger.info(f"Writing data files to disk for iteration#{step.iteration}")
        if data_format == "csv":
            step.write_csv(self.id)
        elif data_format == "hdf5":
            step.write_hdf5(self.id)
        else:
            raise Exception(f"Data format {data_format} not supported; must be one of ('csv', 'hdf5')")


    @property
    def parameters(self):
        return {
            "run_id": self.id,
            "dt": self.dt,
            "fluid": {
                "mu": self.fluid.mu,
                "shape": list(self.fluid.xv.shape)
            },
            "membrane": {
                "k": self.membrane.k,
                "shape": list(self.fluid.membrane.X.shape),
                "reference_type": self.fluid.membrane.reference_kind,
                "reference_configuration": self.fluid.membrane.reference_configuration
            }
        }


    def write_parameters(self):
        with open(f"artifacts/{self.id}/parameters.json", "w") as f:
            json.dump(self.parameters, f)


    def write_plots(self, step, image_format="png"):
        self.logger.info(f"Writing plots to disk for {step.iteration=}")
        step.iteration_path.mkdir(exist_ok=True)
        step.plots_path.mkdir(exist_ok=True)
        step.generate_plots(image_format=image_format)


    def calculate_forces(self):
        F = self.membrane.F
        return F.real, F.imag

    def spread_forces(self, Fx, Fy):
        return self.fluid.spread(Fx), self.fluid.spread(Fy)

    def stokes_solve(self, fx, fy):
        return self.fluid.stokes_solve(fx, fy)

    def calculate_velocities(self, u, v):
        return self.membrane.interp(u), self.membrane.interp(v)

    def update_membrane_positions(self, U, V):
        self.membrane.Z.real += self.dt*U
        self.membrane.Z.imag += self.dt*V

    def step(self):
        Fx, Fy = self.calculate_forces()
        fx, fy = self.spread_forces(Fx, Fy)
        u, v, p = self.stokes_solve(fx, fy)
        U, V = self.calculate_velocities(u, v)
        self.update_membrane_positions(U, V)
        self.t += self.dt
        step = SimulationStep(
           xv=self.fluid.xv, yv=self.fluid.yv, fx=fx, fy=fy, X=self.membrane.X,
           Y=self.membrane.Y, X_ref=self.membrane.X_ref, Y_ref=self.membrane.Y_ref,
           Fx=Fx, Fy=Fy, t=self.t, p=p, u=u, v=v, U=U, V=V,
           iteration=self.iteration, simulation_id=self.id, simulation=self
        )
        if self.save_history:
            self.history.append(step)
        self.current_step = step
        return step


    def save(self, filename="data.hdf5"):
        with h5py.File(filename, "w") as f:
            f.create_dataset("Membrane Positions: X", data=self.membrane.X)
            f.create_dataset("Membrane Positions: Y", data=self.membrane.Y)
            f.create_dataset("Fluid Pressure Field", data=self.fluid.solver.p)
            f.create_dataset("t", data=self.t)


class SimulationStep:

   def __init__(self, *, xv=None, yv=None, fx=None, fy=None,
                Fx=None, Fy=None, X=None, Y=None, X_ref=None,  Y_ref=None,
                t=None, p=None, u=None, v=None, U=None, V=None,
                simulation_id=None, iteration=None, simulation=None):
      self.xv = xv
      self.yv = yv
      self.fx = fx
      self.fy = fy
      self.Fx = Fx
      self.Fy = Fy
      self.X = X
      self.Y = Y
      self.X_ref = X_ref
      self.Y_ref = Y_ref
      self.t = t
      self.p = p
      self.u = u
      self.v = v
      self.U = U
      self.V = V
      self.simulation_id = simulation_id
      self.iteration = iteration
      self.simulation = simulation


   def plot_membrane_positions(self, to_file=None):
      fig, ax = plt.subplots()
      ax.plot(self.X, self.Y, 'o', label="Membrane positions")
      ax.plot(self.X_ref, self.Y_ref, "x", label="Reference configuration")
      ax.set_title(f"t={self.t:.3f}")
      ax.legend()
      ax.set_xlim(0, 1)
      ax.set_ylim(0, 1)
      if to_file:
         fig.savefig(to_file)
      else:
         fig.show()
      plt.close(fig)
      self.simulation.append_gifimage(to_file)


   def plot_pressure(self, to_file=None):
       fig, ax = plt.subplots()
       cm = ax.pcolor(self.xv, self.yv, self.p)
       ax.set_title(f"t={self.t:.3f}")
       ax.set_xlim(0, 1)
       ax.set_ylim(0, 1)
       fig.colorbar(cm)
       if to_file:
           fig.savefig(to_file)
       else:
           fig.show()
       plt.close(fig)


   def plot_lag_force(self, to_file=None):
       fig, ax = plt.subplots()
       ax.quiver(self.X, self.Y, self.Fx, self.Fy)
       ax.set_title(f"t={self.t:.3f}")
       ax.set_xlim(0, 1)
       ax.set_ylim(0, 1)
       if to_file:
          fig.savefig(to_file)
       else:
          fig.show()
       plt.close(fig)



   def plot_eul_force(self, to_file=None):
       fig, ax = plt.subplots()
       ax.quiver(self.xv, self.yv, self.fx, self.fy)
       ax.set_title(f"t={self.t:.3f}")
       ax.set_xlim(0, 1)
       ax.set_ylim(0, 1)
       if to_file:
           fig.savefig(to_file)
       else:
           fig.show()
       plt.close(fig)



   def plot_lag_vel(self, to_file=None):
       fig, ax = plt.subplots()
       ax.quiver(self.X, self.Y, self.U, self.V)
       ax.set_title(f"t={self.t:.3f}")
       ax.set_xlim(0, 1)
       ax.set_ylim(0, 1)
       if to_file:
           fig.savefig(to_file)
       else:
           fig.show()
       plt.close(fig)


   def plot_eul_vel(self, to_file=None):
       fig, ax = plt.subplots()
       ax.quiver(self.xv, self.yv, self.u, self.v)
       ax.set_title(f"t={self.t}")
       ax.set_xlim(0, 1)
       ax.set_ylim(0, 1)
       if to_file:
           fig.savefig(to_file)
       else:
           fig.show()
       plt.close(fig)


   def generate_plots(self, image_format="png"):
       self.plot_membrane_positions(self.plots_path / f"membrane_positions.{image_format}")
       self.plot_pressure(self.plots_path / f"pressure.{image_format}")
       self.plot_lag_force(self.plots_path / f"lag_forces.{image_format}")
       self.plot_eul_force(self.plots_path / f"eul_forces.{image_format}")
       self.plot_lag_vel(self.plots_path / f"lag_vel.{image_format}")
       self.plot_eul_vel(self.plots_path / f"eul_vel.{image_format}")


   @property
   def artifacts_path(self):
       return Path(f"artifacts/{self.simulation_id}/")


   @property
   def iteration_path(self):
       return self.artifacts_path / Path(f"{self.iteration}")


   @property
   def plots_path(self):
       return self.iteration_path / "plots"


   def write_csv(self, simulation_id=None):
       self.iteration_path.mkdir(exist_ok=True)
       np.savetxt(self.iteration_path / "fx.csv", self.fx, delimiter=",")
       np.savetxt(self.iteration_path / "fy.csv", self.fy, delimiter=",")
       np.savetxt(self.iteration_path / "Fx.csv", self.Fx, delimiter=",")
       np.savetxt(self.iteration_path / "Fy.csv", self.Fy, delimiter=",")
       np.savetxt(self.iteration_path / "X.csv", self.X, delimiter=",")
       np.savetxt(self.iteration_path / "Y.csv", self.Y, delimiter=",")
       np.savetxt(self.iteration_path / "p.csv", self.p, delimiter=",")
       np.savetxt(self.iteration_path / "u.csv", self.u, delimiter=",")
       np.savetxt(self.iteration_path / "v.csv", self.v, delimiter=",")
       np.savetxt(self.iteration_path / "t.csv", np.array([self.t]), delimiter=",")


   def write_hdf5(self, simulation_id=None):
       self.artifacts_path.mkdir(exist_ok=True)
       with h5py.File(self.artifacts_path / "data.hdf5", "w") as f:
          group = f.create_group(f"{self.iteration}")
          group.create_dataset("fx", data=self.fx)
          group.create_dataset("fy", data=self.fy)
          group.create_dataset("X", data=self.X)
          group.create_dataset("Y", data=self.Y)
          group.create_dataset("p", data=self.p)
          group.create_dataset("u", data=self.u)
          group.create_dataset("v", data=self.v)
          group.create_dataset("t", data=self.t)


class Fluid:

    def __init__(self, xv, yv, mu=1, membrane=None):
        self.xv = xv
        self.yv = yv
        self.mu = mu
        self.membrane = membrane
        self.solver = StokesSolver(self.xv, self.yv, mu=mu)

    def register(self, membrane):
        self.membrane = membrane
        self.membrane.fluid = self

    def stokes_solve(self, fx, fy):
        self.solver.F = -fx
        self.solver.G = -fy
        return self.solver.u, self.solver.v, self.solver.p

    @property
    def shape(self):
        return self.xv.shape

    def spread(self, F):
        return spread_to_fluid(F, self, self.membrane)

    def __repr__(self):
        return f"<Fluid mu={self.mu}>"


class Membrane:

    def __init__(self, X, Y, k, X_ref=None, Y_ref=None, reference_kind="circle", fluid=None, p=2):
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
    def tau_plus(self):
        delta_plus_Z = self.delta_plus_Z
        norms = np.sqrt(delta_plus_Z.real**2 + delta_plus_Z.imag**2)
        return delta_plus_Z/norms


    @property
    def tau_minus(self):
        delta_minus_Z = self.delta_minus_Z
        norms = np.sqrt(delta_minus_Z.real**2 + delta_minus_Z.imag**2)
        return delta_minus_Z/norms


    @property
    def tension_plus(self):
        return self.k*norm(self.delta_plus_Z)


    @property
    def tension_minus(self):
        return self.k*norm(self.delta_minus_Z)


    @property
    def F(self):
        return self.k*(self.tension_plus*self.tau_plus - self.tension_minus*self.tau_minus)/self.delta_theta
