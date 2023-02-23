import datetime
import json
import time
import traceback
import warnings

import logging
from os.path import join, isfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import h5py

from wand.image import Image


class Simulation:

    def __init__(self, fluid, membrane, dt, t=0, id=None, save_history=False, iterations=1000,
                       write=True, write_frequency=100, plot_frequency=100, data_format="csv", image_format="png"):
        self.fluid = fluid
        self.membrane = membrane
        self.dt = dt
        self.t = t
        self.iteration = 0
        self.cache = []
        self.current_step = None
        self.id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") if id is None else id
        self.save_history = save_history
        self.history = []
        self.warnings = []
        self.gif = None
        self.start_time = None
        self.end_time = None
        self.iterations = iterations
        self.write = write
        self.write_frequency = write_frequency
        self.plot_frequency = plot_frequency
        self.data_format = data_format
        self.image_format = image_format


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


    def initial_write(self):
        self.current_step = self.initial_step()
        self.write_data()
        self.write_plots()
        self.iteration += 1


    def perform_step(self):
        with warnings.catch_warnings(record=True) as w:
            self.current_step = self.step()
            if not (self.iteration % self.write_frequency):
                self.write_data()
            if not (self.iteration % self.plot_frequency):
                self.write_plots()
            self.warnings.extend(w)
            return self.current_step


    def perform_simulation(self):
        while self.iteration <= self.iterations:
           self.perform_step()
           self.iteration += 1
        if self.iteration > 1:
           self.write_data()
           self.write_plots()


    def setup(self):
       self.start_time = time.time()
       self.logger.info(f"Beginning simulation with fluid and membrane parameters: {self.parameters}")
       self.logger.info(f"Simulation parameters: {self.iterations=} {self.write_frequency=} {self.plot_frequency=} {self.data_format=}")
       self.prepare_filesystem()
       self.write_parameters()
       self.initial_write()


    def finish(self):
       self.end_time = time.time()
       self.logger.info(f"Ending simulation. Simulation took {self.end_time - self.start_time}s")


    def prepare_filesystem(self):
        if not self.write:
            return
        self.logger.debug(f"Setting up folder structure on the filesystem")
        artifacts = Path(f"artifacts")
        artifacts.mkdir(exist_ok=True)
        simulation = artifacts / Path(self.id)
        simulation.mkdir(exist_ok=True)


    def write_data(self, step=None):
        if not self.write:
            return
        if step is None:
            step = self.current_step
        step.iteration_path.mkdir(exist_ok=True)
        self.logger.info(f"Writing data files to disk for iteration#{step.iteration}")
        if self.data_format == "csv":
            step.write_csv()
        elif data_format == "hdf5":
            step.write_hdf5()
        else:
            raise Exception(f"Data format {self.data_format} not supported; must be one of ('csv', 'hdf5')")


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


    def write_plots(self, step=None, image_format="png"):
        if not self.write:
            return
        if step is None:
            step = self.current_step
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


    def __enter__(self):
        if self.write:
            self.gif = Image()
            self.setup()
        return self


    def __exit__(self, exception_type, exception_value, tb):
        if self.write:
            self.write_gif()
            self.gif.close()

        if exception_type:
            tb = '\n\r'.join(traceback.format_tb(tb))
            self.logger.error(f"Simulation failed due to an unhandled exception: \n\n {tb}")
            self.logger.error(f"Exiting")


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


   def write_csv(self):
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


   def write_hdf5(self):
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




