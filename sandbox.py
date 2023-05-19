from functools import cached_property

import distmesh as dm
import matplotlib.pyplot as plt
import numpy as np

from numpy.linalg import inv

from solver.fluid import Fluid
from solver.ib_utils import spread, interp

"""
Structured vs unstructured
Make the force 1
Spread it to the fluid grid
Should get a little plateau (approximately 1 in the place where there is mesh, and drops off to 0 in the place where there is not mesh)
"""

def meshplot(p, t, write_location=None):
    fig, ax = plt.subplots()
    # PyDistMesh appears to access this deprecated property
    # from matplotlib.Axes, so we initialize it to keep it from
    # throwing an exception
    ax._hold = True
    dm.plotting.axes_simpplot2d(ax, p, t)
    if not write_location:
        fig.show()
    else:
        fig.savefig(write_location)


def circle_mesh(r: float, q: np.ndarray, h=0.1):
    def distance(x):
        return np.sqrt(((x - q)**2).sum(1)) - r
    bbox = (q[0] - r, q[1] - r, q[0] + r, q[1] + r)
    return dm.distmesh2d(distance, dm.huniform, h, bbox)


def calculate_area(point1, point2, point3):
    (x1, y1), (x2, y2), (x3, y3) = point1, point2, point3
    area = abs(0.5*(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)))
    return area


def adjacencies(connections):
    adjacencies = {}
    for point1_idx, point2_idx, point3_idx in connections:
        if point1_idx not in adjacencies:
            adjacencies[point1_idx] = set()
        if point2_idx not in adjacencies:
            adjacencies[point2_idx] = set()
        if point3_idx not in adjacencies:
            adjacencies[point3_idx] = set()
        adjacencies[point1_idx].add(point2_idx)
        adjacencies[point1_idx].add(point3_idx)
        adjacencies[point2_idx].add(point1_idx)
        adjacencies[point2_idx].add(point3_idx)
        adjacencies[point3_idx].add(point1_idx)
        adjacencies[point3_idx].add(point2_idx)

    return adjacencies


class Mesh:

    def __init__(self, coordinates, edges):
        self.coordinates = coordinates
        self._edges = edges
        self._ref = None
        self.faces_seen = set()
        self.faces = []


    @property
    def X(self):
        return np.array([p.x for p in self.points])


    @property
    def Y(self):
        return np.array([p.y for p in self.points])


    @property
    def dA(self):
        return np.array([p.area() for p in self.points])


    @cached_property
    def adjacencies(self):
        return adjacencies(self._edges)


    @cached_property
    def points(self):
        points = []
        for point in self.adjacencies:
            point_index = point
            coordinates = self.coordinates[point]
            adjacencies = self.adjacencies[point]
            points.append(
                Point(
                    point_index, self
                )
            )
        points = sorted(points, key=lambda z: z.index)
        return np.array(points)


    def area(self):
        area = 0
        for point in self.points:
            area += point.area()
        return area


    @staticmethod
    def standardize(indices):
        return tuple(sorted(list(indices)))


    def register_face(self, indices):
        indices = self.standardize(indices)
        if indices not in self.faces_seen:
            self.faces.append(
                Face(indices, self)
            )
            self.faces_seen.add(indices)



class Face:

    def __init__(self, indices, mesh):
        self.indices = indices
        self.mesh = mesh
        self._ref = None


    def calculate_coordinates(self):
        coordinates = []
        for index in self.indices:
            coordinates.append(
                self.mesh.points[index].coordinates
            )
        # coordinates[:] returns a copy of coordinates
        # simply assigning ref to coordinates would result
        # in a pointer to the same object
        self._ref = [np.array(c, copy=True) for c in coordinates]
        return coordinates


    @cached_property
    def coordinates(self):
        return self.calculate_coordinates()


    @cached_property
    def reference_coordinates(self):
        if self._ref is None:
            self.calculate_coordinates()
        return self._ref


    @cached_property
    def area(self):
        return calculate_area(*self.coordinates)


    def __repr__(self):
        return f"<Face with vertices {self.coordinates}>"


    @cached_property
    def s1(self):
        return self.coordinates[1] - self.coordinates[0]


    @cached_property
    def s2(self):
        return self.coordinates[2] - self.coordinates[0]


    @property
    def x1(self):
        return self.reference_coordinates[1] - self.reference_coordinates[0]


    @property
    def x2(self):
        return self.reference_coordinates[2] - self.reference_coordinates[0]


    @cached_property
    def s(self):
        return np.column_stack([self.s1, self.s2])


    @property
    def x(self):
        return np.column_stack([self.x1, self.x2])


    @property
    def A(self):
        return self.x@inv(self.s)



class Point:

    def __init__(self, point_index, parent_mesh):
        self.index = point_index
        self.mesh = parent_mesh
        self._coordinates = None

    @property
    def x(self):
        return self.coordinates[0]

    @property
    def y(self):
        return self.coordinates[1]

    @cached_property
    def coordinates(self):
        return self.mesh.coordinates[self.index]


    @cached_property
    def adjacencies(self):
        return self.mesh.adjacencies[self.index]


    @cached_property
    def faces(self):
        faces = []
        for adjacency in self.adjacencies:
            adjacency_adjacencies = self.mesh.adjacencies[adjacency]
            common = adjacency_adjacencies.intersection(self.adjacencies)
            if common:
                # There should be one node in common per adjacency
                face = (self.index, adjacency, common.pop())
                self.mesh.register_face(face)
                faces.append(
                    face
                )
        return faces


    def area(self):
        area = 0
        for face in self.faces:
            coordinates = []
            for idx in face:
                coordinates.append(self.mesh.points[idx].coordinates)
            face_area = calculate_area(*coordinates)
            area += (1/3)*face_area
        return area


    def __add__(self, other):
        return self.coordinates + other


    def __mul__(self, other):
        return self.coordinates*other


    def __truediv__(self, other):
        return self.coordinates/other

    def __exp__(self, other):
        return self.coordinates**other

    def __repr__(self):
        return f"<Point at {self.coordinates}>"


if __name__ == '__main__':
    print("Building mesh")
    points, edges = circle_mesh(0.4, np.array([0.5, 0.5]), h=0.01)
    print("Points and edges determined, building mesh object")
    mesh = Mesh(points, edges)
    X = np.linspace(0, 1, 20)
    Y = np.linspace(0, 1, 20)
    xv, yv = np.meshgrid(X, Y)
    lagrange_force = np.ones(mesh.dA.shape)
    spread_force = spread(
        lagrange_force,
        xv,
        yv,
        mesh.X,
        mesh.Y,
        mesh.dA
    )
    print(f"Checking that the face for each A is close to the identity matrix")
    close_to_identity_matrix = np.all(np.isclose([f.A for f in mesh.faces], np.array([[1, 0], [0, 1]])))
    print(f"If A is approximately the identity matrix, this should be True: {close_to_identity_matrix=}")
    print(f"Now shifting all points in the mesh 2 units to the right")
    for point in mesh.points:
        point += 2
    print(f"Checking that the face for each A is still close to the identity matrix")
    close_to_identity_matrix = np.all(np.isclose([f.A for f in mesh.faces], np.array([[1, 0], [0, 1]])))
    print(f"If A is still approximately the identity matrix, this should be True: {close_to_identity_matrix=}")
