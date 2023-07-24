from functools import cached_property

import distmesh as dm
import matplotlib.pyplot as plt
import numpy as np

from numpy.linalg import inv

from solver.fluid import Fluid
from solver.ib_utils import spread, interp

MU = 1
K = 1

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
        self.faces_seen = {}


    @property
    def faces(self):
        return list(self.faces_seen.values())


    @property
    def X(self):
        return np.array([p.x for p in self.points])


    @property
    def Y(self):
        return np.array([p.y for p in self.points])


    @property
    def dA(self):
        return np.array([p.area() for p in self.points])


    @property
    def adjacencies(self):
        return adjacencies(self._edges)


    @cached_property
    def points(self):
        points = []
        for point in self.adjacencies:
            point_index = point
            points.append(
                Point(
                    point_index, self
                )
            )
        points = sorted(points, key=lambda z: z.index)
        return np.array(points)
    

    def face_points(self, indices):
        one, two, three = indices
        return [self.points[one], self.points[two], self.points[three]]


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
            face = Face(indices, self)
            self.faces_seen[indices] = face
            return face
        return self.faces_seen[indices]


    def calculate_point_forces(self):
        for face in self.faces:
            face.calculate_point_forces()


    @property
    def forces(self):
        return np.array([np.array(p.force) for p in self.points])
    

    @property
    def forcesX(self):
        return np.array([force[0] for force in self.forces])
    
    @property
    def forcesY(self):
        return np.array([force[1] for force in self.forces])


    @property
    def reference_x(self):
        return np.array([face.reference_coordinates[0] for face in self.faces])


    @property
    def reference_y(self):
        return np.array([face.reference_coordinates[1] for face in self.faces])

    

    def initialize(self):
        for point in self.points:
            point.find_faces()



class Face:

    def __init__(self, indices, mesh):
        self.indices = indices
        self.mesh = mesh
        self._points = None
        self._ref = self.calculate_coordinates()


    @property
    def points(self):
        return self.mesh.face_points(self.indices)
    

    @points.setter
    def points(self, value):
        self.points = value


    def calculate_coordinates(self):
        return [np.copy(point.coordinates) for point in self.points]


    @property
    def coordinates(self):
        return self.calculate_coordinates()


    @property
    def reference_coordinates(self):
        return self._ref


    def calculate_area(self):
        return calculate_area(*self.coordinates)


    def area(self):
        return self.calculate_area()
    

    def __repr__(self):
        return f"<Face with vertices {self.coordinates}>"


    @property
    def x1(self):
        return self.coordinates[1] - self.coordinates[0]


    @property
    def x2(self):
        return self.coordinates[2] - self.coordinates[0]


    @property
    def s1(self):
        return self.reference_coordinates[1] - self.reference_coordinates[0]


    @property
    def s2(self):
        return self.reference_coordinates[2] - self.reference_coordinates[0]


    @property
    def s(self):
        return np.column_stack([self.s1, self.s2])


    @property
    def x(self):
        return np.column_stack([self.x1, self.x2])


    @property
    def A(self):
        return self.x@inv(self.s)


    @property
    def delta(self):
        return self.s[0, 0]*self.s[1, 1] - self.s[1,0]*self.s[0,1]
    

    @property
    def J(self):
        return np.linalg.det(self.A)


    @property
    def dAdx1_1(self):
        return (1/self.delta)*np.array([
            [self.s2[1], -self.s2[0]],
            [0, 0]
        ])

    @property
    def dAdx1_2(self):
        return (1/self.delta)*np.array([
            [0, 0],
            [self.s2[1], -self.s2[0]]
        ])


    @property
    def dAdx2_1(self):
        return (1/self.delta)*np.array([
            [-self.s1[1], self.s1[0]],
            [0, 0]
        ])

    @property
    def dAdx2_2(self):
        return (1/self.delta)*np.array([
            [0, 0],
            [-self.s1[1], self.s1[0]]
        ])

    @property
    def dAdx0_1(self):
        return (1/self.delta)*np.array([
            [self.s1[1] - self.s2[1], self.s2[0] - self.s1[0]],
            [0, 0]
        ])


    @property
    def dAdx0_2(self):
        return (1/self.delta)*np.array([
            [0, 0],
            [self.s1[1] - self.s2[1], self.s2[0] - self.s1[0]]
        ])


    def dAdx1(self, i):
        return [self.dAdx1_1, self.dAdx1_2][i-1]


    def dAdx2(self, i):
        return [self.dAdx2_1, self.dAdx2_2][i-1]


    def dAdx0(self, i):
        return [self.dAdx0_1, self.dAdx0_2][i-1]
    

    def dAdx(self, k, ell):
        return [
            self.dAdx0(ell),
            self.dAdx1(ell),
            self.dAdx2(ell)
        ][k]

    

    @property
    def dWdA(self):
        invAT = inv(self.A.T)
        term_1 = MU*(self.A/self.J - np.trace(self.A*self.A.T)*invAT/(2*self.J))
        term_2 = K*(self.J - 1)*self.J*invAT
        return term_1 + term_2
    

    def calculate_point_forces(self):
        dWdA = self.dWdA
        for k in (0, 1, 2):
            for ell in (1, 2):
                component_force = 0
                component_force += (dWdA*self.dAdx(k, ell)).sum()
                component_force *= -1/self.area() #self.points[k].area()
                self.points[k].force[ell-1] += component_force
                


class Point:

    def __init__(self, point_index, parent_mesh):
        self.index = point_index
        self.mesh = parent_mesh
        self._coordinates = None
        self.force = [0, 0]
        self._faces = []


    def add_force(self, ell, force_value):
        self.force[ell - 1] += force_value
        print(self.force, force_value)

    @property
    def x(self):
        return self.coordinates[0]

    @property
    def y(self):
        return self.coordinates[1]

    @property
    def coordinates(self):
        return self.mesh.coordinates[self.index]
    

    @property
    def setter(self, other):
        self.mesh.coordinates[self.index] = other

    

    @property
    def coordinates_tuple(self):
        return tuple(self.coordinates)


    @property
    def adjacencies(self):
        return self.mesh.adjacencies[self.index]


    def find_faces(self):
        faces = []
        for adjacency in self.adjacencies:
            adjacency_adjacencies = self.mesh.adjacencies[adjacency]
            common = adjacency_adjacencies.intersection(self.adjacencies)
            if common:
                # There should be one node in common per adjacency
                face_indices = (self.index, adjacency, common.pop())
                face = self.mesh.register_face(face_indices)
                self._faces.append(
                    face
                )
        return faces
    

    @property
    def faces(self):
        if not self._faces:
            self._faces = self.find_faces()
        return self._faces


    def area(self):
        area = 0
        for face in self.faces:
            area += (1/3)*face.area()
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
    


def compression_test(mesh):
    fig, ax = plt.subplots()
    coordinates = [p.coordinates for p in mesh.points]
    foo = np.array(coordinates)
    ax.plot(foo[:,0],foo[:,1],'or')


    for point in mesh.points:
        shift = np.array([0.5, 0.5])
        mesh.coordinates[point.index] -= shift
        mesh.coordinates[point.index][0] *= 0.9
        mesh.coordinates[point.index] += shift

    mesh.calculate_point_forces()


    X = np.linspace(0, 1, 1000)
    Y = np.linspace(0, 1, 1000)
    xv, yv = np.meshgrid(X, Y)
    coordinates = [p.coordinates for p in mesh.points]
    foo = np.array(coordinates)
    bar = np.array(mesh.forces)
    ax.plot(foo[:,0],foo[:,1],'ob')
    ax.quiver(foo[:,0],foo[:,1], bar[:,0],bar[:,1])
    fig.show()


def shear_test(mesh):
    fig, ax = plt.subplots()
    coordinates = [p.coordinates for p in mesh.points]
    foo = np.array(coordinates)
    ax.plot(foo[:,0],foo[:,1],'or')


    for point in mesh.points:
        shift = np.array([0.5, 0.5])
        mesh.coordinates[point.index] -= shift
        mesh.coordinates[point.index][0] += mesh.coordinates[point.index][0] + 2*mesh.coordinates[point.index][1]
        mesh.coordinates[point.index] += shift

    mesh.calculate_point_forces()


    X = np.linspace(0, 1, 1000)
    Y = np.linspace(0, 1, 1000)
    xv, yv = np.meshgrid(X, Y)
    coordinates = [p.coordinates for p in mesh.points]
    foo = np.array(coordinates)
    bar = np.array(mesh.forces)
    ax.plot(foo[:,0],foo[:,1],'ob')
    ax.quiver(foo[:,0],foo[:,1], bar[:,0],bar[:,1])
    fig.show()


def spread_test(mesh):
    for point in mesh.points:
        shift = np.array([0.5, 0.5])
        mesh.coordinates[point.index] -= shift
        mesh.coordinates[point.index][0] += mesh.coordinates[point.index][0] + 2*mesh.coordinates[point.index][1]
        mesh.coordinates[point.index] += shift

    mesh.calculate_point_forces()
    X = np.linspace(0, 1, 55)
    Y = np.linspace(0, 1, 55)
    xv, yv = np.meshgrid(X, Y)
    spread_force = spread(
        np.sqrt((mesh.forces[:,0])**2 + (mesh.forces[:,1])**2),
        xv,
        yv,
        mesh.X,
        mesh.Y,
        mesh.dA
    )
    fig, ax = plt.subplots()
    X = np.linspace(0, 1, 55)
    Y = np.linspace(0, 1, 55)
    xv, yv = np.meshgrid(X, Y)
    coordinates = [p.coordinates for p in mesh.points]
    foo = np.array(coordinates)
    bar = np.array(mesh.forces)
    ax.plot(foo[:,0],foo[:,1],'ob')
    ax.quiver(X,Y, spread_force[:,0],spread_force[:,1])
    fig.show()


def run_tests():
    print("Building mesh")
    points, edges = circle_mesh(0.4, np.array([0.5, 0.5]), h=0.1)
    print("Points and edges determined, building mesh object")
    for test in [compression_test, shear_test, spread_test]:
        mesh = Mesh(points, edges)
        mesh.initialize()
        test(mesh)



if __name__ == '__main__':
    print("Building mesh")
    points, edges = circle_mesh(0.4, np.array([0.5, 0.5]), h=0.1)
    print("Points and edges determined, building mesh object")
    mesh = Mesh(points, edges)
    mesh.initialize()

    for point in mesh.points:
        shift = np.array([0.5, 0.5])
        mesh.coordinates[point.index] -= shift
        mesh.coordinates[point.index][0] += mesh.coordinates[point.index][0] + 2*mesh.coordinates[point.index][1]
        mesh.coordinates[point.index] += shift

    mesh.calculate_point_forces()
    X = np.linspace(0, 1, 55)
    Y = np.linspace(0, 1, 55)
    xv, yv = np.meshgrid(X, Y)
    spread_force = spread(
        np.sqrt((mesh.forces[:,0])**2 + (mesh.forces[:,1])**2),
        xv,
        yv,
        mesh.X,
        mesh.Y,
        mesh.dA
    )
    fig, ax = plt.subplots()
    X = np.linspace(0, 1, 55)
    Y = np.linspace(0, 1, 55)
    xv, yv = np.meshgrid(X, Y)
    coordinates = [p.coordinates for p in mesh.points]
    foo = np.array(coordinates)
    bar = np.array(mesh.forces)
    ax.plot(foo[:,0],foo[:,1],'ob')
    ax.quiver(X,Y, spread_force[:,0],spread_force[:,1])
    fig.show()