from functools import cached_property

import distmesh as dm
import matplotlib.pyplot as plt
import numpy as np

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


def circle_mesh(r: float, q: np.ndarray, h=0.2):
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
        return points


    def area(self):
        area = 0
        for point in self.points:
            area += point.area()
        return area



class Point:

    def __init__(self, point_index, parent_mesh):
        self.index = point_index
        self.mesh = parent_mesh
        self._coordinates = None


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
                faces.append(
                    (self.index, adjacency, common.pop())
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


    def __repr__(self):
        return f"<Point at {self.coordinates}>"



if __name__ == '__main__':
    points, edges = circle_mesh(2.5, np.array([3, 4]))
    mesh = Mesh(points, edges)
    print(mesh.area())
