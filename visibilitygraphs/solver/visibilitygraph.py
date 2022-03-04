from .solver import Solver
from visibilitygraphs.raytrace import RayTracer
from visibilitygraphs.transformation import makeRotation, makeTransformation
import numpy as np
import pyvista as pv
import scipy.spatial


APPROX_ZERO = .0001


class VisibilityGraph(Solver):
    def __init__(self, targetSize:float=10, depth:int=3, marginRatio:float=2) -> None:
        self.rayTracer = RayTracer()
        self.targetSize = targetSize
        self.depth = depth
        self.marginRatio = marginRatio
        super().__init__()
    
    def solve(self, q0: np.ndarray, q1: np.ndarray, radius: float, flightAngle: float, environment: pv.PolyData):
        l = self.targetSize / 2
        bounds = []
        for i in range(3):
            bounds = bounds + [q1[0, i] - l, q1[0, i] + l]
        target = pv.Box(bounds=bounds, quads=False)
        cells = self._solverHelper(q0[:, :3], environment, target, radius * self.marginRatio, set(), self.depth)
    
    def _solverHelper(self, q0: np.ndarray, environment: pv.PolyData, target:pv.PolyData, margin: float, oldCells: set, depth: int):
        if depth <= 0:
            return oldCells
        # ray trace for target
        rotations = [
            makeRotation('y', np.pi / 2),
            makeRotation('y', -np.pi / 2),
            makeRotation('y', 0),
            makeRotation('y', np.pi),
            makeRotation('x', np.pi / 2), 
            makeRotation('x', -np.pi / 2)
        ]
        allCells = set()
        for rotation in rotations:
            transform = makeTransformation(rotation, q0[:, :3])
            cells, tarCells = self.rayTracer.trace(environment, target, transform)
            if len(tarCells) > 0:
                return set()
            allCells = allCells.union(cells)
        newCells = allCells.difference(oldCells)

        for newPoint in self._pointsFromCells(newCells, margin, environment):
            allCells = self._solverHelper(newPoint, environment, target, margin, allCells)

        return allCells
    
    def _pointsFromCells(self, cells, margin, environment: pv.PolyData):
        planes = PlaneCollection()
        for cell in cells:
            points = environment.cell_points(cell)
            planes.add(points)
        # map to xy plane and do convex hull
        for plane in planes.itr():
            t = np.arctan2(plane.n[1], plane.n[0])
            p = np.arctan2(plane.n[2], np.linalg.norm(plane.n[:2]))
            r = np.matmul(makeRotation('y', p), makeRotation('z', t))
            xy = np.matmul(r, np.array(plane.points))
            xy = np.unique(xy, axis=0)
            hull = scipy.spatial.ConvexHull(xy[:, :2])
            z = np.zeros([hull.points.shape[0], 1])
            hulled = np.append(hull.points, z, axis=1)
            og = np.matmul(r.T, hulled)
            for i in range(og.shape[0]):
                yield og[i, :] + margin * plane.n
            


class PlaneCollection:
    def __init__(self):
        self.collection: 'list[Plane]' = []

    def add(self, points: np.ndarray):
        for plane in self.collection:
            if plane.coincident(points):
                plane.addPoints(points)
                return
        n = np.cross(points[0, :] - points[1, :], points[0, :] - points[2, :])
        n = n / np.linalg.norm(n)
        self.collection.append(Plane(n, points[0, :], points))
    
    def itr(self):
        for plane in self.collection:
            yield plane


class Plane:
    def __init__(self, normal, point, points):
        self.d = normal[0] * point[0] + normal[1] * point[1] + normal[2] * point[2]
        self.n = normal
        self.points = []
        self.addPoints(points)
    
    def addPoints(self, points: np.ndarray):
        self.points = self.points + [points[i, :] for i in range(points.shape[0])]
    
    def coincident(self, points) -> bool:
        r = self.d - np.dot(self.n, points)
        return (np.abs(r) < APPROX_ZERO).all()

        