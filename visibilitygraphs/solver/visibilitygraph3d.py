from typing import Callable
from .helpers import polygonsFromMesh, inflatePolygon, heapUpdatePriority
from .solver import Solver
from visibilitygraphs.models import Vertex, DubinsPath
from visibilitygraphs.dubinspath import VanaAirplane
import pyvista as pv
import numpy as np
import heapq


APPROX_ZERO = .0001

class VisibilityGraph3D(Solver):
    def __init__(self, numLevelSets, inflateFactor):
        self.numLevelSets = numLevelSets
        self.inflateFactor = inflateFactor
        self.dubins = VanaAirplane()

    def solve(self, q0: np.ndarray, q1: np.ndarray, radius: float, flightAngle: float, environment: pv.PolyData) -> 'list[DubinsPath]':

        def modifiedDistance(a , b):
            """calculate best path based on flight angle"""
            dz = abs(b[0, 2] - a[0, 2])
            h = dz / np.sin(flightAngle)
            return int(np.ceil(max(h, np.linalg.norm(b - a))))

        _, _, _, _, zMin, zMax = environment.bounds

        vertices, costMatrix = self.makeGraph(q0[:3], q1[:3], environment, np.linspace(zMin, zMax, self.numLevelSets), radius * self.inflateFactor, modifiedDistance)
        end = vertices[1]

        validPath = False
        while not validPath:
            sequence = self.branchAndBound(vertices[0], vertices[1], vertices, costMatrix)
            if np.isinf(end.cost):
                raise NoPathFoundException()

            sequence = self.findHeadings(sequence, radius, flightAngle)
            
            validPath = True
            paths = []
            for i in range(len(sequence)):
                s = sequence[i - 1]
                e = sequence[i]
                path = self.dubins.calculatePath(s, e, radius, flightAngle)

                if not self.validPath(path, environment):
                    costMatrix[i - 1, i] = -1
                    costMatrix[i, i - 1] = -1
                    validPath = False
                else:
                    paths.append(path)
        
        return validPath
        

    def branchAndBound(self, start: Vertex, end: Vertex, vertices: 'list[Vertex]', costMatrix: np.ndarray, costFunction: 'Callable[[np.ndarray, np.ndarray], float]'):
        for vertex in vertices:
            vertex.cost = np.inf
            vertex.traceback = np.inf
        start.cost = 0
        start.traceback = 0
        queue = vertices.copy()
        heapq.heapify(queue)
        bound = costFunction(start.asArray()[:3], end.asArray()[:3])
        indices = np.arange(vertices.shape[0])

        while len(queue) > 0:
            current = heapq.heappop(queue)
            for i in indices[costMatrix[current.id, :] > 0]:
                vertex = vertices[i]
                newCost = current.traceback + costFunction(current.asArray()[:3], vertex.asArray()[:3])
                totalCost = newCost + costFunction(vertex.asArray()[:3], end.asArray()[:3])
                if totalCost < vertex.cost and totalCost < bound:
                    vertex.cost = totalCost
                    vertex.traceback = newCost
                    vertex.parent = current
                    heapUpdatePriority(queue, vertex)
                    if vertex == end:
                        break

        sequence: list[Vertex] = [end]
        current = end
        while current.parent is not None:
            current = current.parent
            sequence.append(current)
        return sequence


    def findHeadings(self, vertices: 'list[Vertex]', radius: float, flightAngle: float):
        for i in range(len(vertices)):
            vertices[i].psi, vertices[i].gamma = self.bisectAnglePerpendicular(
                vertices[i - 1],
                vertices[i],
                vertices[(i + 1) % len(vertices)]
            )
        state = 0
        for i in range(len(vertices)):
            if state == 0:
                a = vertices[i - 1].asArray()[:3]
                b = vertices[i].asArray()[:3]
                dist = np.linalg.norm(a - b)
                if dist <= 4 * radius:
                    state = 1
                    psi, gamma = self.vectorAngle((b - a))
                    psi = (psi + 2 * np.pi) % (2 * np.pi)
                    gamma = np.clip(gamma, -flightAngle, flightAngle)
                    vertices[i - 1].psi = psi
                    vertices[i - 1].gamma = gamma
                    vertices[i].psi = psi
                    vertices[i].gamma = gamma
            elif state == 1:
                state = 0
        return vertices

    def bisectAnglePerpendicular(self, a: Vertex, b: Vertex, c: Vertex, flightAngle):
        vA = np.array([[a.x, a.y, a.z]])
        vB = np.array([[b.x, b.y, b.z]])
        vC = np.array([[c.x, c.y, c.z]])
        ba = vB - vA
        bc = vC - vB
        psiBa, gammaBa = self.vectorAngle(ba)
        psiBc, gammaBc = self.vectorAngle(bc)
        psi = ((psiBc + psiBa) / 2 + 2 * np.pi) % (2 * np.pi)
        u = np.array([np.cos(psi), np.sin(psi)])
        if (np.dot(u, ba[0, :2]) < 0).all() and (np.dot(u, bc[0, :2]) < 0).all():
            psi = (psi + np.pi) % (2 * np.pi)
        gamma = (gammaBa + gammaBc) / 2
        return psi, np.clip(gamma, -flightAngle, flightAngle)
    
    def vectorAngle(self, vector):
        theta = np.arctan2(vector[0, 1], vector[0, 0])
        phi = np.arctan2(vector[0, 2], np.linalg.norm(vector[0, :2])) # phi in [-pi/2 to pi/2]

        return theta, phi


    def makeGraph(self, start, end, environment: pv.PolyData, levelSets, inflateRadius, costFunction: 'Callable[[np.ndarray, np.ndarray], float]'):

        # do 2d problem at different altitude slices
        levels = [polygonsFromMesh(levelSet, environment) for levelSet in levelSets]

        # get all points from different 2d problems
        allX = [start[0], end[0]]
        allY = [start[1], end[1]]
        allZ = [start[2], end[2]]
        for j, polygons in enumerate(levels):
            for i in range(len(polygons)):
                x, y = inflatePolygon(polygons[i], inflateRadius).exterior.xy
                allX += x
                allY += y
                allZ += [levelSets[j]] * len(y)
        points = np.array([allX, allY, allZ]).T

        # make graph
        # evaluate visibility with ray tracing
        indices = np.array(np.triu_indices(points.shape[0], k=1)).T
        origins = np.apply_along_axis(lambda x: points[x[0]], 0, indices)
        directions = np.apply_along_axis(lambda x: points[x[1]] - points[x[0]], indices)
        _, ray_indices, _ = environment.multi_ray_trace(origins, directions, first_point=True)
        select = np.in1d(points.shape[0], ray_indices)
        visible = indices[~select]
        costMatrix = np.ones([points.shape[0], points.shape[0]]) * -1
        vertices = [Vertex(x[0], x[1], x[2], i, np.inf) for i, x in enumerate(points)]
        for pair in visible:
            costMatrix[pair[0], pair[1]] = costFunction(points[pair[0], :], points[pair[1], :])
        return vertices, costMatrix
    
    def validPath(self, dubinsPath: DubinsPath, environment: pv.PolyData):
        #TODO figure out circle polygon raytracing algorithm to see if any paths collide with environment
        return True

class NoPathFoundException(Exception):
    pass