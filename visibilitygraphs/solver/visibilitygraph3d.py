from typing import Callable
from .helpers import polygonsFromMesh, inflatePolygon, heapUpdatePriority
from .solver import Solver
from visibilitygraphs.models import AStarVertex, DubinsPath
from visibilitygraphs.dubinspath import VanaAirplane, vanaAirplaneCurve, maneuverToDir
import pyvista as pv
import numpy as np
import heapq
"""
solves 3d path planning with obsticles for dubins airplane

Authors
-------
Collin Hague : chague@uncc.edu

References
----------
D’Amato, E., Notaro, I., Blasi, L., &#38; Mattei, M. (2019). Smooth path planning for fixed-wing aircraft in 3D environment using a layered essential visibility graph. 2019 International Conference on Unmanned Aircraft Systems, ICUAS 2019, 9–18. https://doi.org/10.1109/ICUAS.2019.8797929
"""

APPROX_ZERO = .0001

class VisibilityGraph3D(Solver):
    """
    Solves the dubins airplane problem between two points while avoiding obsticles using visibility graphs
    
    Methods
    -------
    __init__(numLevelSets: int, inflateFactor: float, sampleDistance: float, checkSegments: int): VisibilityGraph3D
    solve(q0: ndarray, q1: ndarray, radius: float, flightAngle: float, environment: PolyData): list[DubinsPath]
    aStar(start: Vertex, end: Vertex, vertices: list[Vertex], costMatrix: ndarray, costFunction: Callable[[ndarray, ndarray], float]): list[Vertex]
    findHeadings(vertices: list[Vertex], radius: float, flightAngle: float): list[Vertex]
    bisectAnglePerpendicular(a: Vertex, b: Vertex, c: Vertex, flightAngle)
    vectorAngle(vector: ndarray): tuple[[float, float]]
    makeGraph(start: ndarray, end: ndarray, environment: PolyData, levelSets: ndarray, inflateRadius: float, costFunction: Callable[[ndarray, ndarray], float]): tuple[list[Vertex], ndarray]
    validPath(dubinsPath: DubinsPath, environment: PolyData): bool
    collisionFree(environment: PolyData, y: ndarray): bool
    """
    def __init__(self, numLevelSets: int, inflateFactor: float, sampleDistance: float, checkSegments: int):
        """
        Parameters
        ----------
        numLevelSets: int
            number of z altitude slices to make
        inflateFactor: float
            distance from obsticles / turn radius
        sampleDistance: float
            distance along level set polygons to sample
        checkSegments: int
            number of segments to check for dubins curve collisions
        """
        self.numLevelSets = numLevelSets
        self.inflateFactor = inflateFactor
        self.dubins = VanaAirplane()
        self.sampleDistance = sampleDistance
        self.checkSegments = checkSegments

    def solve(self, q0: np.ndarray, q1: np.ndarray, radius: float, flightAngle: float, environment: pv.PolyData) -> 'list[DubinsPath]':
        """
        Raises
        ------
        NoPathFoundException: no valid paths are found

        References
        ----------
        see visibilitygraphs.solver.Solver for more
        """
        def modifiedDistance(a , b):
            """calculate best path based on flight angle"""
            dz = abs(b[2] - a[2])
            h = dz / np.sin(flightAngle)
            return int(np.ceil(max(h, np.linalg.norm(b - a))))

        _, _, _, _, zMin, zMax = environment.bounds

        vertices, costMatrix = self.makeGraph(q0[0, :3], q1[0, :3], environment, np.linspace(zMin + (zMax - zMin) / self.numLevelSets + radius, zMax, self.numLevelSets), radius * self.inflateFactor, modifiedDistance)
        end = vertices[1]

        validPath = False
        while not validPath:
            sequence = self.aStar(vertices[0], vertices[1], vertices, costMatrix, modifiedDistance)
            if np.isinf(end.cost):
                raise NoPathFoundException()

            sequence = self.findHeadings(sequence, radius, flightAngle)
            
            validPath = True
            paths = []
            for i in range(1, len(sequence)):
                s = sequence[i - 1]
                e = sequence[i]
                path = self.dubins.calculatePath(s, e, radius, flightAngle)

                if not self.validPath(path, environment):
                    costMatrix[i - 1, i] = -1
                    costMatrix[i, i - 1] = -1
                    validPath = False
                else:
                    paths.append(path)
        
        return paths
        

    def aStar(self, start: AStarVertex, end: AStarVertex, vertices: 'list[AStarVertex]', costMatrix: np.ndarray, costFunction: 'Callable[[np.ndarray, np.ndarray], float]'):
        """
        implementation of branch and bound algorithm for seaching for path from start to end

        Parameters
        ----------
        start: Vertex
            starting vertex
        end: Vertex
            ending vertex
        vertices: list[Vertex]
            list of graph vertices, length n
        costMatrix: ndarray
            upper triangular matrix of edge costs -1 for not valid,
            shape n x n
        costFunction: Callable[[ndarray, ndarray], float]
            lower bound for traveling between vertices
        
        Returns
        -------
        list[Vertex]
            sequence of vertices traveling from start to end (start, ..., end)
        """
        for vertex in vertices:
            vertex.cost = np.inf
            vertex.traceback = np.inf
        start.cost = 0
        start.traceback = 0
        queue = vertices.copy()
        heapq.heapify(queue)
        indices = np.arange(len(vertices))

        while len(queue) > 0:
            current = heapq.heappop(queue)
            for i in indices[costMatrix[current.id, :] > 0]:
                vertex = vertices[i]
                newCost = current.traceback + costFunction(current.asArray()[0, :3], vertex.asArray()[0, :3])
                totalCost = newCost + costFunction(vertex.asArray()[0, :3], end.asArray()[0, :3])
                if totalCost < vertex.cost:
                    vertex.cost = totalCost
                    vertex.traceback = newCost
                    vertex.parent = current
                    heapUpdatePriority(queue, vertex)
                    if vertex == end:
                        break
            if vertex == end:
                break

        sequence: list[AStarVertex] = [end]
        current = end
        while current.parent is not None:
            current = current.parent
            sequence.append(current)
        sequence.reverse()
        return sequence


    def findHeadings(self, vertices: 'list[AStarVertex]', radius: float, flightAngle: float):
        """
        uses angle bisector and alternating algorithms to find headings for dubins airplane

        Parameters
        ----------
        vertices: list[Vertex]
            list of vertices to transverse
        radius: float
            turn radius of airplane
        flightAngle:
            flightAngle of airplane
        
        Returns
        -------
        list[Vertex]
            list of vertices updated with heading angles
        """
        # 0 vertices
        if len(vertices) < 1:
            return vertices
        
        # 1 vertex
        if len(vertices) == 1:
            vertices[0].psi = 0
            vertices[0].gamma = 0
            return vertices
            
        # two vertices
        if len(vertices) == 2:
            a = vertices[0].asArray()[:3]
            b = vertices[1].asArray()[:3]
            psi, gamma = self.vectorAngle((b - a))
            psi = (psi + 2 * np.pi) % (2 * np.pi)
            gamma = np.clip(gamma, -flightAngle, flightAngle)
            vertices[0].psi = psi
            vertices[0].gamma = gamma
            vertices[1].psi = psi
            vertices[1].gamma = gamma
            return vertices
        
        #three or more
        for i in range(1, len(vertices) - 1):
            vertices[i].psi, vertices[i].gamma = self.bisectAnglePerpendicular(
                vertices[i - 1],
                vertices[i],
                vertices[(i + 1) % len(vertices)],
                flightAngle
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

    def bisectAnglePerpendicular(self, a: AStarVertex, b: AStarVertex, c: AStarVertex, flightAngle: float):
        """
        find angle bisectors between two points

        Parameters
        ----------
        a: Vertex
            first point
        b: Vertex
            middle point
        c: Vertex
            last point
        fightAngle: float

        Returns
        -------
        tuple[float, float]
            xy angle psi and sz angle gamma
        """
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
    
    def vectorAngle(self, vector: np.ndarray):
        """
        find angles created by vector

        Paramters
        ---------
        vector: ndarray
            vector to find angles for
        
        Returns
        -------
            xy angle psi and sz angle gamma
        """
        psi = np.arctan2(vector[0, 1], vector[0, 0])
        gamma = np.arctan2(vector[0, 2], np.linalg.norm(vector[0, :2])) # phi in [-pi/2 to pi/2]

        return psi, gamma


    def makeGraph(self, start: np.ndarray, end: np.ndarray, environment: pv.PolyData, levelSets: np.ndarray, inflateRadius: float, costFunction: 'Callable[[np.ndarray, np.ndarray], float]'):
        """
        make visibility graph

        Parameters
        ----------
        start: ndarray
            start location
        end: ndarray
            end location
        environment: PolyData
            mesh of the environment to transverse
        levelSets: ndarray
            list of z levels to slice environment with
        inflateRadius: float
            amount to inflate obstacles by
        costFunction: Callable[[ndarray, ndarray], float]
            lower found function to traveling between two points
        
        Returns
        -------
        tuple[list[Vertex], npdarray]
            (vertices, costMatrix) vertices is a vertex list for the graph cost matrix is an
            upper triangular matrix of lower bound travel costs

        """
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
            if j == len(levels) - 1:
                for i in range(len(polygons)):
                    x, y = inflatePolygon(polygons[i], inflateRadius).exterior.xy
                    allX += x
                    allY += y
                    allZ += [levelSets[j] + inflateRadius] * len(y)
                
        points = np.array([allX, allY, allZ]).T

        # make graph
        # evaluate visibility with ray tracing
        indices = np.array(np.triu_indices(points.shape[0], k=1)).T
        origins = np.apply_along_axis(lambda x: points[x[0]], 1, indices)
        directions = np.apply_along_axis(lambda x: points[x[1]] - points[x[0]], 1, indices)
        intersections, ray_indices, _ = environment.multi_ray_trace(origins, directions, first_point=True)
        t = np.linalg.norm(intersections - origins[ray_indices],axis=1) / np.linalg.norm(directions[ray_indices], axis=1)
        select = np.in1d(np.arange(indices.shape[0]), ray_indices[t < 1])
        visible = indices[~select]
        costMatrix = np.ones([points.shape[0], points.shape[0]]) * -1
        vertices = [AStarVertex(x=x[0], y=x[1], z=x[2], id=i, cost=np.inf) for i, x in enumerate(points)]
        for pair in visible:
            costMatrix[pair[0], pair[1]] = costFunction(points[pair[0], :], points[pair[1], :])
            costMatrix[pair[1], pair[0]] = costMatrix[pair[0], pair[1]]
        return vertices, costMatrix
    
    def validPath(self, dubinsPath: DubinsPath, environment: pv.PolyData):
        """
        sees if a dubins path intersects with environment

        Parameters
        ----------
        dubinsPath: DubinsPath
            dubins path to test
        environment: PloyData
            environment to test against
        
        Returns
        -------
        bool
            true if path doesn't intersect with environment false if path intersects with environment
        """
        a = np.array([dubinsPath.a, dubinsPath.b, dubinsPath.c])
        a = a / np.sum(a)
        b = np.array([dubinsPath.d, dubinsPath.e, dubinsPath.f])
        b = b / np.sum(b)
        start = 0
        f = vanaAirplaneCurve(dubinsPath)
        for i in range(3):
            k = maneuverToDir(dubinsPath.type.name[i])
            j = maneuverToDir(dubinsPath.type.name[i])
            # curve
            if np.abs(k) + np.abs(j) > 0:
                end = np.clip(start + np.max([a[i], b[i]]),0, 1)
                t = np.linspace(start, end, self.checkSegments)
                y = np.array([f(s) for s in t])
                if not self.collisionFree(environment, y):
                    return False
                start = end
            # line segment
            else:
                end =  start + np.min([a[i], b[i]])
                y = np.array([f(s) for s in [start, end]])
                if not self.collisionFree(environment, y):
                    return False
                start = end
        return True
    
    def collisionFree(self, environment: pv.PolyData, y: np.ndarray) -> bool:
        """
        checks to see if the path is collision free

        Parameters
        ----------
        environment: PolyData
            environment for transversal
        y: list[float]
            path waypoints
        
        Returns
        -------
        bool
            True if the path is collision free
        """
        origins = y[:-1]
        directions = y[1:] - origins
        intersections, ray_indices, _ = environment.multi_ray_trace(origins, directions, first_point=True)
        t = np.linalg.norm(intersections - origins[ray_indices],axis=1) / np.linalg.norm(directions[ray_indices], axis=1)
        select = np.in1d(np.arange(origins.shape[0]), ray_indices[t < 1])
        return np.sum(select) <= 0

class NoPathFoundException(Exception):
    """
    Exception that is raise when to valid path between start and end is found
    """
    pass