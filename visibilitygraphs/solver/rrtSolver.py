from visibilitygraphs.dubinspath.vanaAirplane import FailureToConvergeException
from .solver import Solver
import pyvista as pv
import numpy as np
from visibilitygraphs.models import DubinsPath, DubinsPathFraction, RRTVertex, mapClass
from visibilitygraphs.dubinspath import VanaAirplane, vanaAirplaneCurve, maneuverToDir


class RRTSolver(Solver):
    def __init__(self, numberPoints: int, checkSegments: int, goalRadius: float, segmentDistance: float):
        self.numberPoints = numberPoints
        self.checkSegments = checkSegments
        self.segmentDistance = segmentDistance
        self.goalRadius = goalRadius
        self.tryGoal = 10
        self.dubins = VanaAirplane()
    
    def solve(self, q0: np.ndarray, q1: np.ndarray, radius: float, flightAngle: float, environment: pv.PolyData) -> 'list[DubinsPath]':
        xMin, xMax, yMin, yMax, zMin, zMax = environment.bounds
        r = np.array([xMax - xMin, yMax - yMin, zMax - zMin])
        offset = np.array([xMin, yMin, zMin])
        vertices = [
            RRTVertex(
                x=q1[0, 0],
                y=q1[0, 1],
                z=q1[0, 2],
                psi=q1[0, 3],
                gamma=q1[0, 4],
                id=1,
                cost=np.inf,
            ),
            RRTVertex(
                x=q0[0, 0],
                y=q0[0, 1],
                z=q0[0, 2],
                psi=q0[0, 3],
                gamma=q0[0, 4],
                id=0,
            )
        ]
        i = 2
        k = 0
        while i < self.numberPoints:
            if k % self.tryGoal == 0:
                point = vertices[0].asArray()[:3]
            else:
                point = np.random.random([3]) * r + offset
            j = np.argmin([np.linalg.norm(point - v.asArray()[:3]) for v in vertices[1:]]) + 1
            dx = point[0] - vertices[j].x
            dy = point[1] - vertices[j].y
            dz = point[2] - vertices[j].z
            vertex = RRTVertex(
                x=point[0],
                y=point[1],
                z=point[2],
                psi=np.arctan2(dy, dx),
                gamma=np.clip(np.arctan2(dz, np.sqrt(dx ** 2 + dy ** 2)), -flightAngle, flightAngle),
                id=i
            )
            path = self.dubins.calculatePath(vertices[j],vertex , radius, flightAngle)
            path = mapClass(path, DubinsPathFraction)
            frac = self.segmentDistance / path.cost
            path.fend = np.min([frac, 1])
            parent = vertices[j]
            f = vanaAirplaneCurve(path)
                
            # extend rrt
            while path.fend < 1 and self.validPath(path, environment):
                curve = np.array([f(t) for t in [(path.fend - path.fstart) * .99 + path.fstart, path.fend]])

                # reaches goal state
                if np.linalg.norm(curve[-1, :] - q1[0, :3]) < self.goalRadius:
                    path = self.dubins.calculatePath(vertices[j], vertices[0], radius, flightAngle)
                    path = mapClass(path, DubinsPathFraction)
                    path.fstart = 0
                    path.fend = 1
                    if self.validPath(path, environment):
                        vertices[0].parent = vertices[j]
                        vertices[0].cost = path.cost
                        vertices[0].pathFromParent = path
                        break
                
                # create new vertex
                dx = curve[-1, 0] - curve[-2, 0]
                dy = curve[-1, 1] - curve[-2, 1]
                dz = curve[-1, 2] - curve[-2, 2]
                vertices.append(
                    RRTVertex(
                        x=curve[-1, 0],
                        y=curve[-1, 1],
                        z=curve[-1, 2],
                        psi=np.arctan2(dy, dx),
                        gamma=np.clip(np.arctan2(dz, np.sqrt(dx ** 2 + dy ** 2)), -flightAngle, flightAngle),
                        id=i,
                        cost=self.segmentDistance,
                        parent=parent,
                        pathFromParent=path
                    )
                )
                i += 1
                parent = vertices[-1]
                path = mapClass(path, DubinsPathFraction)
                path.fstart = path.fend
                path.fend += frac
            k += 1
        
        current = vertices[0]
        finalPath: 'list[DubinsPathFraction]' = []
        while current.parent is not None:
            finalPath.append(current.pathFromParent)
            current = current.parent
        
        finalPath.reverse()
        
        if len(finalPath) <= 0:
            finalPath = [v.pathFromParent for v in vertices if v.parent is not None]
        
        return finalPath

    def validPath(self, dubinsPath: DubinsPathFraction, environment: pv.PolyData):
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
        if np.isinf(dubinsPath.cost):
            return False

        a = np.array([dubinsPath.a, dubinsPath.b, dubinsPath.c])
        a = a / np.sum(a)
        b = np.array([dubinsPath.d, dubinsPath.e, dubinsPath.f])
        b = b / np.sum(b)
        start = dubinsPath.fstart
        f = vanaAirplaneCurve(dubinsPath)
        for i in range(3):
            k = maneuverToDir(dubinsPath.type.name[i])
            j = maneuverToDir(dubinsPath.zType.name[i])
            if start > dubinsPath.fend:
                return True
            # curve
            if np.abs(k) + np.abs(j) > 0:
                end = np.clip(start + np.max([a[i], b[i]]),0, dubinsPath.fend)
                t = np.linspace(start, end, self.checkSegments)
                y = np.array([f(s) for s in t])
                if not self.collisionFree(environment, y):
                    return False
                start = end
            # line segment
            else:
                end =  np.min([start + np.min([a[i], b[i]]), dubinsPath.fend])
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