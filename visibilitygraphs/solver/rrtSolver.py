from .solver import Solver
import pyvista as pv
import numpy as np
from visibilitygraphs.models import DubinsPath, DubinsPathFraction, RRTVertex
from visibilitygraphs.dubinspath import VanaAirplane, vanaAirplaneCurve, maneuverToDir


class RRTSolver(Solver):
    def __init__(self, numberPoints: int, checkSegments: int, goalRadius: float, segmentDistance: float):
        self.numberPoints = numberPoints
        self.checkSegments = checkSegments
        self.segmentDistance = segmentDistance
        self.goalRadius = goalRadius
        self.dubins = VanaAirplane()
    
    def solve(self, q0: np.ndarray, q1: np.ndarray, radius: float, flightAngle: float, environment: pv.PolyData) -> 'list[DubinsPath]':
        vertices = [
            RRTVertex(
                x=q1[0],
                y=q1[1],
                z=q1[2],
                psi=q1[3],
                gamma=q1[4],
                id=1,
                cost=np.inf,
            ),
            RRTVertex(
                x=q0[0],
                y=q0[1],
                z=q0[2],
                psi=q0[3],
                gamma=q0[4],
                id=0,
            )
        ]
        i = 2
        while i < self.numberPoints:
            point = np.random.random([3])
            j = np.argmin([np.linalg.norm(point - v.asArray()) for v in vertices[1:]])
            path = self.dubins.calculatePath(vertices[j], point, radius, flightAngle)
            path = DubinsPathFraction(**path)
            frac = self.segmentDistance / path.cost
            path.fraction = np.min(frac, 1)
            if not self.validPath(path, environment):
                i += 1
                continue

            f = vanaAirplaneCurve(path)
            curve = np.array([f(t) for t in np.linspace(0, frac, 100)])

            # reaches goal state
            if np.linalg.norm(curve[-1, :] - q1[:2]) < self.goalRadius:
                path = self.dubins.calculatePath(vertices[j], vertices[0], radius, flightAngle)
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
                    psi=np.arctan2(dx, dy),
                    gamma=np.arctan2(dz, dx ** 2 + dy ** 2),
                    id=i,
                    cost=self.segmentDistance,
                    parent=vertices[j],
                    pathFromParent=path
                )
            )
            i += 1
        
        current = vertices[0]
        finalPath: 'list[DubinsPathFraction]' = []
        while current is not None:
            finalPath.append(current.pathFromParent)
            current = current.parent
        
        return finalPath.reverse()

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
        a = np.array([dubinsPath.a, dubinsPath.b, dubinsPath.c])
        a = a / np.sum(a)
        b = np.array([dubinsPath.d, dubinsPath.e, dubinsPath.f])
        b = b / np.sum(b)
        start = 0
        f = vanaAirplaneCurve(dubinsPath)
        for i in range(3):
            k = maneuverToDir(dubinsPath.type.name[i])
            j = maneuverToDir(dubinsPath.zType.name[i])
            if start > dubinsPath.fraction:
                return True
            # curve
            if np.abs(k) + np.abs(j) > 0:
                end = np.clip(start + np.max([a[i], b[i]]),0, dubinsPath.fraction)
                t = np.linspace(start, end, self.checkSegments)
                y = np.array([f(s) for s in t])
                if not self.collisionFree(environment, y):
                    return False
                start = end
            # line segment
            else:
                end =  np.max(start + np.min([a[i], b[i]]), dubinsPath.fraction)
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