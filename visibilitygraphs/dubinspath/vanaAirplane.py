from .dubinsCar import DubinsCar
from visibilitygraphs.models import DubinsPath, Vertex, DubinsPathType
import numpy as np
"""
Authors
-------
    Collin Hague
References
----------
    https://github.com/robotics-uncc/RobustDubins
    Lumelsky, V. (2001). Classification of the Dubins set.
    Vana, P., Alves Neto, A., Faigl, J.; MacHaret, D. G. (2020). Minimal 3D Dubins Path with Bounded Curvature and Pitch Angle.
"""


APPROX_ZERO = .0001
MAX_ITER = 1000


# a, b, c, c*, d, e, f, cost, xyType, szType
DEFUALT_DUBINS = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.inf, DubinsPathType.UNKNOWN, DubinsPathType.UNKNOWN)

class VanaAirplane(DubinsCar):
    """
    Calculates the 3D Dubins path for a fixed-wing aircraft with minimun turn radius and flight angle constraint
    """
    def calculatePath(self, q0: Vertex, q1: Vertex, r, flightAngle):
        """
        Calculates path between starting and final configurations with minimum turn radius r and flight angle constraint

        Parameters
        ----------
        q0: Vertex
            initial configuration
        q1: Vertex
            final configuration
        r: float
            minimum turn radius
        flightAngle: float
            flight angle constraint
        
        Returns
        -------
        DubinsPath
            3D Dubins path
        """
        rHorizontal = r * 2
        xyEdge, szEdge = self.decoupled(q0, q1, r, rHorizontal, flightAngle)
        i = 0
        while not self.isFeasible(szEdge, flightAngle) and i < MAX_ITER:
            rHorizontal *= 2
            xyEdge, szEdge = self.decoupled(q0, q1, r, rHorizontal, flightAngle)
            i += 1
        delta = .1 * r
        i = 0
        while abs(delta) > APPROX_ZERO and i < MAX_ITER:
            rPrime = max(r, rHorizontal + delta)
            xyEdgePrime, szEdgePrime = self.decoupled(q0, q1, r, rPrime, flightAngle)
            if self.isFeasible(szEdgePrime, flightAngle) and szEdgePrime.cost < szEdge.cost:
                rHorizontal = rPrime
                szEdge = szEdgePrime
                xyEdge = xyEdgePrime
                delta *= 2
            else:
                delta = -.1 * delta
            i += 1
        return DubinsPath(
            start=q0,
            end=q1,
            a=xyEdge.a,
            b=xyEdge.b,
            c=xyEdge.c,
            d=szEdge.a,
            e=szEdge.b,
            f=szEdge.c,
            type=xyEdge.type,
            zType=szEdge.type,
            cost=szEdge.cost,
            r=xyEdge.r,
            rz=szEdge.r,
            n=3
        )
    
    
    def decoupled(self, q0: Vertex, q1: Vertex, r, rHorizontal, flightAngle):
        """
        calculate 2 dubins paths one in xy plane and one in arclength z plane

        Parameters
        ----------
        q0: Vertex
            initial configuration
        q1: Vertex 
            final Configuration
        r: float
            minimum turn radius
        rHorizontal: float
            minimum turn radius in xy plane
        flightAngle: float
            flight angle constraint
        
        Returns
        -------
        tuple[DubinsPath, DubinsPath]
            xyDubinspath, szDubinsPath
        """
        xyEdge = super().calculatePath(q0, q1, rHorizontal)
        rVertical = 1 / np.sqrt(r ** -2 - rHorizontal ** -2)
        qz0 = Vertex(x=0, y=q0.z, psi=q0.gamma)
        qz1 = Vertex(x=xyEdge.cost, y=q1.z, psi=q1.gamma)
        szEdge = super().calculatePath(qz0, qz1, rVertical)
        return xyEdge, szEdge
    
    def isFeasible(self, szEdge: DubinsPath, flightAngle):
        """
        is the szDubins path valid

        Parameters
        ----------
        szEdge: DubinsPath
            sz dubins path
        flightAngle: float
            flight angle constraint
        
        Returns
        -------
        bool
            if the sz dubins path is valid
        """
        if szEdge.type == DubinsPathType.LRL or szEdge.type == DubinsPathType.RLR or szEdge.type == DubinsPathType.UNKNOWN:
            return False
        if abs(szEdge.start.psi + szEdge.a / szEdge.r) >  flightAngle:
            return False
        return True

class FailureToConvergeException(Exception):
    pass