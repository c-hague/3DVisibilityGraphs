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
from .dubinsCar import DubinsCar
from visibilitygraphs.models import DubinsPath, Vertex, DubinsPathType
import numpy as np



APPROX_ZERO = .0001


# a, b, c, c*, d, e, f, cost, xyType, szType
DEFUALT_DUBINS = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.inf, DubinsPathType.UNKNOWN, DubinsPathType.UNKNOWN)

class VanaAirplane(DubinsCar):
    def calculatePath(self, q0: Vertex, q1: Vertex, r, flightAngle):
        rHorizontal = r * 2
        xyEdge, szEdge = self.decoupled(q0, q1, r, rHorizontal, flightAngle)
        while not self.isFeasible(szEdge, flightAngle):
            rHorizontal *= 2
            xyEdge, szEdge = self.decoupled(q0, q1, r, rHorizontal, flightAngle)
        delta = .1 * r
        while abs(delta) > APPROX_ZERO:
            rPrime = max(r, rHorizontal + delta)
            xyEdgePrime, szEdgePrime = self.decoupled(q0, q1, r, rPrime, flightAngle)
            if self.isFeasible(szEdgePrime, flightAngle) and szEdgePrime.cost < szEdge.cost:
                rHorizontal = rPrime
                szEdge = szEdgePrime
                xyEdge = xyEdgePrime
                delta *= 2
            else:
                delta = -.1 * delta
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
        xyEdge = super().calculatePath(q0, q1, rHorizontal)
        rVertical = 1 / np.sqrt(r ** -2 - rHorizontal ** -2)
        qz0 = Vertex(x=0, y=q0.z, psi=q0.gamma)
        qz1 = Vertex(x=xyEdge.cost, y=q1.z, psi=q1.gamma)
        szEdge = super().calculatePath(qz0, qz1, rVertical)
        return xyEdge, szEdge
    
    def isFeasible(self, szEdge: DubinsPath, flightAngle):
        if szEdge.type == DubinsPathType.LRL or szEdge.type == DubinsPathType.RLR or szEdge.type == DubinsPathType.UNKNOWN:
            return False
        if abs(szEdge.start.psi + szEdge.a / szEdge.r) >  flightAngle:
            return False
        return True
