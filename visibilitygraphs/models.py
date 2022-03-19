from dataclasses import dataclass
import numpy as np


@dataclass
class DubinsPath:
    """class for dubins path data"""
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    r: float
    rz: float
    type: int
    zType: int
    cost: float

@dataclass
class Vertex:
    x: float
    y: float
    z: float
    id: int

    @staticmethod
    def fromList(a):
        return Vertex(x=a[0], y=a[1], z=a[2])

    def asList(self):
        return [self.x, self.y, self.z]

    def asArray(self):
        return np.array([[self.x, self.y, self.z]])


@dataclass
class Edge:
    start: Vertex
    end: Vertex