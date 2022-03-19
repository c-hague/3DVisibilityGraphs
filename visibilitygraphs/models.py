from dataclasses import dataclass
from functools import total_ordering
import numpy as np
from enum import IntEnum, auto


class DubinsPathType(IntEnum):
    """Different Dubins Paths"""
    UNKNOWN = auto()
    LSL = auto()
    LSR = auto()
    RSL = auto()
    RSR = auto()
    LRL = auto()
    RLR = auto()


@dataclass
class DubinsPath:
    """class for dubins path data"""
    start: 'Vertex'
    end: 'Vertex'
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    r: float
    rz: float
    type: DubinsPathType
    zType: DubinsPathType
    cost: float
    n: int


@total_ordering
@dataclass
class Vertex:
    """class for vertex of a graph"""
    x: float
    y: float
    z: float
    psi: float
    gamma: float
    id: int
    cost: float
    parent: 'Vertex'
    traceback: float

    @staticmethod
    def fromList(a):
        return Vertex(*a)

    def asList(self):
        return [self.x, self.y, self.z, self.psi, self.gamma]

    def asArray(self):
        return np.array([[self.x, self.y, self.z, self.psi, self.gamma]])


    def __eq__(self, other):
        if not isinstance(Vertex):
            return False
        return self.id == other.id
    
    def __lt__(self, other):
        if not isinstance(Vertex):
            return False
        return self.cost < other.cost
