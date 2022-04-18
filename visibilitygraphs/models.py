from dataclasses import dataclass
from functools import total_ordering
import numpy as np
from enum import IntEnum, auto
"""
models for the visibiltygraphs package

Authors
-------
Collin Hague : chague@uncc.edu
"""

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
    start: 'Vertex' = None
    end: 'Vertex' = None
    a: float = 0
    b: float = 0
    c: float = 0
    d: float = 0
    e: float = 0
    f: float = 0
    r: float = 0
    rz: float = 0
    type: DubinsPathType = 0
    zType: DubinsPathType = 0
    cost: float = 0
    n: int = 0


@dataclass
class DubinsPathFraction:
    """class for dubins path data"""
    start: 'Vertex' = None
    end: 'Vertex' = None
    a: float = 0
    b: float = 0
    c: float = 0
    d: float = 0
    e: float = 0
    f: float = 0
    r: float = 0
    rz: float = 0
    type: DubinsPathType = 0
    zType: DubinsPathType = 0
    cost: float = 0
    n: int = 0
    fraction: float = 1

@dataclass
class Vertex:
    """class for vertex of a graph"""
    x: float = 0
    y: float = 0
    z: float = 0
    psi: float = 0
    gamma: float = 0
    id: int = -1

    @staticmethod
    def fromList(a):
        return Vertex(*a)

    def __eq__(self, other):
        if not isinstance(other, Vertex):
            return False
        return self.id == other.id


@total_ordering
@dataclass
class RRTVertex(Vertex):
    """class for vertex of a graph"""
    x: float = 0
    y: float = 0
    z: float = 0
    psi: float = 0
    gamma: float = 0
    id: int = -1
    cost: float = 0
    parent: 'RRTVertex' = None
    pathFromParent: DubinsPathFraction = None

    @staticmethod
    def fromList(a):
        return RRTVertex(*a)

    def asList(self):
        return [self.x, self.y, self.z, self.psi, self.gamma]

    def asArray(self):
        return np.array([self.x, self.y, self.z, self.psi, self.gamma])


    def __eq__(self, other):
        if not isinstance(other, RRTVertex):
            return False
        return self.id == other.id
    
    def __lt__(self, other):
        if not isinstance(other, RRTVertex):
            return False
        return self.cost < other.cost

@total_ordering
@dataclass
class AStarVertex(Vertex):
    """class for vertex of a graph"""
    x: float = 0
    y: float = 0
    z: float = 0
    psi: float = 0
    gamma: float = 0
    id: int = -1
    cost: float = 0
    parent: 'AStarVertex' = None
    traceback: float = 0

    @staticmethod
    def fromList(a):
        return AStarVertex(*a)

    def asList(self):
        return [self.x, self.y, self.z, self.psi, self.gamma]

    def asArray(self):
        return np.array([[self.x, self.y, self.z, self.psi, self.gamma]])

    def __eq__(self, other):
        if not isinstance(other, AStarVertex):
            return False
        return self.id == other.id
    
    def __lt__(self, other):
        if not isinstance(other, AStarVertex):
            return False
        return self.cost < other.cost