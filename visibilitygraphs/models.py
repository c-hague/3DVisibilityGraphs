from dataclasses import dataclass

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