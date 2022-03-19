import numpy as np
import pyvista as pv
from visibilitygraphs.models import DubinsPath
"""
Authors
-------
Collin Hague : chague@uncc.edu
"""

class Solver:
    """
    Finds a possible path between two points for a fixed-wing aircraft in a given environment
    """
    def solve(self, q0: np.ndarray, q1: np.ndarray, radius: float, flightAngle: float, environment: pv.PolyData) -> 'list[DubinsPath]':
        """
        Finds a possible path between two points for a fixed-wing aircraft in a given environment
        
        Parameters
        ----------
        q0 : np.ndarray
            1 by 5 column vector for initial state
        q1 : np.ndarray
            1 by 5 column vector for final state
        radius : float
            fixed-wing aircraft turn radius
        flightAngle : float
            min/max flight angle for aircraft
        environment : pv.PolyData
            environment for aircraft to transverse
        """
        pass
