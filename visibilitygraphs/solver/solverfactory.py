import numpy as np

from .visibilitygraph3d import VisibilityGraph3D
from .solver import Solver
from .rrtSolver import RRTSolver

"""
methods for creating solvers

Authors
-------
Collin Hague : chague@uncc.edu
"""
class SolverType:
    """
    enumeration of different solver type
    """
    UNKNOWN = 0
    VISIBILITY_GRAPH = 1
    RRT = 2


class SolverBuilder(object):
    """
    builder pattern for solvers

    Methods
    -------
    setType(type: SolverType): SolverBuilder
    build(): Solver
    """
    def __init__(self):
        self._type = SolverType.UNKNOWN
        self._levelSets = 0
        self._inflateFactor = 2
        self._sampleDistance = 1
        self._checkSegments = 8
        self._numberPoints = 2000
        self._goalRadius = 100
    
    def setNumberPoints(self, numberPoints: int):
        """
        set solver number of points for rrt
        
        Parameters
        ----------
        numberPoints: int
            how many points to generate

        Returns
        -------
        SolverBuilder
            self for method chaining
        """
        self._numberPoints = numberPoints
        return self
    
    def setGoalRadius(self, goalRadius: float):
        """
        set goal radius for rrt

        Parameters
        ----------
        goalRadius: float
            how close the rrt needs to get to goal

        Returns
        -------
        SolverBuilder
            self for method chaining
        """
        self._goalRadius = goalRadius
        return self
    
    def setLevelSets(self, levelSets: int):
        """
        set solver level set factor for VisibilityGraph3D
        
        Parameters
        ----------
        levelSets: int
            how many z slices to make

        Returns
        -------
        SolverBuilder
            self for method chaining
        """
        self._levelSets = levelSets
        return self
    
    def setInflateFactor(self, inflateFactor: float):
        """
        set solver inflate factor for VisibilityGraph3D
        
        Parameters
        ----------
        inflateFactor: float
            how much to inflate polygons

        Returns
        -------
        SolverBuilder
            self for method chaining
        """
        self._inflateFactor = inflateFactor
        return self

    def setSampleDistance(self, sampleDistance: float):
        """
        set solver sample distance for VisibilityGraph3D
        
        Parameters
        ----------
        sampleDistance: float
            distance to sample along edge of polygons

        Returns
        -------
        SolverBuilder
            self for method chaining
        """
        self._sampleDistance = sampleDistance
        return self
    
    def setCheckSegments(self, checkSegments: int):
        self._checkSegments = checkSegments
        return self

    def setType(self, type: SolverType):
        """
        set solver type
        
        Parameters
        ----------
        type: SolverType type of solver to build

        Returns
        -------
        SolverBuilder
            self for method chaining
        """
        self._type = type
        return self

    def build(self) -> Solver:
        """
        builds the solver

        Returns
        -------
        Solver
        
        Raises
        ------
        SolverBuilderException
            on invalid solver configuration
        """
        if self._type == SolverType.UNKNOWN:
            raise SolverBuilderException('cannot create solver with type SolverType.UNKNOWN')
        elif self._type == SolverType.VISIBILITY_GRAPH:
            return VisibilityGraph3D(self._levelSets, self._inflateFactor, self._sampleDistance, self._checkSegments)
        elif self._type == SolverType.RRT:
            return RRTSolver(self._numberPoints, self._checkSegments, self._goalRadius, self._sampleDistance)
        else:
            raise SolverBuilderException(f'cannot create solver with type {self._type}')

class SolverBuilderException(Exception):
    """
    exception raise when SolverBuilder fails to build a Solver
    """
    pass

