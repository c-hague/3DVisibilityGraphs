"""
Authors
-------
Collin Hague : chague@uncc.edu
References
----------
Ray Tracing From Scratch In Python, Omar Aflak https://medium.com/swlh/ray-tracing-from-scratch-in-python-41670e6a96f9
Ray Tracing: Rendering a Triangle, Scratch Pixel https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
"""
import numpy as np
import pyvista as pv
from typing import Tuple

APPROX_ZERO = .0001


class RayTracer(object):
    """
        Ray traces a triangulated environment
        
        Methods
        -------
        trace(vertices, faceIndices, cameraTranform=np.eye(4)) : np.ndarray
    """
    def __init__(self, width:int=64, height:int=64, zeroCutoff=APPROX_ZERO, renderingLimit:float=500.0):
        """
        """
        self.width = width
        self.height = height
        self.ratio = float(width) / height
        self.screen = (-1, 1/self.ratio, 1, -1/self.ratio)
        self.zeroCutoff = zeroCutoff
        self.renderingLimit = renderingLimit
    
    def trace(self, environment: pv.PolyData, target: pv.PolyData, cameraTransform:np.ndarray=None) -> 'Tuple[set[int], set[int]]':
        """
        ray trace environment with camera transformation
        camera is at origin pointing in the z direction

        Parameters
        ----------
        vertices : np.ndarray
            n by 3 matrix of vertices
        faceIndices: np.ndarray
            m by 3 matrix of indices into vertices defining triangular faces
        cameraTransform: np.ndarray
            4 by 4 tranformation matrix transforming the camera orientation and location

        Returns
        -------
        np.ndarray
            a set of points that are visible from the  camera
        """
        cameraTransform = np.eye(4) if cameraTransform is None else cameraTransform
        camera = np.array([0, 0, 1])
        origin = cameraTransform[3, 0:3]
        rotation = cameraTransform[0:3, 0:3]
        tarCells = set()
        envCells = set()

        for i, y in enumerate(np.linspace(self.screen[1], self.screen[3], self.height)):
            for j, x in enumerate(np.linspace(self.screen[0], self.screen[1], self.width)):
                pixel = np.array([x, y, 0])
                direction = pixel - camera
                direction = direction / np.linalg.norm(direction)
                # rotate vector normal
                direction = np.matmul(rotation, direction)
                stop = direction * self.renderingLimit
                points, cells = environment.ray_trace(origin, stop)
                distances = np.linalg.norm(points - origin, axis=0)
                k = np.argmin(distances)

                # target
                targetPoints, targetCells = environment.ray_trace(origin, stop)
                targetDistances = np.linalg.norm(targetPoints - origin, axis=0)
                l = np.argmin(targetDistances)
                if targetDistances[l] < distances[k]:
                    tarCells.add(targetCells[l])
                else:
                    envCells.add(cells[k])
        return envCells, tarCells
