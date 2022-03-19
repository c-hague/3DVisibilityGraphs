from .helpers import polygonsFromMesh, inflatePolygon
from visibilitygraphs.models import Vertex, Edge
import pyvista as pv
import numpy as np


APPROX_ZERO = .0001

class VisibilityGraph3D(object):
    def __init__(self):
        pass

    def makeGraph(self, start, end, environment: pv.PolyData, levelSets, inflateRadius):

        # do 2d problem at different altitude slices
        levels = [polygonsFromMesh(levelSet, environment) for levelSet in levelSets]

        # get all points from different 2d problems
        allX = [start[0], end[0]]
        allY = [start[1], end[1]]
        allZ = [start[2], end[2]]
        for j, polygons in enumerate(levels):
            for i in range(len(polygons)):
                x, y = inflatePolygon(polygons[i], inflateRadius).exterior.xy
                allX += x
                allY += y
                allZ += [levelSets[j]] * len(y)
        points = np.array([allX, allY, allZ]).T

        # make graph
        # evaluate visibility with ray tracing
        indices = np.array(np.triu_indices(points.shape[0], k=1)).T
        origins = np.apply_along_axis(lambda x: points[x[0]], 0, indices)
        directions = np.apply_along_axis(lambda x: points[x[1]] - points[x[0]], indices)
        _, ray_indices, _ = environment.multi_ray_trace(origins, directions, first_point=True)
        select = np.in1d(points.shape[0], ray_indices)
        visible = indices[~select]
        vertices = [Vertex(x[0], x[1], x[2], i) for i, x in enumerate(points)]
        return [Edge(vertices[x[0]], vertices[x[1]]) for x in visible]
        

