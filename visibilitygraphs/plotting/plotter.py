import numpy as np
import pyvista as pv
from .helpers import vanaAirplaneCurve

class SolutionPlotter:
    def __init__(self):
        pass

    def plotSolution(self, environment, start, end, paths):
        plotter = pv.Plotter()
        plotter.add_mesh(environment)
        for path in paths:
            poly = pv.PolyData()
            f = vanaAirplaneCurve(path)
            t = np.linspace(0, 1, 100)
            points = np.array([f(s) for s in t])    
            poly.points = points
            cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
            cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
            cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
            poly.lines = cells
            poly['scalars'] = np.arange(poly.n_points)
            mesh = poly.tube(radius=1)
            plotter.add_mesh(mesh)
        s = pv.Sphere(radius=5, center=start[:, :3])
        e = pv.Sphere(radius=5, center=end[:, :3])
        plotter.add_mesh(s, color='green')
        plotter.add_mesh(e, color='red')
        plotter.show()
    
