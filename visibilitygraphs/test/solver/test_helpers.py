from visibilitygraphs.solver.helpers import polygonsFromMesh, inflatePolygon
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

def testPolygonsFromMesh():
    reader = pv.get_reader('data/uptownCharlotte.obj')
    environment: pv.PolyData = reader.read()
    environment.transform(np.array([[1, 0, 0, 0], [0 , 0 , 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
    polygs = polygonsFromMesh(20, environment)
    for poly in polygs:
        plt.plot(*poly.exterior.xy)
    plt.show()

def testInflatePolygon():
    polygon = Polygon(np.array([[ -82.67499797, -368.65649923],
       [ -90.19999695, -377.41400146],
       [ -86.25333023, -380.82983398],
       [ -66.51999664, -397.90899658],
       [ -63.99049695, -395.16666158],
       [ -51.3429985 , -381.45498657],
       [ -47.85699876, -384.76332092],
       [ -30.42700005, -401.30499268],
       [ -26.05600007, -396.75732931],
       [  -4.20100021, -374.01901245],
       [  -7.56266689, -370.78134155],
       [ -24.37100029, -354.59298706],
       [ -26.03516674, -356.2906545 ],
       [ -34.35599899, -364.7789917 ],
       [ -34.62116559, -364.53216044],
       [ -35.9469986 , -363.29800415],
       [ -33.90699895, -361.05867004],
       [ -23.70700073, -349.86199951],
       [ -27.97650083, -346.10666402],
       [ -49.32400131, -327.32998657],
       [ -50.54400126, -328.61765544],
       [ -56.64400101, -335.05599976],
       [ -56.93050067, -334.8258311 ],
       [ -58.36299896, -333.67498779],
       [ -62.8416659 , -338.64932251],
       [ -85.23500061, -363.52099609],
       [ -84.5575002 , -364.08499654],
       [ -81.16999817, -366.90499878],
       [ -82.67499797, -368.65649923]]))
    newPolygon = inflatePolygon(polygon, 50)
    plt.plot(*polygon.exterior.xy)
    plt.plot(*newPolygon.exterior.xy)
    plt.show()
