from visibilitygraphs.dubinspath.dubinsCar import DubinsCar
from visibilitygraphs.dubinspath.vanaAirplane import VanaAirplane
from visibilitygraphs.models import Vertex
from visibilitygraphs.plotting.helpers import dubinsCurve2d, vanaAirplaneCurve
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d



def testDubinsCurve2dLSL():
    dubins = DubinsCar()
    path = dubins.calculatePath(
        Vertex(x=0, y=0, psi=0),
        Vertex(x=1, y=1, psi=np.pi / 2),
        .25
    )
    f = dubinsCurve2d([path.start.x, path.start.y, path.start.psi], path.a, path.b, path.c, path.r, path.type)
    t = np.linspace(0, 1, 100)
    xy = np.array([f(s) for s in t])
    plt.plot(xy[:, 0], xy[:, 1])
    plt.title('LSL')
    plt.show()

def testDubinsCurve2dLSR():
    dubins = DubinsCar()
    path = dubins.calculatePath(
        Vertex(x=0, y=0, psi=0),
        Vertex(x=1, y=1, psi=0),
        .25
    )
    f = dubinsCurve2d([path.start.x, path.start.y, path.start.psi], path.a, path.b, path.c, path.r, path.type)
    t = np.linspace(0, 1, 100)
    xy = np.array([f(s) for s in t])
    plt.plot(xy[:, 0], xy[:, 1])
    plt.title('LSR')
    plt.show()

def testDubinsCurve2dRSL():
    dubins = DubinsCar()
    path = dubins.calculatePath(
        Vertex(x=0, y=0, psi=0),
        Vertex(x=1, y=-1, psi=0),
        .25
    )
    f = dubinsCurve2d([path.start.x, path.start.y, path.start.psi], path.a, path.b, path.c, path.r, path.type)
    t = np.linspace(0, 1, 100)
    xy = np.array([f(s) for s in t])
    plt.plot(xy[:, 0], xy[:, 1])
    plt.title('RSL')
    plt.show()

def testDubinsCurve2dRSR():
    dubins = DubinsCar()
    path = dubins.calculatePath(
        Vertex(x=0, y=0, psi=0),
        Vertex(x=1, y=-1, psi=-np.pi/2),
        .25
    )
    f = dubinsCurve2d([path.start.x, path.start.y, path.start.psi], path.a, path.b, path.c, path.r, path.type)
    t = np.linspace(0, 1, 100)
    xy = np.array([f(s) for s in t])
    plt.plot(xy[:, 0], xy[:, 1])
    plt.title('RSR')
    plt.show()

def testDubinsCurve2dLRL():
    dubins = DubinsCar()
    path = dubins.calculatePath(
        Vertex(x=0, y=0, psi=0),
        Vertex(x=0, y=-.125, psi=np.pi),
        .25
    )
    f = dubinsCurve2d([path.start.x, path.start.y, path.start.psi], path.a, path.b, path.c, path.r, path.type)
    t = np.linspace(0, 1, 100)
    xy = np.array([f(s) for s in t])
    plt.plot(xy[:, 0], xy[:, 1])
    plt.title('RSL')
    plt.show()

def testDubinsCurve2dRLR():
    dubins = DubinsCar()
    path = dubins.calculatePath(
        Vertex(x=0, y=0, psi=0),
        Vertex(x=0, y=.125, psi=np.pi),
        .25
    )
    f = dubinsCurve2d([path.start.x, path.start.y, path.start.psi], path.a, path.b, path.c, path.r, path.type)
    t = np.linspace(0, 1, 100)
    xy = np.array([f(s) for s in t])
    plt.plot(xy[:, 0], xy[:, 1])
    plt.title('RSL')
    plt.show()

def testVanaAirplaneCurve():
    dubins = VanaAirplane()
    path = dubins.calculatePath(
        Vertex(x=0, y=0, z=0, psi=0, gamma=0),
        Vertex(x=1, y=1, z=1, psi=0, gamma=0),
        .25,
        2 * np.pi / 9
    )
    f = vanaAirplaneCurve(path)
    t = np.linspace(0, 1, 100)
    x = np.array([f(s) for s in t])
    ax = plt.axes(projection='3d')
    ax.plot(x[:, 0], x[:, 1], x[:, 2])
    plt.title('3d curve')
    plt.show()
