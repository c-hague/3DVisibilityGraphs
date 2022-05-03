"""
Main logic

Authors
-------
Collin Hague chague@uncc.edu

"""
import argparse
import numpy as np
import pyvista as pv
import os
from .solver import SolverBuilder, NoPathFoundException, SolverType
from .plotting import SolutionPlotter

def main():
    """
    main method
    """
    parser = argparse.ArgumentParser(
        prog='visibilitygraphs',
        description='''
            Finds a possible path between two points for a fixed-wing aircraft in a given environment
        '''
    )
    parser.add_argument('final', nargs=5, type=float, help='aircraft final state x y z heading pitch')
    parser.add_argument('environment', type=str, help='file for environment')
    parser.add_argument('-i', '--initial', nargs=5, type=float, default=[0, 0, 0, 0, 0], help='aircraft initial state x y z heading pitch')
    parser.add_argument('-r', '--radius', default=1, type=float, help='aircraft turn radius')
    parser.add_argument('-f', '--flightangle', default=np.pi/4, type=float, help='aircraft max/min flight angle')
    parser.add_argument('-t', '--type', default=1, type=int, help='solver type 1-visibility graph')
    parser.add_argument('--levels', default=4, type=int, help='type 1 number of z slices')
    parser.add_argument('--inflate', type=float, default=2, help='type 1 polygon inflation factor')
    parser.add_argument('-p', '--plot', type=bool, default=True, help='plot solution when finished')
    parser.add_argument('--check', type=int, default=8, help='number of segments to decompose dubins path curves to when checking collisions')
    parser.add_argument('--points', type=int, default=2000, help='number of points for algorithm')
    parser.add_argument('--distance', type=float, default=100, help='set sample distance')
    parser.add_argument('--rgoal', type=float, default=50, help='rrt goal radius')
    args = parser.parse_args()

    q0 = np.array([args.initial])
    q1 = np.array([args.final])
    radius = args.radius
    flightAngle = args.flightangle
    fname = args.environment
    if not os.path.isfile(fname):
        raise FileNotFoundError(f'{fname} not found')
    reader = pv.get_reader(fname)
    environment = reader.read()
    environment.transform(np.array([[1, 0, 0, 0], [0 , 0 , 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
    builder = SolverBuilder()
    solver = builder.setType(args.type)\
        .setInflateFactor(args.inflate)\
        .setCheckSegments(args.check)\
        .setGoalRadius(args.rgoal)\
        .setNumberPoints(args.points)\
        .setSampleDistance(args.distance)\
        .setLevelSets(args.levels).build()
    if not environment.is_all_triangles():
        raise ValueError(f'{fname} must be only be triangles')
    try:
        solution = solver.solve(q0, q1, radius, flightAngle, environment)
    except NoPathFoundException:
        solution = []
    if args.plot:
        plotter = SolutionPlotter(args.type)
        plotter.plotSolution(environment, q0, q1, solution)

    print(solution)
    cost = sum([s.cost for s in solution])
    print(f'cost: {cost}')


if __name__ == '__main__':
    main()