import numpy as np
import pyvista as pv
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
import heapq
"""
Authors
-------
Collin Hague : chague@uncc.edu
"""

APPROX_ZERO = .0001

class Node:
    """
        node class for connecting line segments
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.next: Node = None
        self.root: Node = None

def polygonsFromMesh(zLevel: float, mesh: pv.PolyData) -> 'list[Polygon]':
    """
    slices a mesh along a plane parallel to xy plane at height zLevel

    Parameters
    ----------
    zLevel: float
        z height to slice at
    mesh: PolyData
        environment mesh
    Returns
    -------
    list[Polygons]
        list of polygons resulting from z slice
    """
    points = np.array([mesh.cell_points(i) for i in range(mesh.n_cells)])
    vectors = np.roll(points, 1,axis=1) - points
    t = np.einsum('k, ijk->ij', [0, 0, 1], np.subtract(points, np.array([[0, 0, zLevel]]))) / np.einsum('ijk, k->ij', -vectors, [0, 0, 1])
    indexLine = np.sum((t >= 0) & (t < 1), axis=1) > 1
    intersections = np.sum(indexLine)
    indexIntersection = (t[indexLine] > 0) & (t[indexLine] < 1)
    p = np.reshape(points[indexLine][indexIntersection], [intersections, 2, 3])
    d = np.reshape(vectors[indexLine][indexIntersection], [intersections, 2, 3])
    s = np.reshape(t[indexLine][indexIntersection], [intersections, 2])
    segments = np.zeros_like(p)
    for ii in range(p.shape[0]):
        for jj in range(p.shape[1]):
            segments[ii, jj, :] = p[ii, jj, :] + s[ii, jj] * d[ii, jj, :]

    # make polygons out of segments
    if len(segments) <= 0:
        return []
    ring = Node(segments[0, 1, :].copy(), segments[0, 0, :].copy())
    segments[0, :, :] = np.inf
    rings = []
    while not np.isinf(segments).all():
        miss = True
        vec = np.linalg.norm(segments - ring.end, axis=2)
        i = np.argmin(vec, axis=0)
        # check for duplicate segment
        a = vec[i, [0, 1]] < APPROX_ZERO
        if i[0] == i[1] and a.all():
            segments[i] = np.inf
            miss = False
            continue

        # if the end matches
        if a.any():
            miss = False
            if a[0]:
                segment = segments[i[0], :, :].copy()
                ring.next = Node(segment[0, :], segment[1, :])
                segments[i[0], :, :] = np.inf
            else:
                segment = segments[i[1], :, :].copy()
                ring.next = Node(segment[1, :], segment[0, :])
                segments[i[1], :, :] = np.inf
            if ring.root is None:
                ring.next.root = ring
            else:
                ring.next.root = ring.root
            ring = ring.next

            # check to see if loop closed
            if np.linalg.norm(ring.root.start - ring.end) < APPROX_ZERO:
                ring.next = ring.root
                rings.append(ring.root)
                if not np.isinf(segments).all():
                    i, _, _ = np.where(~np.isinf(segments))
                    ring = Node(segments[i[0], 1, :].copy(), segments[i[0], 0, :].copy())
                    segments[i[0], :, :] = np.inf

        if miss and not np.isinf(segments).all():
            # bad ring
            i, _, _ = np.where(~np.isinf(segments))
            ring = Node(segments[i[0], 1, :].copy(), segments[i[0], 0, :].copy())
            segments[i[0], :, :] = np.inf

    polygons = []
    for ring in rings:
        ps = []
        current: Node = ring.next
        while current.root is not None:
            ps.append(current.start)
            current = current.next
        ps.append(current.start)
        if len(ps) < 3:
            continue
        polygons.append(orient(Polygon(shell=ps)))

    return polygons


def inflatePolygon(polygon: Polygon, radius: float) -> Polygon:
    """
    inflates polygon by moving points away from polygon and taking convex hull
    
    Parameters
    ----------
    polygon: Polygon
        polygon to inflate
    radius: float
        radius to inflate py
    
    Returns
    -------
    Polygon
        inflated polygon
    """
    points = np.array(polygon.exterior.xy).T[:-1, :]
    directions = points - np.roll(points, 1, axis=0)
    directions = np.divide(directions.T, np.linalg.norm(directions,axis=1)).T
    orth = np.matmul(directions, np.array([[0, -1], [1, 0]]))
    newStarts = points + orth * radius
    orth1 = np.roll(orth, 1, axis=0)
    l = -np.einsum('ij, ij->i', np.roll(newStarts, 1, axis=0), orth1)
    t = -(l + np.einsum('ij, ij->i',orth1, newStarts)) / np.einsum('ij, ij->i',orth1, directions)
    newPoints = newStarts + np.einsum('ij, i -> ij', directions, t)
    cleanedPoints = np.reshape(newPoints[~np.isinf(newPoints)], [-1, 2])
    
    return Polygon(shell=cleanedPoints).convex_hull


def heapUpdatePriority(heap:list, item):
    """
    update heap item priority

    Parameters
    ----------
    heap: list
        heap created by heapq, is modified
    item: Any
        item with priority update
    """
    i = heap.index(item)
    if i < 0:
        return
    # move to top of heap
    while i > 0:
        heap[(i - 1) // 2], heap[i] = heap[i], heap[(i - 1) // 2]
        i = (i - 1) // 2
    # remove from heap and reinsert
    temp = heapq.heappop(heap)
    heapq.heappush(heap, temp)