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
    segments = [segments[i, :, :] for i in range(segments.shape[0])]
    if len(segments) <= 0:
        return []
    ring = Node(segments[0][1, :], segments[0][0, :])
    segments.pop(0)
    rings = []
    while len(segments) > 0:
        miss = True
        i = 0
        #TODO make binary search
        while i < len(segments):
            a = np.linalg.norm(np.subtract(ring.end, segments[i]), axis=1) < APPROX_ZERO
            # check for duplicate segment
            if ((np.linalg.norm(np.subtract(ring.start, - segments[i]), axis=1) < APPROX_ZERO) | a).all():
                segments.pop(i)
                miss = False
                continue

            # if the end matches
            if a.any():
                miss = False
                segment = segments.pop(i)
                if a[0]:
                    ring.next = Node(segment[0, :], segment[1, :])
                else:
                    ring.next = Node(segment[1, :], segment[0, :])
                if ring.root is None:
                    ring.next.root = ring
                else:
                    ring.next.root = ring.root
                ring = ring.next

                # check to see if loop closed
                if np.linalg.norm(ring.root.start - ring.end) < APPROX_ZERO:
                    ring.next = ring.root
                    rings.append(ring.root)
                    if len(segments) > 0:
                        ring = Node(segments[0][1, :], segments[0][0, :])
                        segments.pop(0)
                continue
            
            # no match go to next one
            i += 1
        if miss and len(segments) > 0:
            # bad ring
            ring = Node(segments[0][1, :], segments[0][0, :])
            segments.pop(0)


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
    newPoints = []
    for i in range(points.shape[0]):
        a = points[i- 1, :] - points[i - 2, :]
        b = points[i - 1, :] - points[i, :]
        c = a * np.linalg.norm(b) + b * np.linalg.norm(a)
        c = c / np.linalg.norm(c) * radius
        newPoints.append(points[i - 1] + c)
    return Polygon(shell=newPoints).convex_hull


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