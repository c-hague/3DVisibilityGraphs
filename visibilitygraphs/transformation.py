"""
Authors
-------
    Collin Hague, chague@uncc.edu
"""
import numpy as np

def transformVector(q: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    applies transformation to input vector q

    Parameters
    ----------
    q : np.ndarray
        1 by 3 input vector
    transform : np.ndarray
        4 by 4 transformation matrix
    
    Returns
    -------
    np.ndarray
        1 by 3 input vector transformed
    """
    p = np.ones((1, 4))
    p[:, 0:3] = q
    r = transform * p
    return r[:, 0:3]

def makeTransformation(rotation, translation) -> np.ndarray:
    """
    Make transformation matrix out of rotation matrix and translation matrix

    Parameters
    ----------
    rotation : np.ndarray
        3 by 3 rotation matrix
    translation : np.ndarray
        1 by 3 translation matrix

    Returns
    ----------
    np.ndarray
        4 by 4 translation matrix
    """
    t = np.eye(4)
    t[0:3, 0:3] = rotation
    t[3, 0:3] = translation
    return t


def makeRotation(axis: str, angle: float):
    """
        make rotation matrix about an axis

        Parameters
        ----------
        axis : str
            x, y or z
        
        angle : float
            radians
        
        Returns
        -------
            3 by 3 rotaion matrix
        
        Raises
        ------
        ValueError
            on incorrect axis
    """
    axis = axis.lower()
    s = np.sin(angle)
    c = np.cos(angle)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError(f'invalid axis {axis}')