# Functions for 2 light coefficient solving

import numpy as np
from typing import *

def ransac_2d_subspace(pts: np.ndarray, 
                      inlier_angle: Optional[float] = None,
                      iter_count: Optional[int] = None,
                      inliers: Optional[int] = None,
                      M: Optional[int] = None):
    """
    RANSAC implementation for finding 2D subspace.
    """
    if M is None:
        M = 2
    if inliers is None:
        inliers = pts.shape[1] // 2
    if inlier_angle is None:
        inlier_angle = np.pi/180
    if iter_count is None:
        iter_count = 100

    max_inlier = 0
    n0_max = None
    
    for _ in range(iter_count):
        if pts.shape[1] < 2:
            continue
            
        x0 = np.random.choice(pts.shape[1], 2, replace=False)
        n0 = np.cross(pts[:, x0[0]], pts[:, x0[1]])
        norm = np.linalg.norm(n0)
        if norm < 1e-10:
            continue
        n0 = n0 / norm
        
        ang = np.abs(np.arcsin(np.clip(n0 @ pts, -1, 1)))
        in_count = np.sum(ang <= inlier_angle)
        
        if max_inlier <= in_count:
            max_inlier = in_count
            n0_max = n0

    print(f'Found {max_inlier/pts.shape[1]:.3f} %')
    
    if n0_max is None:
        n0_max = np.array([0, 0, 1])
    
    pts_filtered = pts[:, np.abs(np.arcsin(np.clip(n0_max @ pts, -1, 1))) <= inlier_angle]
    if pts_filtered.shape[1] < 2:
        n0 = n0_max
        # Create orthogonal basis in the plane perpendicular to n0
        v1 = np.array([1, 0, 0])
        if abs(n0 @ v1) > 0.9:
            v1 = np.array([0, 1, 0])
        nx1 = v1 - (v1 @ n0) * n0
        nx1 = nx1 / np.linalg.norm(nx1)
        nx2 = np.cross(n0, nx1)
        nx = np.column_stack([nx1, nx2])
    else:
        U, _, _ = np.linalg.svd(pts_filtered, full_matrices=False)
        n0 = U[:, -1]
        nx = U[:, :2]
    
    return n0, nx

def est_two_light_coeff(gamma: np.ndarray, 
                       mask_t: np.ndarray, 
                       cfactor: float):
    """
    Estimate lighting coefficients for two-light scenario.
    """
    # Get valid points using mask
    valid_points = gamma[:, mask_t.flatten() > 0]
    
    if valid_points.shape[1] < 2:
        # Handle degenerate case
        return np.array([1, 0, 0]), np.array([0, 1, 0])
    
    # Find 2D subspace using RANSAC
    n0, nx = ransac_2d_subspace(valid_points, 0.3*np.pi/180, 1000)
    
    # Generate circle points
    theta = np.arange(0, 361, 0.1) * np.pi/180
    circ = nx @ np.vstack([np.cos(theta), np.sin(theta)])
    
    # Find indices of masked points
    iit = np.where(mask_t.flatten() > 0)[0]
    
    if len(iit) == 0:
        # Handle case with no valid points
        return np.array([1, 0, 0]), np.array([0, 1, 0])
    
    # Find maximum projection angles for each point
    idx = np.zeros(len(iit), dtype=int)
    for ii in range(len(iit)):
        proj = circ.T @ gamma[:, iit[ii]]
        idx[ii] = np.argmax(proj)
    
    # Calculate histogram of angles
    h_idx, _ = np.histogram(idx, bins=np.arange(circ.shape[1] + 1))
    
    # Calculate cutoff threshold
    cutoff = cfactor * np.sum(h_idx) / len(theta)
    
    # Find angles where histogram exceeds cutoff
    above_cutoff = np.where(h_idx > cutoff)[0]
    
    if len(above_cutoff) == 0:
        # Handle case where no peaks are found
        theta1, theta2 = 0, np.pi
    else:
        theta1 = theta[np.min(above_cutoff)]
        theta2 = theta[np.max(above_cutoff)]
    
    # Calculate final illuminant vectors
    illum1 = nx @ np.array([[np.cos(theta1)], 
                           [np.sin(theta1)]])
    illum2 = nx @ np.array([[np.cos(theta2)], 
                           [np.sin(theta2)]])
    
    return illum1.flatten(), illum2.flatten()