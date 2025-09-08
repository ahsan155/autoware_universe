import lanelet2
from lanelet2.core import BasicPoint2d, BasicPoint3d
from lanelet2.routing import RoutingGraph
from lanelet2.projection import LocalCartesianProjector
from lanelet2.io import load, Origin
from collections import deque
import math
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import torch

def classify_intersection_by_geometry(centerline_points, dx_thresh=10, dy_thresh=10, curvature_thresh=1.2):
    # centerline_points: np.array of shape (N, 2)
    start = centerline_points[0]
    end = centerline_points[-1]
    delta_x = abs(end[0] - start[0])
    delta_y = abs(end[1] - start[1])

    # Compute arc length
    diffs = np.diff(centerline_points, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    arc_length = np.sum(seg_lengths)

    # Straight line distance
    straight_dist = np.linalg.norm(end - start)
    curvature_ratio = arc_length / straight_dist if straight_dist > 0 else 1.0

    # Check if lanelet has large displacement in both directions or has high curvature
    is_intersection_candidate = (
        (delta_x > dx_thresh and delta_y > dy_thresh) or
        curvature_ratio > curvature_thresh
    )

    # Exclude long straight vertical or horizontal lines with only one big delta
    is_long_straight_x = (delta_x > dx_thresh and delta_y < 1.0)
    is_long_straight_y = (delta_y > dy_thresh and delta_x < 1.0)

    if is_intersection_candidate and not (is_long_straight_x or is_long_straight_y):
        return True
    else:
        return False


def find_current_lanelets(lanelet_map, pose_x, pose_y, search_radius=10, num_sample=20):
    """
    Find lanelets near the vehicle pose and return those containing the point.
    """
    pt2d = BasicPoint2d(pose_x, pose_y)
    # This returns all lanelets within 'radius' meters of the query point.
    nearby = lanelet2.geometry.findWithin2d(lanelet_map.laneletLayer, pt2d, search_radius)

    lanelets_with_width = []
    for _, lanelet in nearby:
        # Skip non-road lanelets (e.g., crosswalks)
        if "subtype" in lanelet.attributes:
            val = lanelet.attributes["subtype"]
            if val in ["Crosswalk", "Walkway"]:
                continue
        
        # Extract boundaries
        left = np.array([[p.x, p.y] for p in lanelet.leftBound])
        right = np.array([[p.x, p.y] for p in lanelet.rightBound])
        if len(left) < 2 or len(right) < 2:
            continue  # skip degenerate

        # Resample to same number of points
        left_resampled = resample_line(left, num_points=num_sample)
        right_resampled = resample_line(right, num_points=num_sample)

        # Compute width at each sample point
        widths = np.linalg.norm(left_resampled - right_resampled, axis=1)
        
        lanelets_with_width.append((lanelet, widths.mean()))

       
    return lanelets_with_width

def resample_line(points, num_points=20):
    # centerline_points: np.array shape (N, 2), each row is (x, y)
    # Step 1: Compute cumulative arc length
    deltas = np.diff(points, axis=0)
    seg_lengths = np.hypot(deltas[:,0], deltas[:,1])
    arc_lengths = np.insert(np.cumsum(seg_lengths), 0, 0)

    # Step 2: Interpolate x and y as functions of arc length
    interp_x = interp1d(arc_lengths, points[:,0], kind='linear')
    interp_y = interp1d(arc_lengths, points[:,1], kind='linear')

    # Step 3: Create uniform arc length samples
    uniform_samples = np.linspace(0, arc_lengths[-1], num_points)

    # Step 4: Interpolate at uniform samples
    x_new = interp_x(uniform_samples)
    y_new = interp_y(uniform_samples)
    return np.vstack([x_new, y_new]).T  # shape (num_points, 2)

def get_centerline(lanelet_map, ego_pos_x, ego_pos_y):
    current_lanelets_widths = find_current_lanelets(lanelet_map, ego_pos_x, ego_pos_y, search_radius=110)

    lane_centerlines = []        # Will store (20, 2) arrays
    lane_intersections = []      # Will store bools
    lane_attrs = []              # Will store [0, width, is_intersection_int]

    for lanelet, width  in current_lanelets_widths:
        centerline_points = np.array([[pt.x, pt.y] for pt in lanelet.centerline])
        centerline_resampled = resample_line(centerline_points, num_points=20)
        is_intersection = classify_intersection_by_geometry(centerline_resampled, dx_thresh=5, dy_thresh=5, curvature_thresh=1.1)

        lane_centerlines.append(centerline_resampled)
        lane_intersections.append(is_intersection)
        lane_attrs.append([0, width, int(is_intersection)])  # [vehicle_lane, width, intersection_flag]
    
    lane_centerlines = torch.from_numpy(np.stack(lane_centerlines, axis=0)).float()  # (num_lanes, 20, 2)
    lane_attrs = torch.tensor(np.stack(lane_attrs, axis=0)).float()  # (num_lanes, 3)
    #lane_attrs = lane_attrs.unsqueeze(0).expand(num_scenario, -1, -1) # (S, num_lanes, 3)
    is_intersections = torch.tensor(lane_intersections, dtype=torch.bool) # (num_lanes)
    #is_intersections = is_intersections.unsqueeze(0).expand(num_scenario, -1) # (S, num_lanes)
    
    return lane_centerlines, lane_attrs, is_intersections