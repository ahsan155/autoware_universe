import numpy as np
import uuid
import math
from autoware_ml_motion_prediction.nodes.path_generator import find_current_lanelets, slice_trajectory_ahead_vec


def is_consecutive(timestamps, expected_delta=0.1, tolerance=0.01):
        for i in range(1, len(timestamps)):
            delta = timestamps[i] - timestamps[i-1]
            if abs(delta - expected_delta) > tolerance:
                return False
        return True

def uuid_to_str(uuid_msg):
    # uuid_msg.uuid is a numpy array of 16 bytes
    return str(uuid.UUID(bytes=bytes(uuid_msg.uuid)))


def preprocess_and_vectorize_paths(
    paths, 
    reference_point, 
    num_paths=3, 
    path_length=29,
    traj_x_scaler=None,
    traj_y_scaler=None,
    pos_x_scaler=None,
    pos_y_scaler=None):
    """
    Preprocess paths, convert to relative, flatten, and concatenate.
    Args:
        paths: List of list of (x, y) tuples.
        reference_point: (x, y) tuple or np.array of shape (2,)
        num_paths: Number of paths to output (default 3)
        path_length: Number of points per path (default 29)
    Returns:
        np.array of shape (num_paths * path_length * 2,)
    """
    # Step 1: Convert input to numpy arrays for vectorization
    np_paths = [np.array(p, dtype=np.float32) for p in paths]

    # Step 2: Normalize x and y in each path using trajectory scalers
    if traj_x_scaler is not None and traj_y_scaler is not None:
        for i, path in enumerate(np_paths):
            if len(path) > 0:
                path[:, 0] = traj_x_scaler.transform(path[:, 0].reshape(-1, 1)).flatten()
                path[:, 1] = traj_y_scaler.transform(path[:, 1].reshape(-1, 1)).flatten()
                np_paths[i] = path

    # Step 3: Ensure exactly num_paths
    if len(np_paths) > num_paths:
        np_paths = np_paths[:num_paths]
    elif len(np_paths) < num_paths:
        while len(np_paths) < num_paths:
            np_paths.append(np_paths[-1].copy())
        
    # Step 4: Pad/truncate each path to path_length
    processed_paths = []
    for path in np_paths:
        if path.shape[0] > path_length:
            path = path[:path_length]
        elif path.shape[0] < path_length:
            if path.shape[0] > 0:
                pad = np.tile(path[-1], (path_length - path.shape[0], 1))
                path = np.vstack([path, pad])
            else:
                path = np.zeros((path_length, 2), dtype=np.float32)
        processed_paths.append(path)

    # Step 4: Convert to relative, flatten, and concatenate
    reference_point = np.array(reference_point, dtype=np.float32)
    if pos_x_scaler is not None and pos_y_scaler is not None:
        ref_x = pos_x_scaler.transform([[reference_point[0]]])[0, 0]
        ref_y = pos_y_scaler.transform([[reference_point[1]]])[0, 0]
        reference_point = np.array([ref_x, ref_y], dtype=np.float32)

    rel_flattened = []
    for path in processed_paths:
        rel_path = path - reference_point  # (29, 2)
        rel_flattened.append(rel_path.flatten())  # (58,)
    result = np.concatenate(rel_flattened)  # (174,)
    return result

def autoware_to_carla_yaw(autoware_yaw_deg):
    """
    Convert Autoware yaw (in degrees, CCW-positive, ENU) to Carla style
    yaw (degrees, CW-positive, Unreal frame).
    """
    carla_yaw = -autoware_yaw_deg
    return (carla_yaw + 180) % 360 - 180






def choose_best_successor(start_ll, routing_graph, ego_yaw_rad):
    cands = routing_graph.following(start_ll)
    if not cands:
        return None
    # compute yaw of each candidate’s first centerline segment
    errors = []

    for ll in cands:
        pts = np.array([[p.x, p.y] for p in ll.centerline], dtype=float)
        
        if len(pts) < 2:
            errors.append((ll, float('inf')))
            continue
        
        min_idx = min(10, pts.shape[0]-1)
        dx, dy = pts[min_idx] - pts[0]
        ll_yaw = math.atan2(dy, dx)   
        
        # minimal angular difference on circle
        diff = ( (ll_yaw - ego_yaw_rad + math.pi) % (2*math.pi) ) - math.pi
        errors.append((ll, abs(diff)))
    
    # pick lowest error
    best_ll, _ = min(errors, key=lambda x: x[1])
    return best_ll

def get_lanelet_chain_with_heading(
    start_ll,
    routing_graph,
    ego_yaw_rad,
    num_ahead=1
):
    chain = [start_ll]
    current = start_ll
    for _ in range(num_ahead):
        best = choose_best_successor(current, routing_graph, ego_yaw_rad)
        if best is None:
            break
        chain.append(best)
        current = best
    return chain


def filter_close_points(pts, min_spacing=0.5):
    if len(pts) == 0:
        return pts
    filtered = [pts[0]]
    for pt in pts[1:]:
        if np.linalg.norm(np.array(pt) - np.array(filtered[-1])) >= min_spacing:
            filtered.append(pt)
    return np.array(filtered, dtype=np.float32)

def avg_n_closest_front(pts, veh, heading_vec, n_points=5):
    if pts.size == 0:
        return 0.0
    vecs_to_pts = pts - veh[None, :]
    in_front_mask = np.dot(vecs_to_pts, heading_vec) > 0
    pts_in_front = pts[in_front_mask]
    if pts_in_front.shape[0] == 0:
        return 0.0
    dists = np.linalg.norm(pts_in_front - veh[None, :], axis=1)
    m = min(n_points, len(dists))
    kth = m - 1
    idx = np.argpartition(dists, kth)[:m]
    
    return float(np.mean(dists[idx]))

def select_best_current_lanelet(cls, veh_xy, ego_yaw_rad, lookahead=5):
    best = None
    best_err = float('inf')
    for ll in cls:
        # 1) get centerline points
        pts = np.array([[p.x, p.y] for p in ll.centerline], dtype=float)
        # 2) slice off behind‐vehicle pts
        pts = slice_trajectory_ahead_vec(pts, veh_xy)
        if len(pts) < 2:
            continue

        # 3) pick a point lookahead steps ahead
        k = min(lookahead, len(pts)-1)
        dx, dy = pts[k] - pts[0]
        ll_yaw = math.atan2(dy, dx)

        # 4) absolute minimal angular difference
        diff = ((ll_yaw - ego_yaw_rad + math.pi) % (2*math.pi)) - math.pi
        err  = abs(diff)

        if err < best_err:
            best_err = err
            best = ll

    return best

def calculate_autoware_lanelet_boundary_dists_with_next(
    lanelet_map,
    routing_graph,
    veh_x,
    veh_y,
    veh_yaw_deg,
    n_points: int = 5,
    num_lanelets_ahead: int = 1,
    lookahead: int = 10
    
):
    ego_yaw_rad = math.radians(veh_yaw_deg)
    # find “current” lanelet
    cls = find_current_lanelets(lanelet_map, veh_x, veh_y)
    if not cls:
        return 0.0, 0.0, 0.0
    start_ll = cls[0]

    # —— NEW: choose the lanelet whose centerline is closest to ego yaw —— 
    start_ll = select_best_current_lanelet(
        cls,
        (veh_x, veh_y),
        ego_yaw_rad,
        lookahead=lookahead
    )
    if start_ll is None:
        start_ll = cls[0]


    # build a chain *aligned with vehicle heading*
    ll_chain = get_lanelet_chain_with_heading(
        start_ll, routing_graph, ego_yaw_rad, num_ahead=num_lanelets_ahead
    )
    
    # Collect all points from the current and next lanelets
    center_pts = []
    left_pts = []
    right_pts = []
    for ll in ll_chain:
        center_pts.extend([[p.x, p.y] for p in ll.centerline])
        left_pts.extend([[p.x, p.y] for p in ll.leftBound])
        right_pts.extend([[p.x, p.y] for p in ll.rightBound])

    center_pts = np.array(center_pts, dtype=np.float32)#[1::2]
    left_pts = np.array(left_pts, dtype=np.float32)
    right_pts = np.array(right_pts, dtype=np.float32)

    center_pts = filter_close_points(center_pts)
    left_pts   = filter_close_points(left_pts)
    right_pts  = filter_close_points(right_pts)


    veh = np.array([veh_x, veh_y], dtype=np.float32)
    yaw_rad = math.radians(veh_yaw_deg)
    heading_vec = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float32)
            
    avg_center = avg_n_closest_front(center_pts, veh, heading_vec, n_points)
    avg_right = avg_n_closest_front(right_pts, veh, heading_vec, n_points)
    avg_left = avg_n_closest_front(left_pts, veh, heading_vec, n_points)

    return avg_center, avg_right, avg_left

