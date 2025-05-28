import lanelet2
from lanelet2.core import BasicPoint2d, BasicPoint3d
from lanelet2.routing import RoutingGraph
from collections import deque
import math
import numpy as np

def find_current_lanelets(lanelet_map, pose_x, pose_y, search_radius=10.0):
    """
    Find lanelets near the vehicle pose and return those containing the point.
    """
    pt2d = BasicPoint2d(pose_x, pose_y)
    # Find nearest lanelets (returns list of (distance, lanelet) pairs)
    nearest = lanelet2.geometry.findNearest(lanelet_map.laneletLayer, pt2d, 10)
    current = []
    for _, lanelet in nearest:
        # Skip non-road lanelets (e.g., crosswalks)
        if "subtype" in lanelet.attributes:
            val = lanelet.attributes["subtype"]
            if val in ["Crosswalk", "Walkway"]:
                continue
        # Check if point is inside the lanelet polygon
        if lanelet2.geometry.inside(lanelet, pt2d):
            current.append(lanelet)
    return current

def build_routing_graph(lanelet_map):
    """
    Build a routing graph for vehicular traffic on the map.
    """
    # Use the Lanelet2 traffic rules for vehicles (right-hand traffic assumed)
    #traffic_rules = TrafficRulesFactory.create(lanelet_map, 
    #                TrafficRulesFactory.Type.VEHICLE, lanelet2.traffic_rules.Locations.Germany)
    # Use default routing cost (distance)
    traffic_rules = lanelet2.traffic_rules.create(
        lanelet2.traffic_rules.Locations.Germany,
        lanelet2.traffic_rules.Participants.Vehicle
    )

    graph = RoutingGraph(lanelet_map, traffic_rules)
    return graph

def get_candidate_paths(lanelet_map, routing_graph, start_lanelet, max_distance):
    """
    Enumerate possible lanelet paths up to max_distance ahead of start_lanelet.
    Uses BFS on the routing graph.
    """
    paths = [[start_lanelet]]
    results = []
    # Helper to compute length of a lanelet (arc length)
    def lanelet_length(llet):
        pts = llet.centerline
        length = 0.0
        for i in range(1, len(pts)):
            dx = pts[i].x - pts[i-1].x
            dy = pts[i].y - pts[i-1].y
            length += (dx*dx + dy*dy)**0.5
        return length

 
    queue = deque(paths)
    while queue:
        path = queue.popleft()
        last = path[-1]
        # Compute path length so far
        length_so_far = sum(lanelet_length(l) for l in path)
        if length_so_far > max_distance:
            # path is long enough
            results.append(path)
            continue
        # Extend path by successors
        successors = routing_graph.following(last, withLaneChanges=True)
        for next_lanelet in successors:
            # Avoid loops
            if next_lanelet in path:
                continue
            # Filter out non-road lanelets
            if "subtype" in next_lanelet.attributes:
                val = next_lanelet.attributes["subtype"]
                if val in ["Crosswalk", "Walkway"]:
                    continue
            new_path = path + [next_lanelet]
            queue.append(new_path)
        # If no successors or all filtered, finalize this branch
        if not successors:
            results.append(path)
    return results

def lanelet_sequence_to_trajectory(path, step=1.0):
    """
    Convert a sequence of lanelets into an interpolated trajectory (list of 3D points).
    """
    traj = []
    # Concatenate centerline points
    for i, lanelet in enumerate(path):
        line = lanelet.centerline
        # For intermediate lanelets, skip first point to avoid duplicates
        points = list(line) #.basicLineString()  # gets list of points
        if i > 0 and traj:
            points = points[1:]
        for p in points:
            traj.append((p.x, p.y, p.z))
    # Simple linear interpolation to uniform spacing (could use spline)
    interpolated = []
    if not traj:
        return interpolated
    accumulated = 0.0
    interpolated.append(traj[0])
    for i in range(1, len(traj)):
        (x0,y0,z0) = traj[i-1]
        (x1,y1,z1) = traj[i]
        dx = x1 - x0; dy = y1 - y0; dz = z1 - z0
        segment_len = (dx*dx + dy*dy + dz*dz)**0.5
        steps = max(int(segment_len/step), 1)
        for j in range(1, steps+1):
            t = j/steps
            x = x0 + dx*t; y = y0 + dy*t; z = z0 + dz*t
            interpolated.append((x,y,z))
    return interpolated

def filter_trajectories_by_initial_direction(trajectories, ego_yaw, max_angle_deg=60.0): #  max_angle_deg=60.0
    def angle_diff(a, b):
        d = (a - b + math.pi) % (2*math.pi) - math.pi
        return abs(d)
    thresh = math.radians(max_angle_deg)
    out = []
    for traj in trajectories:
        if len(traj) < 2:
            continue
        dx = traj[1][0] - traj[0][0]
        dy = traj[1][1] - traj[0][1]
        traj_yaw = math.atan2(dy, dx)
        if angle_diff(traj_yaw, ego_yaw) <= thresh:
            out.append(traj)
    return out


def slice_trajectory_ahead_vec(traj_xy, ego_xy):
    """
    Vectorized version of slice_trajectory_ahead.
    traj_xy: (N,2) numpy array of (x,y)
    ego_xy:  (2,) tuple or array
    Returns trimmed trajectory as a (M,2) array.
    """
    pts   = np.asarray(traj_xy, dtype=float)
    ego   = np.asarray(ego_xy,   dtype=float)

    # compute segment vectors and lengths
    vecs  = pts[1:] - pts[:-1]              # shape (N-1,2)
    seg2  = np.sum(vecs**2, axis=1)         # squared norms, shape (N-1,)

    # vector from each segment start to ego
    rel   = ego - pts[:-1]                  # shape (N-1,2)

    # projection parameter u (unclamped), shape (N-1,)
    u_raw = np.sum(rel * vecs, axis=1) / seg2
    u     = np.clip(u_raw, 0.0, 1.0)

    # projection points
    proj = pts[:-1] + (vecs.T * u).T        # shape (N-1,2)

    # distances from ego to each proj
    d2   = np.sum((proj - ego)**2, axis=1)  # squared distances
    idx  = np.argmin(d2)                    # best segment index

    # build the trimmed array
    best_u = u[idx]
    p0, p1 = pts[idx], pts[idx+1]
    proj_pt = p0 + best_u * (p1 - p0)       # single projection

    # stack proj_pt and all subsequent pts
    trimmed = np.vstack([proj_pt, pts[idx+1:]])

    return trimmed


'''

# get possible trajectories--------------------------------------------------------------------
current_lanelets = find_current_lanelets(lanelet_map, vehicle.get_location().x, -vehicle.get_location().y)
graph = build_routing_graph(lanelet_map)
raw_paths = []
for lanelet in current_lanelets:
    paths = get_candidate_paths(lanelet_map, graph, lanelet, max_distance=50.0)
    for path in paths:
        traj = lanelet_sequence_to_trajectory(path, step=0.5)
        raw_paths.append(traj)




ego_yaw = -math.radians(vehicle.get_transform().rotation.yaw)
filtered_paths = filter_trajectories_by_initial_direction(raw_paths, ego_yaw, max_angle_deg=60.0)
if not filtered_paths:
    # you just turned — allow up to 120° until you’re fully on the new lane
    filtered_paths = filter_trajectories_by_initial_direction(raw_paths, ego_yaw, max_angle_deg=120.0)


#print("vehicle yaw degree", vehicle.get_transform().rotation.yaw)
print("raw path", np.array(raw_paths[0]).shape)


sliced_filtered_paths = []
pos_x, pos_y = vehicle.get_location().x, -vehicle.get_location().y
for path in filtered_paths:
    sliced_traj = slice_trajectory_ahead_vec(np.array(path)[:, :2], (pos_x, pos_y))
    sliced_filtered_paths.append(sliced_traj.tolist())

autoware_relative_possible_t = preprocess_and_vectorize_paths(
    sliced_filtered_paths, 
    [pos_x, pos_y], 
    traj_x_scaler=loaded_traj_x_scaler,
    traj_y_scaler=loaded_traj_y_scaler,
    pos_x_scaler=loaded_pos_x_scaler,
    pos_y_scaler=loaded_pos_y_scaler
)


'''