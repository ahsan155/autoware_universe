import math 
import random 
import time 
import os
import numpy as np
import torch




def reorder_agents(tensor, num_agents, reorder_idx):
    # tensor: [num_agents, num_agents, ...]
    # Gather along dim=1 (agent dim) using reorder_idx per scenario (batch)
    idx_shape = [num_agents, num_agents] + [1] * (tensor.dim() - 2)
    gather_idx = reorder_idx.view(*idx_shape).expand_as(tensor)
    return torch.gather(tensor, 1, gather_idx)


def build_model_inputs(agent_buffers, lane_centerlines, lane_attrs, is_intersections):
    
    filtered_agents = {}
    cur_pos_list = []
    cur_heading_list = []
    actor_ids = []

    for agent_id, data in agent_buffers.items():
        if len(data["pos"]) == 50:
            filtered_agents[agent_id] = data
            actor_ids.append(agent_id)
            cur_pos_list.append(data["pos"][49])
            cur_heading_list.append(data["heading"][49])

        
    if len(actor_ids) == 0:
        return False, None, None

    num_nodes = len(filtered_agents)
    cur_pos = torch.tensor(cur_pos_list, dtype=torch.float)  # [num_nodes, 2]
    cur_heading = torch.tensor(cur_heading_list, dtype=torch.float)  # [num_nodes]

    # Collect raw numpy arrays from buffers for positions, heading, velocity for ease
    pos_hist = np.zeros((num_nodes, 110, 2), dtype=np.float32)
    heading_hist = np.zeros((num_nodes, 110), dtype=np.float32)
    velocity_hist = np.zeros((num_nodes, 110), dtype=np.float32)
    x_attr = torch.zeros((num_nodes, 3), dtype=torch.uint8)
    scored_agents_mask = torch.ones(num_nodes, dtype=bool)

    for i, (agent_id, data) in enumerate(filtered_agents.items()):
        timesteps = len(data["pos"])
        pos_hist[i, :timesteps, :] = np.array(data["pos"], dtype=np.float32)
        heading_hist[i, :timesteps] = np.array(data["heading"], dtype=np.float32)
        velocity_hist[i, :timesteps] = np.array(data["velocity"], dtype=np.float32)
        x_attr[i] = torch.tensor([0, 1, 0], dtype=torch.uint8)
        

    # Convert to torch tensors
    pos_hist = torch.tensor(pos_hist)         # [num_nodes, 110, 2]
    heading_hist = torch.tensor(heading_hist) # [num_nodes, 110]
    velocity_hist = torch.tensor(velocity_hist) # [num_nodes, 110]

    # start adding scenario dimension to tensors, each scenario with distinct focal agent 
    # Create tensors for outputs: shape [num_nodes, num_nodes, 110, ...]
    x = torch.zeros(num_nodes, num_nodes, 110, 2)
    x_heading = torch.zeros(num_nodes, num_nodes, 110)
    x_velocity = torch.zeros(num_nodes, num_nodes, 110)
    padding_mask = torch.ones(num_nodes, num_nodes, 110, dtype=torch.bool)

    # Compute each scenario i's coordinate frame (origin + rotation) from pos_hist[i], heading_hist[i]
    for i in range(num_nodes):
        origin = pos_hist[i, 49]   # focal agent position at last obs [2]
        heading_i = heading_hist[i, 49] # focal agent heading history [1]

        cos_theta = torch.cos(-heading_i)
        sin_theta = torch.sin(-heading_i)
        # Rotation matrices from focal agent headings
        rot_matrix = torch.tensor([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ], device=pos_hist.device)  # shape [2, 2]

        # For all agents j, compute relative positions in agent i's frame for all timesteps
        rel_pos = pos_hist - origin  # [num_nodes, 110, 2]
        # Rotate all agents' positions for all timesteps by the single rotation matrix
        # rel_pos: [num_nodes, 110, 2], want to multiply last dim (2) by [2,2]
        rel_pos_rot = torch.matmul(rel_pos, rot_matrix.T)  # shape [num_nodes, 110, 2]

        # Assign to output
        x[i] = rel_pos_rot  # [num_nodes, 110, 2] for scenario i

        # Relative heading: (heading_j - heading_i) wrapped between [-pi, pi]
        # Shape: [num_nodes, 110]
        rel_heading = (heading_hist - heading_i)  # broadcasting [num_nodes, 110] - [110]
        rel_heading = (rel_heading + np.pi) % (2 * np.pi) - np.pi
        x_heading[i] = rel_heading

        # Velocity assumed absolute; optionally transform same; assign directly
        x_velocity[i] = velocity_hist  # no relative conversion

        padding_mask[i, :, :50] = False
    
    num_agents = num_nodes
    device = x.device
    reorder_idx = torch.stack([
        torch.cat([torch.tensor([i], device=device), torch.arange(num_agents, device=device)[torch.arange(num_agents) != i]])
        for i in range(num_agents)
    ], dim=0)  # [num_agents, num_agents]
    
   

    x = reorder_agents(x, num_agents, reorder_idx)
    x_heading = reorder_agents(x_heading, num_agents, reorder_idx)
    x_velocity = reorder_agents(x_velocity, num_agents, reorder_idx)
    x_attr = x_attr.unsqueeze(0).repeat(num_agents, 1, 1)
    x_attr = reorder_agents(x_attr, num_agents, reorder_idx)
    scored_agents_mask = scored_agents_mask.repeat(num_agents, 1)



    # creation of focal-agent centric lane------------------------------------------------------------------------------------------
    focal_pos = np.asarray(cur_pos)         # [S, 2]
    focal_heading_rad = np.asarray(cur_heading)  # [S]
    cos_theta = np.cos(focal_heading_rad)     # [S]
    sin_theta = np.sin(focal_heading_rad)     # [S]
    # Rotation matrices [S, 2, 2]
    rot_mats = np.stack([
        np.stack([cos_theta, -sin_theta], axis=1),
        np.stack([sin_theta,  cos_theta], axis=1),
    ], axis=2)  # shape [S, 2, 2]

    # Subtract focal origin—broadcast: [L, 20, 2] - [S, 1, 1, 2]
    rel_points = lane_centerlines[np.newaxis, :, :, :] - focal_pos[:, np.newaxis, np.newaxis, :]
    lane_positions_transformed = np.empty((len(focal_pos), lane_centerlines.shape[0], lane_centerlines.shape[1], 2), dtype=np.float32)
    for s in range(len(focal_pos)):
        lane_positions_transformed[s] = rel_points[s] @ rot_mats[s].T  # [L, 20, 2]

    lane_positions_torch = torch.from_numpy(lane_positions_transformed)  # [S, L, 20, 2]
    lane_attrs = lane_attrs.unsqueeze(0).expand(num_agents, -1, -1) # (S, num_lanes, 3)
    is_intersections = is_intersections.unsqueeze(0).expand(num_agents, -1) # (S, num_lanes)

    # lane_positions_torch: [S, num_lanes, 20, 2]
    lanes_ctr = lane_positions_torch[:, :, 9:11, :].mean(dim=2)  # [S, num_lanes, 2]
    lanes_angle = torch.atan2(
        lane_positions_torch[:, :, 10, 1] - lane_positions_torch[:, :, 9, 1],
        lane_positions_torch[:, :, 10, 0] - lane_positions_torch[:, :, 9, 0],
    )  # [S, num_lanes]

    search_radius = 110
    x_max, x_min = search_radius, -search_radius
    y_max, y_min = search_radius, -search_radius

    lane_padding_mask = (
        (lane_positions_torch[..., 0] > x_max)
        | (lane_positions_torch[..., 0] < x_min)
        | (lane_positions_torch[..., 1] > y_max)
        | (lane_positions_torch[..., 1] < y_min)
    )  # [S, num_lanes, 20]

    invalid_mask = lane_padding_mask.all(dim=-1)  # [S, num_lanes]
    # Set all positions for invalid lanes to 0
    lane_positions_torch = lane_positions_torch.masked_fill(invalid_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
    # Set all attrs/ intersection for invalid lanes to 0 or False as appropriate
    lane_attrs = lane_attrs.masked_fill(invalid_mask.unsqueeze(-1), 0.0)
    is_intersections = is_intersections.masked_fill(invalid_mask, False)
    lanes_ctr = lanes_ctr.masked_fill(invalid_mask.unsqueeze(-1), 0.0)
    lanes_angle = lanes_angle.masked_fill(invalid_mask, 0.0)
    lane_padding_mask = lane_padding_mask | invalid_mask.unsqueeze(-1)   # keep padding_mask shape [S, num_lanes, 20]


    

    # lane based padding for each focal-agent centric scenario-----------------------------------------------------------------------
    lane_samples = lane_positions_torch[:, :, :, :2].reshape(x.shape[0], -1, 2)  # [num_scenarios, num_lanes*20, 2]
    ref_positions = x[:, 0, 49, :]  # [num_scenario, 2], focal agent at idx 0    
    nearest_dist = []
    for s in range(x.shape[0]):
        dists = torch.cdist(ref_positions[s].unsqueeze(0), lane_samples[s])  # [1, num_lanes*20]
        nearest_dist.append(dists.min().item())
    nearest_dist = torch.tensor(nearest_dist, device=x.device)  # [num_scenarios]
    valid_scenario_mask = nearest_dist < 5

    # Apply mask to all tensors
    x = x[valid_scenario_mask]
    x_attr = x_attr[valid_scenario_mask]
    x_heading = x_heading[valid_scenario_mask]
    x_velocity = x_velocity[valid_scenario_mask]
    padding_mask = padding_mask[valid_scenario_mask]
    scored_agents_mask = scored_agents_mask[valid_scenario_mask]
    cur_pos = cur_pos[valid_scenario_mask]
    cur_heading = cur_heading[valid_scenario_mask]

    lane_positions_torch = lane_positions_torch[valid_scenario_mask]
    lane_attrs = lane_attrs[valid_scenario_mask]
    is_intersections = is_intersections[valid_scenario_mask]
    lanes_ctr = lanes_ctr[valid_scenario_mask]
    lanes_angle = lanes_angle[valid_scenario_mask]
    lane_padding_mask = lane_padding_mask[valid_scenario_mask]

    # Now, x shape is [num_valid_scenarios, num_agents, 110, 2]
    agent_last_pos = x[:, :, 49, :]  # [batch, num_agents, 2]
    min_dist = []
    for s in range(x.shape[0]):
        dists = torch.cdist(agent_last_pos[s], lane_samples[s])  # [num_agents, num_lanes*20]
        md = dists.min(dim=1).values  # [num_agents]
        min_dist.append(md)
    min_dist = torch.stack(min_dist, dim=0)  # [batch, num_agents]
    valid_actor_mask = min_dist < 5  # [batch, num_agents]

    # Update padding: True = padded/invalid, False = real/valid
    padding_mask = ~(valid_actor_mask.unsqueeze(-1).expand(-1, -1, x.shape[2])) | padding_mask



    num_agents = x.shape[0]
    reorder_idx = torch.stack([
        torch.cat([torch.tensor([i], device=device), torch.arange(num_agents, device=device)[torch.arange(num_agents) != i]])
        for i in range(num_agents)
    ], dim=0)  # [num_agents, num_agents]    
    cur_pos_all = cur_pos.unsqueeze(0).expand(num_agents, -1, -1)
    cur_heading_all = cur_heading.unsqueeze(0).expand(num_agents, -1)
    cur_pos_all = reorder_agents(cur_pos_all, num_agents, reorder_idx)
    cur_heading_all = reorder_agents(cur_heading_all, num_agents, reorder_idx)

 


    x_ctrs = x[:, :, 49, :2].clone()
    x_positions = x[:, :, :50, :2].clone()
    x_velocity_diff = x_velocity[:, :, :50].clone()

    x[:, :, 50:] = torch.where(
        (padding_mask[:, :, 49].unsqueeze(-1) | padding_mask[:, :, 50:]).unsqueeze(-1),
        torch.zeros(x.shape[0], x.shape[1], 60, 2, device=x.device, dtype=x.dtype),
        x[:, :, 50:] - x[:, :, 49].unsqueeze(-2),
    )
    x[:, :, 1:50] = torch.where(
        (padding_mask[:, :, :49] | padding_mask[:, :, 1:50]).unsqueeze(-1),
        torch.zeros(x.shape[0], x.shape[1], 49, 2, device=x.device, dtype=x.dtype),
        x[:, :, 1:50] - x[:, :, :49],
    )
    x[:, :, 0] = torch.zeros(x.shape[0], x.shape[1], 2, device=x.device, dtype=x.dtype)
    x_velocity_diff[:, :, 1:50] = torch.where(
        (padding_mask[:, :, :49] | padding_mask[:, :, 1:50]),
        torch.zeros(x_velocity_diff.shape[0], x_velocity_diff.shape[1], 49, device=x.device, dtype=x_velocity_diff.dtype),
        x_velocity_diff[:, :, 1:50] - x_velocity_diff[:, :, :49],
    )
    x_velocity_diff[:, :, 0] = torch.zeros(x_velocity_diff.shape[0], x_velocity_diff.shape[1], device=x.device, dtype=x_velocity_diff.dtype)
    

    final_actor_ids = []
    for s in range(x.shape[0]):  # for all valid scenarios in batch
        # Indices of this scenario's agents in the original actor_ids list
        agent_indices = list(range(len(actor_ids)))
        
        # Move focal agent to front (if not already)
        focal_idx = s  # assuming scenario s's focal agent is original agent s
        agent_indices = [focal_idx] + [i for i in agent_indices if i != focal_idx]
        
        # Apply valid_actor_mask (True=valid)
        valid = valid_actor_mask[s].cpu().numpy()
        per_scenario_ids = [actor_ids[idx] for idx, is_valid in zip(agent_indices, valid) if is_valid]
        
        final_actor_ids.append(per_scenario_ids)

    agent_state_data = (x, x_positions, x_ctrs, x_attr, x_heading, x_velocity, x_velocity_diff, padding_mask, scored_agents_mask, final_actor_ids, cur_pos, cur_heading, cur_pos_all, cur_heading_all)
    lanelet_data = (lane_positions_torch, lane_attrs, is_intersections, lanes_ctr, lanes_angle, lane_padding_mask)

    return True, agent_state_data, lanelet_data