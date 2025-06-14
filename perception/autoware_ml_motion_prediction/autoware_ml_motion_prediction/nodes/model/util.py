import torch

def global_to_relative(trajectory, reference=None):
    if reference is None:
        reference = trajectory[:, 0, 3:5]  # First point of the trajectory
    return trajectory - torch.cat([torch.zeros_like(trajectory[:, :, :3]), reference.unsqueeze(1).repeat(1, trajectory.size(1), 1), torch.zeros_like(trajectory[:, :, 5:])], dim=2)

def relative_to_global(relative_trajectory, reference_point):
    return relative_trajectory + torch.cat([torch.zeros_like(relative_trajectory[:, :, :3]), reference_point.unsqueeze(1).repeat(1, relative_trajectory.size(1), 1), torch.zeros_like(relative_trajectory[:, :, 5:])], dim=2)

def prepare_ground_truth(input_tensor, target_tensor):
    batch_size = input_tensor.size(0)
    ground_truth = []
    
    for i in range(10):  # For each of the 3 iterations
        reference_point = input_tensor[:, -1, 3:5]  # Last point of current input
        future_points = target_tensor[:, i:i+10, 3:5]  # Next 3 points from target
        
        # Convert to relative coordinates
        relative_points = future_points - reference_point.unsqueeze(1)
        ground_truth.append(relative_points.reshape(batch_size, -1))
        
        # Update input_tensor for next iteration (if needed)
        if i < 9:
            input_tensor = torch.cat([input_tensor[:, 1:, :], target_tensor[:, i:i+1, :]], dim=1)
    
    return torch.stack(ground_truth, dim=1)


def relative_to_global_trajectory_old(input_tensor, relative_output, target_tensor):
    batch_size, num_iterations, _ = relative_output.shape
    global_output = torch.zeros(batch_size, num_iterations, 20, device=input_tensor.device)
    
    for i in range(num_iterations):
        reference_point = input_tensor[:, -1, 3:5]  # Last point of input
        
        # Reshape reference_point to [batch_size, 1, 2] for broadcasting
        reference_point = reference_point.unsqueeze(1)
        
        # Reshape relative output for this iteration to [batch_size, 10, 2]
        relative_coords = relative_output[:, i, :].view(batch_size, 10, 2)
        
        # Add reference point to relative coordinates
        global_coords = relative_coords + reference_point
        
        # Flatten the result and assign to global_output
        global_output[:, i, :] = global_coords.view(batch_size, 20)

        next_input = target_tensor[:, i, :].unsqueeze(1)
        input_tensor = torch.cat([input_tensor[:, 1:, :], next_input], dim=1)

    return global_output

def relative_to_global_trajectory_realtime(input_tensor, relative_output): 
    batch_size, num_iterations, _ = relative_output.shape
    global_output = torch.zeros(batch_size, num_iterations, 20, device=input_tensor.device)
    
    for i in range(num_iterations):
        reference_point = input_tensor[:, -1, 3:5]  # Last point of input
        
        # Reshape reference_point to [batch_size, 1, 2] for broadcasting
        reference_point = reference_point.unsqueeze(1)
        
        # Reshape relative output for this iteration to [batch_size, 10, 2]
        relative_coords = relative_output[:, i, :].view(batch_size, 10, 2)
        
        # Add reference point to relative coordinates
        global_coords = relative_coords + reference_point
        
        # Flatten the result and assign to global_output
        global_output[:, i, :] = global_coords.view(batch_size, 20)


    return global_output



def global_to_relative_possible_trajectory(trajectory, reference_point):
    # Reshape the trajectory to (29, 2) for easier processing
    traj = trajectory.reshape(-1, 2)
    # Subtract the reference point from each coordinate
    relative_traj = traj - reference_point
    # Flatten the result back to 58 values
    return relative_traj.flatten()
    
