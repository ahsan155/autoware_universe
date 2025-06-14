import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, random_split, SubsetRandomSampler
import numpy as np
import math
import time
import random
import joblib
import matplotlib.pyplot as plt
from autoware_ml_motion_prediction.nodes.model.util import global_to_relative
#from util import global_to_relative

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BoundaryEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BoundaryEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, input_seq):
        output, hidden = self.gru(input_seq)
        return output, hidden

class VehicleStateEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(VehicleStateEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, input_seq):
        output, hidden = self.gru(input_seq)
        return output, hidden
    
class FinalEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FinalEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, input_seq):
        output, hidden = self.gru(input_seq)
        return output, hidden


class EnhancedCombinedEncoder(nn.Module):
    def __init__(self, boundary_input_size, vehicle_state_input_size, hidden_size, final_hidden_size):
        super(EnhancedCombinedEncoder, self).__init__()
        self.boundary_encoder = BoundaryEncoder(boundary_input_size, hidden_size)
        self.vehicle_state_encoder = VehicleStateEncoder(vehicle_state_input_size, hidden_size)
        self.combine_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.final_encoder = FinalEncoder(hidden_size, final_hidden_size)

    def forward(self, boundary_input, vehicle_state_input):
        boundary_output, _ = self.boundary_encoder(boundary_input)
        vehicle_state_output, _ = self.vehicle_state_encoder(vehicle_state_input)
        
        combined_output = torch.cat((boundary_output, vehicle_state_output), dim=2)
        combined_output = self.combine_layer(combined_output)
        
        final_output, final_hidden = self.final_encoder(combined_output)
        
        return final_output, final_hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # query(decoder prev. hidden) shape -> (batch_size, 1, hidden_size)
        # keys(encoder hidden every timestep) shape -> (batch_size, seq_length, hidden_size)
        # scores shape -> (batch_size, seq_length, 1), alignment score for each encoder hidden
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1) # scores shape -> (batch_size, 1, seq_length)
        weights = F.softmax(scores, dim=-1) # scores become 0-1 probability [adds up to 1]
        context = torch.bmm(weights, keys) # context vector, multiplying alpha with encoder hidden states

        return context, weights
    
class MotionPredictionDecoder(nn.Module):
    def __init__(self, encoder, input_size=186, hidden_size=128, output_size=20):
        super(MotionPredictionDecoder, self).__init__()
        self.encoder = encoder
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 + hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, input_tensor, target_tensor, mode="train"):
        batch_size = input_tensor.size(0)
        
        # Convert input and target to relative coordinates
        input_tensor_relative = global_to_relative(input_tensor)
        if mode != "test":
            target_tensor_relative = global_to_relative(target_tensor, input_tensor[:, -1, 3:5])

        decoder_input = torch.zeros(batch_size, 1, 2, device=device)
        decoder_outputs = []
    
        # Initial encoder run
        encoder_output, encoder_hidden = self.encoder(input_tensor_relative[:, :, :3], input_tensor_relative[:, :, 3:])
        decoder_hidden = encoder_hidden
        
        if mode != "test":
            for i in range(10):  # Predict 10 future trajectories
                decoder_output, decoder_hidden = self.forward_step(
                    decoder_input, decoder_hidden, encoder_output
                )
                decoder_outputs.append(decoder_output)

                if mode == "train":
                    # Use ground truth for training (in relative coordinates)
                    next_input = target_tensor[:, i:i+1, :]
                    decoder_input = target_tensor_relative[:, i:i+1, 3:5]
                elif mode == "eval":
                    # Use prediction for inference
                    next_input = target_tensor[:, i:i+1, :].clone()
                    next_input[:, 0, 3:5] = input_tensor[:, -1, 3:5] + decoder_output[:, 0, :2]
                    decoder_input = decoder_output[:, :, :2]

                # Update input_tensor for next iteration 
                input_tensor = torch.cat([input_tensor[:, 1:, :], next_input], dim=1)
                input_tensor_relative = global_to_relative(input_tensor)
                target_tensor_relative = global_to_relative(target_tensor, input_tensor[:, -1, 3:5]) # updated for decoder_input

                # Re-run encoder with updated input
                encoder_output, encoder_hidden = self.encoder(input_tensor_relative[:, :, :3], input_tensor_relative[:, :, 3:])
                decoder_hidden = encoder_hidden

            decoder_outputs = torch.cat(decoder_outputs, dim=1)
        
            return decoder_outputs, decoder_hidden, None
        else:
            decoder_output, decoder_hidden = self.forward_step(
                    decoder_input, decoder_hidden, encoder_output
                )
            
            return decoder_output, decoder_hidden, None


    def forward_step(self, input, hidden, encoder_outputs):
        query = hidden.permute(1, 0, 2)
        context, _ = self.attention(query, encoder_outputs)
        input_gru = torch.cat((input, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden
