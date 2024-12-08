import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
from vector_quantize_pytorch import VectorQuantize

import preprocess.args
from vqtorch.nn import VectorQuant

from models.losses import MEDLoss, entropy_regularization, avg_individual_entropy


def create_adjacency_matrix(connections, num_nodes):
    # Initialize adjacency matrix with zeros
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)

    # Fill in the adjacency matrix based on connections
    adj_matrix[connections[0], connections[1]] = 1

    return adj_matrix


class VQGNN(nn.Module):
    """
    Vector Quantized Graph Neural Network (VQ-GNN) model.

    """
    def __init__(
            self,
            num_joints,
            num_channels,
            edge_index,
            seq_size,
            hidden_dim=16,
            num_layers=4,
            dropout_ratio=0,
            normalization=True,
            codebook_size=512,
            decay=0.99,
            commitment_weight=0.25,
            lamb_entropy=0.03,
            lamb_diff_seq=0.03,
            lamb_node=0.001,
            lamb_dist=0.001,
            lamb_speed=0.1,
            lamb_acceleration=0.1,
            vq_second_args=None,
            use_vq_first=True,
            ):
        super().__init__()
        if vq_second_args is None:
            vq_second_args = {'beta': 0.98, 'kmeans_int': True, 'affine_lr': 0.001, 'sync_nu': 0.0, 'replace_freq': 20}
        self.use_vq_first = use_vq_first
        self.commitment_weight = commitment_weight
        self.codebook_size = codebook_size
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        self.quantized_dim = hidden_dim * num_joints
        self.normalization = normalization
        self.edge_index = edge_index
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.seq_size = seq_size
        self.lamb_speed = lamb_speed
        self.lamb_acceleration = lamb_acceleration
        # Loss weights
        self.lamb_entropy = lamb_entropy
        self.lamb_node = lamb_node
        self.lamb_dist = lamb_dist
        if dropout_ratio != "None":
            self.dropout = nn.Dropout(dropout_ratio)
        else:
            self.dropout = nn.Identity()

        self.conv1 = GCNConv(self.num_channels, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.conv5 = GCNConv(hidden_dim, hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim)
        self.conv6 = GCNConv(hidden_dim, hidden_dim)
        self.bn6 = nn.BatchNorm1d(hidden_dim)
        self.conv7 = GCNConv(hidden_dim, hidden_dim)
        self.bn7 = nn.BatchNorm1d(hidden_dim)
        self.conv8 = GCNConv(hidden_dim, hidden_dim)
        self.bn8 = nn.BatchNorm1d(hidden_dim)
        self.conv9 = GCNConv(hidden_dim, hidden_dim)
        self.bn9 = nn.BatchNorm1d(hidden_dim)
        self.conv10 = GCNConv(hidden_dim, hidden_dim)

        # double the amount of layers



        self.latent_layer = nn.Linear(hidden_dim*num_joints, hidden_dim*num_joints)


        # Vector Quantization layer
        self.vq_first = VectorQuantize(
            dim=hidden_dim*num_joints,
            codebook_size=self.codebook_size,
            # de
            decay=decay,
            commitment_weight=commitment_weight,
            use_cosine_sim=False
        )
        self.vq_second = VectorQuant(
            feature_size=hidden_dim*num_joints,  # feature dimension corresponding to the vectors
            num_codes=self.codebook_size,  # number of codebook vectors
            beta=vq_second_args['beta'],  # (default: 0.95) commitment trade-off
            kmeans_init=vq_second_args['kmeans_int'],  # (default: False) whether to use kmeans++ init
            norm=None,  # (default: None) normalization for input vector
            cb_norm=None,  # (default: None) normalization for codebook vectors
            affine_lr=vq_second_args['affine_lr'],  # (default: 0.0) lr scale for affine parameters
            sync_nu=vq_second_args['sync_nu'],  # (default: 0.0) codebook synchronization contribution
            replace_freq=vq_second_args['replace_freq'],  # (default: 0) frequency to replace dead codes
            dim=-1,  # (default: -1) dimension to be quantized
            inplace_optimizer=vq_second_args.get('inplace_optimizer', None),  # (default: False) whether to update codebook in-place
        ).cuda()
        activation = nn.LeakyReLU()

        self.linear_decode1 = nn.Linear(hidden_dim*num_joints, hidden_dim*num_joints)
        # Decoder layers for reconstruction
        self.decoder_node = nn.Linear(hidden_dim*num_joints, 3*num_joints)
        self.decoder_edge = nn.Linear(hidden_dim*num_joints, num_joints*num_joints)
        if preprocess.args.add_features:
            self.decoder_speed = nn.Linear(hidden_dim*num_joints, num_joints)
            self.decoder_acceleration = nn.Linear(hidden_dim*num_joints, num_joints)
            # self.decoder_indice = nn.Linear(hidden_dim*num_joints, num_joints)

        # Final linear layer only for label prediction
        # self.linear = nn.Linear(final_hidden_dim, output_dim)

    def encode(self, x, edge_index, batch_index):
        x = self.conv1(x, edge_index)
        x = F.gelu(x, approximate='tanh')
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.gelu(x, approximate='tanh')
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.gelu(x, approximate='tanh')
        x = self.bn3(x)
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        x = F.gelu(x, approximate='tanh')
        if self.num_layers == 4:
            return x
        x = self.bn4(x)
        x = self.dropout(x)
        x = self.conv5(x, edge_index)
        x = F.gelu(x, approximate='tanh')
        x = self.bn5(x)
        x = self.dropout(x)
        x = self.conv6(x, edge_index)
        x = F.gelu(x, approximate='tanh')
        if self.num_layers == 6:
            return x
        x = self.bn6(x)
        x = self.dropout(x)
        x = self.conv7(x, edge_index)
        x = F.gelu(x, approximate='tanh')
        x = self.bn7(x)
        x = self.dropout(x)
        x = self.conv8(x, edge_index)
        x = F.gelu(x, approximate='tanh')
        if self.num_layers == 8:
            return x
        x = self.bn8(x)
        x = self.dropout(x)
        x = self.conv9(x, edge_index)
        x = F.gelu(x, approximate='tanh')
        x = self.bn9(x)
        x = self.dropout(x)
        x = self.conv10(x, edge_index)
        x = F.gelu(x, approximate='tanh')
        return x

    def encode_quantize(self, x, edge_index, batch_index):
        # GNN layers
        x = self.encode(x, edge_index, batch_index)
        # Get latent representation from stacked graph
        b_size = torch.unique(batch_index).size(0)
        dim = x.size(1)
        x = x.view(b_size * self.seq_size, self.num_joints, dim)  # Reshape into (batch_size, 32, dim)
        x = x.view(b_size * self.seq_size, self.num_joints*dim)  # Flatten each set of 32 * 16 into a vector of size 512
        x = self.latent_layer(x)

        if self.use_vq_first:
            quantized, indices, commit_loss = self.vq_first(x)
        else:
            quantized, vq_dict = self.vq_second(x)
            indices = vq_dict['q'].squeeze()
            commit_loss = vq_dict['loss']
        return quantized, indices, commit_loss

    def decode_graph(self, graph_z):
        """
        Decodes a latent vector into a continuous graph representation
        consisting of node types and edge types.
        """
        # Pass through shared layers with leaky ReLU activation
        z = self.linear_decode1(graph_z)
        z = F.gelu(z, approximate='tanh')

        # Decode atom types
        joint_pos = self.decoder_node(z)
        joint_pos = joint_pos.view(-1, self.num_joints, 3)

        if preprocess.args.add_features:
            speed_pred = self.decoder_speed(z)
            acceleration_pred = self.decoder_acceleration(z)
            return joint_pos, speed_pred, acceleration_pred

        # Decode edge types
        edge_logits = self.decoder_edge(z).sigmoid()
        edge_matrix = edge_logits.view(-1, self.num_joints, self.num_joints)

        return joint_pos#, edge_matrix


    def forward(self, data):
        # Get data
        x, edge_index, batch, distances = data.x, data.edge_index, data.batch, data.distance1.squeeze(0)
        b_size = torch.unique(batch).size(0)
        x = x.squeeze(-1)
        real_node = x.view(-1, self.num_joints, self.num_channels)
        if preprocess.args.add_features:
            speed = real_node[:, :, 4]
            acceleration = real_node[:, :, 5]
            real_node = real_node[:, :, :3]


        adj = to_dense_adj(edge_index[:, :(self.num_joints-1)*2]).squeeze(0).to(x.device)
        adj_batch = adj.unsqueeze(0).repeat(b_size, 1, 1)

        # Reconstruction
        quantized, indices, commit_loss = self.encode_quantize(x, edge_index, batch)

        indices = indices.view(len(data.y), -1)

        entropy_loss = entropy_regularization(indices, self.codebook_size)

        individual_entropy_loss = avg_individual_entropy(indices)




        if preprocess.args.add_features:
            joints, speed_pred, acceleration_pred = self.decode_graph(quantized)
            # mse is best for tennsor of (B*F*N)

            speed_loss = F.mse_loss(speed_pred, speed)
            acceleration_loss = F.mse_loss(acceleration_pred, acceleration)
        else:
            speed_loss = torch.tensor(0.0)
            acceleration_loss = torch.tensor(0.0)
            joints = self.decode_graph(quantized)
        # Losses
        feature_rec_loss = (MEDLoss()(real_node, joints))

        #feature_rec_loss = self.lamb_node * MEDLoss()(x, joints)

        start_coords = joints[:, self.edge_index[0]]  # Shape: [Batch, Edges, Channels]
        end_coords = joints[:, self.edge_index[1]]  # Shape: [Batch, Edges, Channels]
        vectors = end_coords - start_coords  # Shape: [Batch, Edges, Channels]
        pred_distances = torch.linalg.vector_norm(vectors, dim=2)  # Shape: [Batch, Edges]
        distance_loss = (self.lamb_dist * F.mse_loss(distances.float(), pred_distances.float()))
        all_distance = True
        if all_distance:
            pred_all_distances = torch.cdist(joints, joints, p=2)
            pred_all_distances = torch.triu(pred_all_distances, diagonal=1)
            real_all_distances = torch.cdist(real_node, real_node, p=2)
            real_all_distances = torch.triu(real_all_distances, diagonal=1)

            all_distance_loss = ( F.mse_loss(real_all_distances.float(), pred_all_distances.float()))
            distance_loss = all_distance_loss



        #adj_quantized = torch.clamp(adj_quantized, min=0, max=1)
        #edge_rec_loss = self.lamb_edge * F.cross_entropy(adj_batch, edge_matrix)

        #dist = torch.squeeze(dist)
        total_loss = (feature_rec_loss + commit_loss + distance_loss) # edge_rec_loss +
        loss_tuple = (self.lamb_node * feature_rec_loss, self.commitment_weight*commit_loss, self.lamb_dist * distance_loss, self.lamb_entropy * individual_entropy_loss,
                      self.lamb_speed * speed_loss, self.lamb_acceleration * acceleration_loss)
        loss_no_weights = (feature_rec_loss, commit_loss, distance_loss, entropy_loss, individual_entropy_loss, speed_loss, acceleration_loss)


        return joints, pred_distances, loss_no_weights, loss_tuple, quantized

    def get_indices(self, data, with_quantized=False):
        # we recieve the data as shape and return the indices
        seq = data.x.squeeze(-1)  # Shape:

        quantized, indices, commit_loss = self.encode_quantize(seq, data.edge_index, data.batch)

        # Reshape indices to [B, F]
        indices = indices.view(len(data.y), -1)

        # reshape quantized to [B, F, Q]
        quantized = quantized.view(len(data.y), indices.size(1), -1)

        if with_quantized:
            return indices, quantized

        return indices

    def get_frames(self, indices, permute=True):
        """
        Reconstructs the sequence from the indices.
        Recieves the indices as shape [B, F] and returns the reconstructed sequence.
        :param indices:
        :return:
        """
        B = indices.size(0)
        Fr = indices.size(1)
        codebook = self.vq_second.get_codebook()
        quantized_list = []
        for subject in range(B):
            for frame in range(Fr):
                quantized_list.append(codebook[indices[subject, frame]])
        quantized = torch.stack(quantized_list, dim=0)

        # quantized is now a tensor of shape [B * F, Q]
        # make sure is [B*F, Q]
        decode = quantized.view(-1, self.quantized_dim)

        quantized = quantized.view(B, Fr, -1)

        joints = self.decode_graph(decode)

        # Reshape back to [B, F_new, N, C]
        joints = joints.view(B, Fr, 32, 3)

        # Permute to get back to [B, N, C, F_new]
        if permute:
            joints = joints.permute(0, 2, 3, 1)

        return joints, quantized


    def eval_model(self, data):
        # data.x is of shape [B*32*F, 3, 1]
        seq = data.x.squeeze(-1)  # Shape: [B*F*32, 3]
        N = 32  # Number of nodes per graph
        C = 3  # Number of features per node
        B = data.num_graphs  # Number of graphs in the batch


        quantized, indices, commit_loss = self.encode_quantize(seq, data.edge_index, data.batch)

        # Reshape indices to [B, F]
        indices = indices.view(len(data.y), -1)

        # Pass through encode_quantize and decode_graph

        joints = self.decode_graph(quantized)

        quantized = quantized.view(len(data.y), indices.size(1), -1)

        # Reshape back to [B, F_new, N, C]
        joints = joints.view(len(data.y), indices.size(1), N, C)

        seq_fixed = seq.view(len(data.y), indices.size(1), N, C)



        # Now, joints is the reconstructed sequence
        # If needed, you can compute loss or further process joints here
        feature_rec_loss = self.lamb_node * (F.mse_loss(seq_fixed, joints))
        # If you want to rebuild it back into the original F dimension size
        # Initialize an empty tensor to hold the full sequence
        # Permute to get back to [B, N, C, F_new]
        #joints = joints.permute(0, 2, 3, 1)  # Shape: [B, 32, 3, F_new]

        # If desired, you can leave the other frames as zeros or fill in with interpolated values

        return joints, quantized, seq_fixed, feature_rec_loss, indices




