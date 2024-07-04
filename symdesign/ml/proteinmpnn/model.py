"""A number of functions/classes are adopted from: https://github.com/jingraham/neurips19-graph-protein-design"""
from __future__ import annotations

import logging
from collections.abc import Container, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional

TorchFeatures = dict[str, torch.Tensor]


# The following gather functions
def gather_edges(edges: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def gather_nodes_t(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features


def cat_neighbors_nodes(h_nodes: torch.Tensor, h_neighbors: torch.Tensor, E_idx: torch.Tensor) -> torch.Tensor:
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden: int, num_ff: int):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff)
        self.W_out = nn.Linear(num_ff, num_hidden)
        self.act = torch.nn.GELU()

    def forward(self, h_V: torch.Tensor) -> torch.Tensor:
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class EncLayer(nn.Module):
    def __init__(self, num_hidden: int, num_in: int, dropout: float = 0.1, scale: int = 30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W2 = nn.Linear(num_hidden, num_hidden)
        self.W3 = nn.Linear(num_hidden, num_hidden)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W12 = nn.Linear(num_hidden, num_hidden)
        self.W13 = nn.Linear(num_hidden, num_hidden)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(
            self, h_V: torch.Tensor, h_E: torch.Tensor, E_idx: torch.Tensor, mask_V: torch.Tensor = None,
            mask_attend: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ Parallel computation of full transformer layer """
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        # Below same as decoder
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        # Above same as decoder

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayer(nn.Module):
    def __init__(self, num_hidden: int, num_in: int, dropout: float = 0.1, scale: int = 30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W2 = nn.Linear(num_hidden, num_hidden)
        self.W3 = nn.Linear(num_hidden, num_hidden)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(
            self, h_V: torch.Tensor, h_E: torch.Tensor, mask_V: torch.Tensor = None, mask_attend: torch.Tensor = None
    ) -> torch.Tensor:
        """ Parallel computation of full transformer layer """
        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)
        # Below same as encoder
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        # Above same as encoder
        return h_V


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings: int, max_relative_feature: int = 32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        d = torch.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature) * mask + (1 - mask) * (
                    2 * self.max_relative_feature + 1)
        d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E


class CAProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings: int = 16,
                 num_rbf: int = 16, top_k: int = 30, augment_eps: float = 0.):
        """ Extract protein features """
        super(CAProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        # Normalization and embedding
        node_in = 3
        edge_in = num_positional_embeddings + num_rbf * 9 + 7
        self.node_embedding = nn.Linear(node_in, node_features, bias=False)  # NOT USED
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_nodes = nn.LayerNorm(node_features)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _quaternions(self, R: torch.Tensor) -> torch.Tensor:
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
            Rxx - Ryy - Rzz,
            - Rxx + Ryy - Rzz,
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i, j: R[:, :, :, i, j]
        signs = torch.sign(torch.stack([
            _R(2, 1) - _R(1, 2),
            _R(0, 2) - _R(2, 0),
            _R(1, 0) - _R(0, 1)
        ], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(functional.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = functional.normalize(Q, dim=-1)
        return Q

    def _orientations_coarse(
            self, X: torch.Tensor, E_idx: torch.Tensor, eps: float = 1e-6
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dX = X[:, 1:, :] - X[:, :-1, :]
        dX_norm = torch.norm(dX, dim=-1)
        dX_mask = (3.6 < dX_norm) & (dX_norm < 4.0)  # exclude CA-CA jumps
        dX = dX * dX_mask[:, :, None]
        U = functional.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = functional.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = functional.normalize(torch.cross(u_1, u_0), dim=-1)

        # Bond angle calculation
        cosA = -(u_1 * u_0).sum(-1)
        cosA = torch.clamp(cosA, -1 + eps, 1 - eps)
        A = torch.acos(cosA)
        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        # Backbone features
        AD_features = torch.stack((torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), 2)
        AD_features = functional.pad(AD_features, (0, 0, 1, 2), 'constant', 0)

        # Build relative orientations
        o_1 = functional.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = functional.pad(O, (0, 0, 1, 2), 'constant', 0)
        O_neighbors = gather_nodes(O, E_idx)
        X_neighbors = gather_nodes(X, E_idx)

        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3, 3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])

        # Rotate into local reference frames
        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = functional.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1, -2), O_neighbors)
        Q = self._quaternions(R)

        # Orientation features
        O_features = torch.cat((dU, Q), dim=-1)
        return AD_features, O_features

    def _dist(
            self, X: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Pairwise euclidean distances """
        # Convolutional network on NCHW
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
        dX = X.unsqueeze(1) - X.unsqueeze(2)
        # dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        # Todo something about the Ca_only input is mismatched in size
        #  D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        #  RuntimeError: The size of tensor a (3744) must match the size of tensor b (3) at non-singleton dimension 3
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)
        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D: torch.Tensor) -> torch.Tensor:
        # Distance radial basis function
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = D.unsqueeze(-1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def _get_rbf(self, A: torch.Tensor, B: torch.Tensor, E_idx: torch.Tensor) -> torch.Tensor:
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(
            self, Ca: torch.Tensor, mask: torch.Tensor, residue_idx: torch.Tensor, chain_labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ Featurize coordinates as an attributed graph """
        if self.augment_eps > 0:
            Ca = Ca + self.augment_eps * torch.randn_like(Ca)

        D_neighbors, E_idx, mask_neighbors = self._dist(Ca, mask)

        Ca_0 = torch.zeros(Ca.shape, device=Ca.device)
        Ca_2 = torch.zeros(Ca.shape, device=Ca.device)
        Ca_0[:, 1:, :] = Ca[:, :-1, :]
        Ca_1 = Ca
        Ca_2[:, :-1, :] = Ca[:, 1:, :]

        V, O_features = self._orientations_coarse(Ca, E_idx)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca_1-Ca_1
        RBF_all.append(self._get_rbf(Ca_0, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_2, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_0, Ca_1, E_idx))
        RBF_all.append(self._get_rbf(Ca_0, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_1, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_1, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_2, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_2, Ca_1, E_idx))

        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all, O_features), -1)

        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return E, E_idx


class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
                 num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X: torch.Tensor, mask: torch.Tensor, eps: float = 1E-6) -> tuple[torch.Tensor, torch.Tensor]:
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
        dX = X.unsqueeze(1) - X.unsqueeze(2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.size(1)), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D: torch.Tensor) -> torch.Tensor:
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = D.unsqueeze(-1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def _get_rbf(self, A: torch.Tensor, B: torch.Tensor, E_idx: torch.Tensor) -> torch.Tensor:
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(
            self, X: torch.Tensor, mask: torch.Tensor, residue_idx: torch.Tensor, chain_labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            X: (batch length, number_residues, number_atoms, 3)
            mask: (batch length, number_residues)
            residue_idx: (batch length, number_residues)
            chain_labels: (batch length, number_residues)

        Returns:
            The edges and the edge indices of the provided coordinates, residue indices, and chain labels
        """
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]
        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca
        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = [
            self._rbf(D_neighbors),  # Ca-Ca
            self._get_rbf(N, N, E_idx),  # N-N
            self._get_rbf(C, C, E_idx),  # C-C
            self._get_rbf(O, O, E_idx),  # O-O
            self._get_rbf(Cb, Cb, E_idx),  # Cb-Cb
            self._get_rbf(Ca, N, E_idx),  # Ca-N
            self._get_rbf(Ca, C, E_idx),  # Ca-C
            self._get_rbf(Ca, O, E_idx),  # Ca-O
            self._get_rbf(Ca, Cb, E_idx),  # Ca-Cb
            self._get_rbf(N, C, E_idx),  # N-C
            self._get_rbf(N, O, E_idx),  # N-O
            self._get_rbf(N, Cb, E_idx),  # N-Cb
            self._get_rbf(Cb, C, E_idx),  # Cb-C
            self._get_rbf(Cb, O, E_idx),  # Cb-O
            self._get_rbf(O, C, E_idx),  # O-C
            self._get_rbf(N, Ca, E_idx),  # N-Ca
            self._get_rbf(C, Ca, E_idx),  # C-Ca
            self._get_rbf(O, Ca, E_idx),  # O-Ca
            self._get_rbf(Cb, Ca, E_idx),  # Cb-Ca
            self._get_rbf(C, N, E_idx),  # C-N
            self._get_rbf(O, N, E_idx),  # O-N
            self._get_rbf(Cb, N, E_idx),  # Cb-N
            self._get_rbf(C, Cb, E_idx),  # C-Cb
            self._get_rbf(O, Cb, E_idx),  # O-Cb
            self._get_rbf(C, O, E_idx),  # C-O
        ]
        RBF_all = torch.cat(RBF_all, dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        # Find self vs non-self interaction
        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        # embeddings.forward() returns .float() -> torch.float32
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx


class ProteinMPNN(nn.Module):
    log = logging.getLogger(f'{__name__}.ProteinMPNN')
    device: torch.device
    model_name: str

    def __init__(self, num_letters: int, node_features: int, edge_features: int,
                 hidden_dim: int, num_encoder_layers: int = 3, num_decoder_layers: int = 3,
                 k_neighbors: int = 64, augment_eps: float = 0.05, dropout: float = 0.1,
                 ca_only: bool = False):
        super(ProteinMPNN, self).__init__()

        # Hyperparameters
        self.num_letters: int = num_letters
        self.node_features: int = node_features
        self.edge_features: int = edge_features
        self.hidden_dim: int = hidden_dim

        # Featurization layers
        if ca_only:
            self.features = CAProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)
            self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        else:
            self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(num_letters, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self, X: torch.Tensor, S: torch.Tensor, mask: torch.Tensor, chain_M: torch.Tensor,
            residue_idx: torch.Tensor,
            chain_encoding_all: torch.Tensor, randn: torch.Tensor, use_input_decoding_order: bool = False,
            decoding_order=None
    ) -> torch.Tensor:
        """ Graph-conditioned sequence model

        Returns:
            The log of the probabilities for each "residue" in the input.
                Has shape (batch length, sequence length, alphabet length), dtype=torch.float32
        """
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend *= mask.unsqueeze(-1)
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        if not use_input_decoding_order:
            chain_M *= mask  # update chain_M to include missing regions
            # numbers are smaller for places where chain_mask = 0.0 and higher for places where chain_mask = 1.0
            decoding_order = torch.argsort((chain_M + 0.0001) * torch.abs(randn))
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',
                                           1 - torch.triu(torch.ones(mask_size, mask_size, device=device)),
                                           permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)  # [B, L, L]
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see.
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            # Todo - updating weights in place corrupts model trainability!
            # h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_ESV *= mask_bw
            h_ESV += h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = functional.log_softmax(logits, dim=-1)

        return log_probs

    def sample(
            self, X: torch.Tensor, randn: torch.Tensor, S_true: torch.Tensor, chain_mask: torch.Tensor,
            chain_encoding_all: torch.Tensor, residue_idx: torch.Tensor, mask: torch.Tensor = None,
            temperature: float = 1.0, omit_AAs_np: np.ndarray = None, bias_AAs_np: np.ndarray = None,
            chain_M_pos: torch.Tensor | int = 1, omit_AA_mask: torch.Tensor = None, pssm_coef: torch.Tensor = None,
            pssm_bias: torch.Tensor = None, pssm_multi: float = 0., pssm_log_odds_flag: bool = False,
            pssm_log_odds_mask=None, pssm_bias_flag: bool = False, bias_by_res: torch.Tensor = None, **kwargs
    ) -> TorchFeatures:
        """

        Args:
            X:
            randn:
            S_true:
            chain_mask:
            chain_encoding_all:
            residue_idx:
            mask:
            temperature:
            omit_AAs_np:
            bias_AAs_np:
            chain_M_pos:
            omit_AA_mask:
            pssm_coef:
            pssm_bias:
            pssm_multi:
            pssm_log_odds_flag:
            pssm_log_odds_mask:
            pssm_bias_flag:
            bias_by_res:

        Returns:
            A dictionary with torch.Tensor features
                'S', the sampled sequence as integers of indexable alphabet values,
                'probs', the probabilities of each alphabet character at each sequence positions, and
                'decoding_order', the order in which the sequence positions were decoded
        """
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = chain_mask * chain_M_pos * mask  # update chain_M to include missing regions
        # numbers are smaller for places where chain_mask = 0.0 and higher for places where chain_mask = 1.0
        decoding_order = torch.argsort((chain_mask + 0.0001) * (torch.abs(randn)))
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',
                                           (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
                                           permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)  # [B, L, L]
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        # log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, self.num_letters), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
        if omit_AAs_np is None:
            constant = torch.zeros(self.num_letters, device=device)
        else:
            constant = torch.tensor(omit_AAs_np, device=device)

        if omit_AAs_np is None:
            constant_bias = torch.zeros(self.num_letters, device=device)
        else:
            constant_bias = torch.tensor(bias_AAs_np, device=device)

        if bias_by_res is None:
            null_bias_gathered = torch.zeros((N_batch, self.num_letters), device=device)  # [B, 21]

        # chain_mask_combined = chain_mask*chain_M_pos

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_ in range(N_nodes):
            t = decoding_order[:, t_]  # [B]
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:, None])  # [B]
            mask_gathered = torch.gather(mask, 1, t[:, None])  # [B]
            if bias_by_res is None:
                bias_by_res_gathered = null_bias_gathered  # [B, 21]
            else:
                # bias_by_res_gathered = bias_by_res[:, t, :]  # [B, 21] This is how it is written in tied_sample()
                bias_by_res_gathered = torch.gather(bias_by_res, 1, t[:, None, None].repeat(1, 1, self.num_letters))[:,
                                       0, :]  # [B, 21]
            if (mask_gathered == 0).all():  # for padded or missing regions only
                S_t = torch.gather(S_true, 1, t[:, None])
            else:
                # Hidden layers
                E_idx_t = torch.gather(E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1]))
                h_E_t = torch.gather(h_E, 1, t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]))
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1,
                                               t[:, None, None, None].repeat(1, 1, h_EXV_encoder_fw.shape[-2],
                                                                             h_EXV_encoder_fw.shape[-1]))
                mask_t = torch.gather(mask, 1, t[:, None])
                for l, layer in enumerate(self.decoder_layers):
                    # Updated relational features for future states
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(h_V_stack[l], 1, t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]))
                    h_ESV_t = torch.gather(mask_bw, 1, t[:, None, None, None].repeat(1, 1, mask_bw.shape[-2],
                                                                                     mask_bw.shape[
                                                                                         -1])) * h_ESV_decoder_t + h_EXV_encoder_t
                    # # h_ESV_t = mask_bw[:, t:t+1, :, :] * h_ESV_decoder_t + h_EXV_encoder_t
                    # Todo. Make this like tied_sample which was ^
                    #  and is now like this v (memory optimization)
                    #  h_ESV_decoder_t *= mask_bw[:, t:t + 1, :, :]
                    #  h_ESV_decoder_t += h_EXV_encoder_t
                    h_V_stack[l + 1].scatter_(1, t[:, None, None].repeat(1, 1, h_V.shape[-1]),
                                              layer(h_V_t, h_ESV_t, mask_V=mask_t))
                # Sampling step
                h_V_t = torch.gather(h_V_stack[-1], 1, t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1]))[:, 0]
                logits = self.W_out(h_V_t) / temperature
                probs = functional.softmax(logits
                                           - constant[None, :] * 1e8
                                           + constant_bias[None, :] / temperature
                                           + bias_by_res_gathered / temperature, dim=-1)
                if pssm_bias_flag:
                    pssm_coef_gathered = torch.gather(pssm_coef, 1, t[:, None])[:, 0]
                    pssm_bias_gathered = torch.gather(pssm_bias, 1, t[:, None, None].repeat(1, 1, pssm_bias.shape[-1]))[
                                         :, 0]
                    multi_times_coeff = pssm_multi * pssm_coef_gathered[:, None]
                    probs = (1 - multi_times_coeff) * probs + multi_times_coeff * pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = torch.gather(pssm_log_odds_mask, 1, t[:, None, None].repeat(1, 1,
                                                                                                              pssm_log_odds_mask.shape[
                                                                                                                  -1]))[
                                                  :, 0]  # [B, 21]
                    probs_masked = probs * pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked / torch.sum(probs_masked, dim=-1, keepdim=True)  # [B, 21]
                if omit_AA_mask is not None:
                    omit_AA_mask_gathered = torch.gather(omit_AA_mask, 1,
                                                         t[:, None, None].repeat(1, 1, omit_AA_mask.shape[-1]))[:,
                                            0]  # [B, 21]
                    probs_masked = probs * (1.0 - omit_AA_mask_gathered)
                    probs = probs_masked / torch.sum(probs_masked, dim=-1, keepdim=True)  # [B, 21]
                S_t = torch.multinomial(probs, 1)
                all_probs.scatter_(1, t[:, None, None].repeat(1, 1, self.num_letters),
                                   (chain_mask_gathered[:, :, None] * probs[:, None, :]).float())
            S_true_gathered = torch.gather(S_true, 1, t[:, None])
            S_t = (S_t * chain_mask_gathered + S_true_gathered * (1.0 - chain_mask_gathered)).long()
            temp1 = self.W_s(S_t)
            h_S.scatter_(1, t[:, None, None].repeat(1, 1, temp1.shape[-1]), temp1)
            S.scatter_(1, t[:, None], S_t)

        return {'S': S, 'probs': all_probs, 'decoding_order': decoding_order}

    def tied_sample(
            self, X: torch.Tensor, randn: torch.Tensor, S_true: torch.Tensor, chain_mask: torch.Tensor,
            chain_encoding_all: torch.Tensor, residue_idx: torch.Tensor, mask: torch.Tensor,
            temperature: float = 1.0, omit_AAs_np: np.ndarray = None, bias_AAs_np: np.ndarray = None,
            chain_M_pos: torch.Tensor | int = 1, omit_AA_mask: torch.Tensor = None,
            pssm_coef: torch.Tensor = None, pssm_bias: torch.Tensor = None, pssm_multi: float = 0.,
            pssm_log_odds_flag: bool = False, pssm_log_odds_mask: torch.Tensor = None,
            pssm_bias_flag: bool = False, bias_by_res: torch.Tensor = None,
            tied_pos: Iterable[Container[int]] = None, tied_beta: torch.Tensor = None, **kwargs
    ) -> TorchFeatures:
        """

        Args:
            X:
            randn:
            S_true:
            chain_mask:
            chain_encoding_all:
            residue_idx:
            mask:
            temperature:
            omit_AAs_np:
            bias_AAs_np:
            chain_M_pos:
            omit_AA_mask:
            pssm_coef:
            pssm_bias:
            pssm_multi:
            pssm_log_odds_flag:
            pssm_log_odds_mask:
            pssm_bias_flag:
            bias_by_res:
            tied_pos:
            tied_beta: Whether a tied position should be weighted during the summation.
                Shape (batch_length, number_of_nodes)

        Returns:
            A dictionary with torch.Tensor features
                'S', the sampled sequence as integers of indexable alphabet values,
                'probs', the probabilities of each alphabet character at each sequence positions, and
                'decoding_order', the order in which the sequence positions were decoded
        """
        if pssm_bias_flag:
            if pssm_coef is None or pssm_bias is None:
                raise ValueError("Can't use pssm_bias_flag without providing pssm_coef or pssm_bias. These should be "
                                 "type torch.Tensor with shape (batch, sequence_length) and "
                                 "(batch, sequence_length, alphabet) respectively")
        if pssm_log_odds_flag:
            if pssm_log_odds_mask is None:
                raise ValueError("Can't use pssm_log_odds_flag without providing pssm_log_odds_mask. This should be "
                                 "type torch.Tensor with shape (batch, sequence_length, alphabet)")
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        # N_batch, N_nodes, N_neighbors = E_idx.size(0), E_idx.size(1), E_idx.size(2)
        # mask_size = N_nodes
        N_batch, N_nodes = X.size(0), X.size(1)
        N_neighbors = E.size(-1)
        h_V = torch.zeros((N_batch, N_nodes, N_neighbors), device=device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend *= mask.unsqueeze(-1)
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        # All positions that are 0 are not sampled...
        chain_mask = chain_mask * chain_M_pos * mask  # update chain_M to include missing regions
        # Smaller integer for indices where chain_mask = 0.0 and higher for indices where chain_mask = 1.0
        # Todo refactor out randn. This is the only place it is used and just for creating the decoding order...
        decoding_order = torch.argsort((chain_mask + 0.0001) * (torch.abs(randn)))

        if tied_pos is None:
            # Make behavior as '.sample()'
            found_decoding_indices = decoding_order[0].cpu().numpy().tolist()
            tied_decoding_order = [[idx] for idx in found_decoding_indices]
        else:
            tied_decoding_order: list[list[int]] = []
            found_decoding_indices = []
            for idx in decoding_order[0].cpu().numpy().tolist():
                if idx not in found_decoding_indices:
                    for _tied_indices in tied_pos:
                        if idx in _tied_indices:
                            break
                    else:
                        _tied_indices = [idx]

                    # Keep list of lists format
                    tied_decoding_order.append(_tied_indices)
                    # Add all found decoding_indices
                    found_decoding_indices.extend(_tied_indices)

        decoding_order = torch.tensor(found_decoding_indices, device=device)[None].repeat(N_batch, 1)

        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=N_nodes).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',
                                           (1 - torch.triu(torch.ones(N_nodes, N_nodes, device=device))),
                                           permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)  # [B, L, L]
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        # log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, self.num_letters), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
        if omit_AAs_np is None:
            constant = torch.zeros(self.num_letters, device=device)
        else:
            constant = torch.tensor(omit_AAs_np, device=device)

        if omit_AAs_np is None:
            constant_bias = torch.zeros(self.num_letters, device=device)
        else:
            constant_bias = torch.tensor(bias_AAs_np, device=device)

        if bias_by_res is None:
            null_bias_gathered = torch.zeros((N_batch, self.num_letters), device=device)  # [B, 21]

        if tied_beta is None:
            tied_beta = torch.ones_like(mask, device=device)

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for tied_indices in tied_decoding_order:
            # logits = 0.
            logits = torch.zeros((N_batch, self.num_letters), device=device)  # [B, 21]

            # number_tied_positions = len(tied_indices)
            total_tied_coefficient = tied_beta[tied_indices].sum()
            for tied_idx in tied_indices:
                if (mask[:, tied_idx] == 0).all():
                    # No observances at this position are visible, remove beta coefficient from total_tied_coefficient
                    total_tied_coefficient -= tied_beta[tied_idx]
                else:  # At least one position needs to be sampled
                    E_idx_t = E_idx[:, tied_idx:tied_idx + 1, :]
                    h_E_t = h_E[:, tied_idx:tied_idx + 1, :, :]
                    h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                    h_EXV_encoder_t = h_EXV_encoder_fw[:, tied_idx:tied_idx + 1, :, :]
                    mask_t = mask[:, tied_idx:tied_idx + 1]
                    # Always start decoding with a fresh encoding node head h_V_stack[0]
                    # then update additional node head copies with the outcome of each layer
                    # for l_idx, layer in enumerate(self.decoder_layers):
                    last_output_h_V_t = h_V[:, tied_idx, :]
                    for h_V_idx, layer in zip(h_V_stack, self.decoder_layers):
                        h_V_idx[:, tied_idx, :] = last_output_h_V_t
                        h_ESV_decoder_t = cat_neighbors_nodes(h_V_idx, h_ES_t, E_idx_t)
                        h_V_t = h_V_idx[:, tied_idx:tied_idx + 1, :]
                        # h_ESV_t = mask_bw[:, tied_idx:tied_idx+1, :, :] * h_ESV_decoder_t + h_EXV_encoder_t
                        h_ESV_decoder_t *= mask_bw[:, tied_idx:tied_idx + 1, :, :]
                        h_ESV_decoder_t += h_EXV_encoder_t
                        # h_V_stack[l_idx+1][:, tied_idx, :] = layer(h_V_t, h_ESV_decoder_t, mask_V=mask_t).squeeze(1)
                        last_output_h_V_t = layer(h_V_t, h_ESV_decoder_t, mask_V=mask_t).squeeze(1)
                    # h_V_t = h_V_stack[-1][:, tied_idx, :]
                    # h_V_t = last_output_h_V_t

                    # logits += tied_beta[tied_idx] * (self.W_out(h_V_t)/temperature) / number_tied_positions
                    logits += tied_beta[tied_idx] * self.W_out(last_output_h_V_t)

            if torch.any(logits == 0):
                # No information was found for this group
                S_t_repeat = S_true[:, tied_idx]
                probs = all_probs[:, tied_idx, :]
            else:  # Finalize sampling for this group of tied_indices
                # Normalize logits
                logits /= total_tied_coefficient
                # Gather sampling bias
                if bias_by_res is None:
                    bias_by_res_gathered = null_bias_gathered  # [B, 21]
                else:
                    bias_by_res_gathered = bias_by_res[:, tied_idx, :]  # [B, 21]

                probs = functional.softmax(logits / temperature
                                           - constant[None, :] * 1e8
                                           + constant_bias[None, :] / temperature
                                           + bias_by_res_gathered / temperature, dim=-1)
                if pssm_bias_flag:
                    pssm_coef_gathered = pssm_coef[:, tied_idx]
                    pssm_bias_gathered = pssm_bias[:, tied_idx]
                    multi_times_coeff = pssm_multi * pssm_coef_gathered[:, None]
                    probs = (1 - multi_times_coeff) * probs + multi_times_coeff * pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = pssm_log_odds_mask[:, tied_idx]
                    probs_masked = probs * pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked / torch.sum(probs_masked, dim=-1, keepdim=True)  # [B, 21]
                if omit_AA_mask is not None:
                    omit_AA_mask_gathered = omit_AA_mask[:, tied_idx]
                    probs_masked = probs * (1.0 - omit_AA_mask_gathered)
                    probs = probs_masked / torch.sum(probs_masked, dim=-1, keepdim=True)  # [B, 21]
                S_t_repeat = torch.multinomial(probs, 1).squeeze(-1)
                # Hard pick fixed positions
                chain_mask_t = chain_mask[:, tied_idx]
                S_t_repeat = (chain_mask_t * S_t_repeat + (1 - chain_mask_t) * S_true[:, tied_idx]).long()

            h_S_t_repeat = self.W_s(S_t_repeat)
            for tied_idx in tied_indices:
                # h_S[:, tied_idx, :] = self.W_s(S_t_repeat)
                h_S[:, tied_idx, :] = h_S_t_repeat
                S[:, tied_idx] = S_t_repeat
                all_probs[:, tied_idx, :] = probs.float()

        return {'S': S, 'probs': all_probs, 'decoding_order': decoding_order}

    def conditional_probs(
            self, X: torch.Tensor, S: torch.Tensor, mask: torch.Tensor, chain_M: torch.Tensor,
            residue_idx: torch.Tensor,
            chain_encoding_all: torch.Tensor, randn: torch.Tensor, backbone_only: bool = False
    ) -> torch.Tensor:
        """Graph-conditioned coordinate model probabilities dependent on sequence and decoding

        Args:
            X:
            S:
            mask:
            chain_M:
            residue_idx:
            chain_encoding_all:
            randn:
            backbone_only: Whether to use backbone data only during node updates for each designable position
                Same as unconditional_probs, but much slower!

        Returns:
            The log probabilities with shape (batch length, sequence length, alphabet length), dtype=torch.float32
        """
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V_enc = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend *= mask.unsqueeze(-1)
        for layer in self.encoder_layers:
            h_V_enc, h_E = layer(h_V_enc, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)

        chain_M *= mask  # update chain_M to include missing regions

        chain_M_np = chain_M.cpu().numpy()
        idx_to_loop = np.argwhere(chain_M_np[0, :] == 1)[:, 0]
        log_conditional_probs = torch.zeros([X.shape[0], chain_M.shape[1], self.num_letters], device=device).float()

        for idx in idx_to_loop.tolist():
            h_V = torch.clone(h_V_enc)
            # order_mask = torch.zeros(chain_M.shape[1], device=device).float()
            if backbone_only:
                order_mask = torch.ones(chain_M.shape[1], device=device).float()
                order_mask[idx] = 0.
            else:
                order_mask = torch.zeros(chain_M.shape[1], device=device).float()
                order_mask[idx] = 1.
            # numbers are smaller for places where chain_mask = 0.0 and higher for places where chain_mask = 1.0
            decoding_order = torch.argsort((order_mask[None] + 0.0001) * (torch.abs(randn)))
            mask_size = E_idx.shape[1]
            permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
            order_mask_backward = torch.einsum('ij, biq, bjp->bqp',
                                               1 - torch.triu(torch.ones(mask_size, mask_size, device=device)),
                                               permutation_matrix_reverse, permutation_matrix_reverse)
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)  # [B, L, L]
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1. - mask_attend)

            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see.
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                # h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_ESV *= mask_bw
                h_ESV += h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

            logits = self.W_out(h_V)
            log_probs = functional.log_softmax(logits, dim=-1)
            log_conditional_probs[:, idx, :] = log_probs[:, idx, :]

        return log_conditional_probs

    def unconditional_probs(
            self, X: torch.Tensor, mask: torch.Tensor, residue_idx: torch.Tensor, chain_encoding_all: torch.Tensor
    ) -> torch.Tensor:
        """ Graph-conditioned sequence model probabilities independent of decoding and sequence

        Returns:
            The log probabilities with shape (batch length, sequence length, alphabet length), dtype=torch.float32
        """
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        order_mask_backward = torch.zeros([X.shape[0], X.shape[1], X.shape[1]], device=device)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)  # [B, L, L]
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        # mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_V = layer(h_V, h_EXV_encoder_fw, mask)

        logits = self.W_out(h_V)
        log_probs = functional.log_softmax(logits, dim=-1)

        return log_probs

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.model_name})'
