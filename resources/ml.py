import os
from typing import Annotated

import numpy as np
import torch

from ProteinMPNN.vanilla_proteinmpnn.protein_mpnn_utils import ProteinMPNN
from utils import start_log
from utils.path import protein_mpnn_weights_dir

logger = start_log(name=__name__)
mpnn_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'  # gapped_protein_letters


class ProteinMPNNFactory:
    """Return a ProteinMPNN instance by calling the Factory instance with the ProteinMPNN model name

        Handles creation and allotment to other processes by saving expensive memory load of multiple instances and
        allocating a shared pointer to the named ProteinMPNN model
    """
    mpnn_alphabet_length = len(mpnn_alphabet)

    def __init__(self, **kwargs):
        self._models = {}
        # self._models = None

    def __call__(self, model_name: str = 'v_48_020', backbone_noise: float = 0., **kwargs) -> ProteinMPNN:
        """Return the specified ProteinMPNN object singleton

        Args:
            model_name: The name of the model to use from ProteinMPNN. v_X_Y where X is neighbor distance, and Y is noise
            backbone_noise: The amount of backbone noise to add to the pose during design
        Returns:
            The instance of the initialized ProteinMPNN model
        """
        model_name_key = f'{model_name}_{backbone_noise}'
        model = self._models.get(model_name_key)
        if model:
            return model
        else:  # Create a new ProteinMPNN model instance
            # Acquire a adequate computing device
            device = torch.device('cpu')
            # Todo, reinstate
            #  device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
            checkpoint = torch.load(os.path.join(protein_mpnn_weights_dir, f'{model_name}.pt'),
                                    map_location=device)
            logger.info(f'Number of edges: {checkpoint["num_edges"]}')
            logger.info(f'Training noise level: {checkpoint["noise_level"]} Angstroms')
            hidden_dim = 128
            num_layers = 3
            model = ProteinMPNN(num_letters=self.mpnn_alphabet_length,
                                node_features=hidden_dim,
                                edge_features=hidden_dim,
                                hidden_dim=hidden_dim,
                                num_encoder_layers=num_layers,
                                num_decoder_layers=num_layers,
                                augment_eps=backbone_noise,
                                k_neighbors=checkpoint['num_edges'])
            model.to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model.device = device
            self._models[model_name_key] = model

        return model

    def get(self, **kwargs) -> ProteinMPNN:
        """Return the specified ProteinMPNN object singleton

        Keyword Args:
            model_name - (str) = 'v_48_020' - The name of the model to use from ProteinMPNN.
                v_X_Y where X is neighbor distance, and Y is noise
            backbone_noise - (float) = 0.0 - The amount of backbone noise to add to the pose during design
        Returns:
            The instance of the initialized ProteinMPNN model
        """
        return self.__call__(**kwargs)


proteinmpnn_factory: Annotated[ProteinMPNNFactory,
                               'Calling this factory method returns the single instance of the ProteinMPNN class '
                               'located at the "source" keyword argument'] = \
    ProteinMPNNFactory()
"""Calling this factory method returns the single instance of the Database class located at the "source" keyword 
argument
"""


def batch_proteinmpnn_input(size: int = None, **kwargs) -> dict[str, np.ndarray]:
    """Set up all data for batches of proteinmpnn design"""
    X = kwargs.get('X', None)
    S = kwargs.get('S', None)
    chain_mask = kwargs.get('chain_mask', None)
    chain_encoding = kwargs.get('chain_encoding', None)
    residue_idx = kwargs.get('residue_idx', None)
    mask = kwargs.get('mask', None)
    # omit_AAs_np = kwargs.get('omit_AAs_np', None)
    # bias_AAs_np = kwargs.get('bias_AAs_np', None)
    residue_mask = kwargs.get('chain_M_pos', None)
    omit_AA_mask = kwargs.get('omit_AA_mask', None)
    pssm_coef = kwargs.get('pssm_coef', None)
    pssm_bias = kwargs.get('pssm_bias', None)
    # pssm_multi = kwargs.get('pssm_multi', None)
    # pssm_log_odds_flag = kwargs.get('pssm_log_odds_flag', None)
    pssm_log_odds_mask = kwargs.get('pssm_log_odds_mask', None)
    # pssm_bias_flag = kwargs.get('pssm_bias_flag', None)
    # tied_pos = kwargs.get('tied_pos', None)
    tied_beta = kwargs.get('tied_beta', None)
    # bias_by_res = kwargs.get('bias_by_res', None)

    # Todo make a dynamic solve based on device memory and memory error routine from Nanohedra
    # Stack sequence design task in "batches"
    if size is None:
        size = X.shape[0]
        # Use X as is
        # X = np.tile(X, (size,) + (1,) * X.ndim)
    else:
        X = np.tile(X, (size,) + (1,) * X.ndim)

    S = np.tile(S, (size,) + (1,) * S.ndim)
    mask = np.tile(mask, (size,) + (1,) * mask.ndim)
    residue_mask = np.tile(residue_mask, (size,) + (1,) * residue_mask.ndim)
    chain_mask = np.tile(chain_mask, (size,) + (1,) * chain_mask.ndim)
    chain_encoding = np.tile(chain_encoding, (size,) + (1,) * chain_encoding.ndim)
    residue_idx = np.tile(residue_idx, (size,) + (1,) * residue_idx.ndim)
    omit_AA_mask = np.tile(omit_AA_mask, (size,) + (1,) * omit_AA_mask.ndim)
    tied_beta = np.tile(tied_beta, (size,) + (1,) * tied_beta.ndim)
    pssm_coef = np.tile(pssm_coef, (size,) + (1,) * pssm_coef.ndim)
    pssm_bias = np.tile(pssm_bias, (size,) + (1,) * pssm_bias.ndim)
    pssm_log_odds_mask = np.tile(pssm_log_odds_mask, (size,) + (1,) * pssm_log_odds_mask.ndim)

    return dict(X=X,
                S=S,
                chain_mask=chain_mask,
                chain_encoding=chain_encoding,
                residue_idx=residue_idx,
                mask=mask,
                # omit_AAs_np=omit_AAs_np,
                # bias_AAs_np=bias_AAs_np,
                chain_M_pos=residue_mask,
                omit_AA_mask=omit_AA_mask,
                pssm_coef=pssm_coef,
                pssm_bias=pssm_bias,
                # pssm_multi=pssm_multi,
                # pssm_log_odds_flag=pssm_log_odds_flag,
                pssm_log_odds_mask=pssm_log_odds_mask,
                # pssm_bias_flag=pssm_bias_flag,
                # tied_pos=tied_pos,
                tied_beta=tied_beta,
                # bias_by_res=bias_by_res
                )


def proteinmpnn_to_device(device: str = None, **kwargs) -> dict[str, torch.Tensor]:
    """Set up all data to torch.Tensors for proteinmpnn design

    Args:
        device: The device to load tensors to
    Returns:
        The torch.Tensor proteinmpnn parameters
    """
    if device is None:
        raise ValueError('Must provide the desired device to load proteinmpnn')

    X = kwargs.get('X', None)
    S = kwargs.get('S', None)
    chain_mask = kwargs.get('chain_mask', None)
    chain_encoding = kwargs.get('chain_encoding', None)
    residue_idx = kwargs.get('residue_idx', None)
    mask = kwargs.get('mask', None)
    # omit_AAs_np = kwargs.get('omit_AAs_np', None)
    # bias_AAs_np = kwargs.get('bias_AAs_np', None)
    residue_mask = kwargs.get('chain_M_pos', None)
    omit_AA_mask = kwargs.get('omit_AA_mask', None)
    pssm_coef = kwargs.get('pssm_coef', None)
    pssm_bias = kwargs.get('pssm_bias', None)
    # pssm_multi = kwargs.get('pssm_multi', None)
    # pssm_log_odds_flag = kwargs.get('pssm_log_odds_flag', None)
    pssm_log_odds_mask = kwargs.get('pssm_log_odds_mask', None)
    # pssm_bias_flag = kwargs.get('pssm_bias_flag', None)
    # tied_pos = kwargs.get('tied_pos', None)
    tied_beta = kwargs.get('tied_beta', None)
    # bias_by_res = kwargs.get('bias_by_res', None)

    # Convert all numpy arrays to pytorch
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    residue_mask = torch.from_numpy(residue_mask).to(dtype=torch.float32, device=device)
    chain_mask = torch.from_numpy(chain_mask).to(dtype=torch.float32, device=device)
    chain_encoding = torch.from_numpy(chain_encoding).to(dtype=torch.long, device=device)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    omit_AA_mask = torch.from_numpy(omit_AA_mask).to(dtype=torch.float32, device=device)
    tied_beta = torch.from_numpy(tied_beta).to(dtype=torch.float32, device=device)
    pssm_coef = torch.from_numpy(pssm_coef).to(dtype=torch.float32, device=device)
    pssm_bias = torch.from_numpy(pssm_bias).to(dtype=torch.float32, device=device)
    pssm_log_odds_mask = torch.from_numpy(pssm_log_odds_mask).to(dtype=torch.float32, device=device)
    # torch.from_numpy(omit_aas).to(dtype=torch.float32, device=device)

    return dict(X=X,
                S=S,
                chain_mask=chain_mask,
                chain_encoding=chain_encoding,
                residue_idx=residue_idx,
                mask=mask,
                # omit_AAs_np=omit_AAs_np,
                # bias_AAs_np=bias_AAs_np,
                chain_M_pos=residue_mask,
                omit_AA_mask=omit_AA_mask,
                pssm_coef=pssm_coef,
                pssm_bias=pssm_bias,
                # pssm_multi=pssm_multi,
                # pssm_log_odds_flag=pssm_log_odds_flag,
                pssm_log_odds_mask=pssm_log_odds_mask,
                # pssm_bias_flag=pssm_bias_flag,
                # tied_pos=tied_pos,
                tied_beta=tied_beta,
                # bias_by_res=bias_by_res
                )


def score_sequences(S, log_probs, mask) -> torch.Tensor:
    """Score ProteinMPNN sequences using Negative log likelihood probabilities

    Args:
        S: The sequence tensor
        log_probs: The logarithmic probabilities as found by a forward pass through ProteinMPNN
        mask: Any positions that are masked in the design task
    Returns:
        The loss calculated over the log probabilites compared to the sequence tensor
    """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)),
        S.contiguous().view(-1)).view(S.size())
    return torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
