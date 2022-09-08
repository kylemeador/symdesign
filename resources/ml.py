import os
from typing import Annotated

import numpy as np
import torch

from ProteinMPNN.protein_mpnn_utils import ProteinMPNN
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
            device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
            # device = torch.device('cpu')
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


def batch_proteinmpnn_input(size: int = None,
                            X: np.ndarray = None,
                            S: np.ndarray = None,
                            chain_mask: np.ndarray = None,
                            chain_encoding: np.ndarray = None,
                            residue_idx: np.ndarray = None,
                            mask: np.ndarray = None,
                            chain_M_pos: np.ndarray = None,  # residue_mask
                            omit_AA_mask: np.ndarray = None,
                            pssm_coef: np.ndarray = None,
                            pssm_bias: np.ndarray = None,
                            pssm_log_odds_mask: np.ndarray = None,
                            bias_by_res: np.ndarray = None,
                            # tied_beta: np.ndarray = None,
                            **kwargs) -> dict[str, np.ndarray]:
    # omit_AAs_np: np.ndarray = None, #
    # bias_AAs_np: np.ndarray = None, #
    # pssm_multi: np.ndarray = None, #
    # pssm_log_odds_flag: np.ndarray = None #
    # pssm_bias_flag: np.ndarray = None, #
    # tied_pos: np.ndarray = None, #
    # bias_by_res: np.ndarray = None, #
    """Set up all data for batches of proteinmpnn design

    Args:
        size: The number of inputs to use. If left blank, the size will be inferred from axis=0 of the X array
        X: The array specifying the parameter X of ProteinMPNN
        S: The array specifying the parameter S of ProteinMPNN
        chain_mask: The array specifying the parameter chain_mask of ProteinMPNN
        chain_encoding: The array specifying the parameter chain_encoding of ProteinMPNN
        residue_idx: The array specifying the parameter residue_idx of ProteinMPNN
        mask: The array specifying the parameter mask of ProteinMPNN
        chain_M_pos: The array specifying the parameter residue_mask of ProteinMPNN
        # residue_mask: The array specifying the parameter residue_mask of ProteinMPNN
        omit_AA_mask: The array specifying the parameter omit_AA_mask of ProteinMPNN
        pssm_coef: The array specifying the parameter pssm_coef of ProteinMPNN
        pssm_bias: The array specifying the parameter pssm_bias of ProteinMPNN
        pssm_log_odds_mask: The array specifying the parameter pssm_log_odds_mask of ProteinMPNN
        bias_by_res: The array specifying the parameter bias_by_res of ProteinMPNN
        # tied_beta: The array specifying the parameter tied_beta of ProteinMPNN
    Returns:
        A dictionary with each of the proteinmpnn parameters formatted in a batch
    """
    # Stack sequence design task in "batches"
    if size is None:  # Use X as is
        size = X.shape[0]
    else:
        X = np.tile(X, (size,) + (1,)*X.ndim)

    S = np.tile(S, (size,) + (1,)*S.ndim)
    mask = np.tile(mask, (size,) + (1,)*mask.ndim)
    residue_mask = np.tile(chain_M_pos, (size,) + (1,)*chain_M_pos.ndim)
    chain_mask = np.tile(chain_mask, (size,) + (1,)*chain_mask.ndim)
    chain_encoding = np.tile(chain_encoding, (size,) + (1,)*chain_encoding.ndim)
    residue_idx = np.tile(residue_idx, (size,) + (1,)*residue_idx.ndim)
    omit_AA_mask = np.tile(omit_AA_mask, (size,) + (1,)*omit_AA_mask.ndim)
    # tied_beta = np.tile(tied_beta, (size,) + (1,)*tied_beta.ndim)
    bias_by_res = np.tile(bias_by_res, (size,) + (1,)*bias_by_res.ndim)
    pssm_coef = np.tile(pssm_coef, (size,) + (1,)*pssm_coef.ndim)
    pssm_bias = np.tile(pssm_bias, (size,) + (1,)*pssm_bias.ndim)
    pssm_log_odds_mask = np.tile(pssm_log_odds_mask, (size,) + (1,)*pssm_log_odds_mask.ndim)

    return dict(X=X,
                S=S,
                chain_mask=chain_mask,
                chain_encoding=chain_encoding,
                residue_idx=residue_idx,
                mask=mask,
                chain_M_pos=residue_mask,
                omit_AA_mask=omit_AA_mask,
                pssm_coef=pssm_coef,
                pssm_bias=pssm_bias,
                pssm_log_odds_mask=pssm_log_odds_mask,
                bias_by_res=bias_by_res,
                # tied_beta=tied_beta
                )


def proteinmpnn_to_device(device: str = None,
                          X: np.ndarray = None,
                          S: np.ndarray = None,
                          chain_mask: np.ndarray = None,
                          chain_encoding: np.ndarray = None,
                          residue_idx: np.ndarray = None,
                          mask: np.ndarray = None,
                          chain_M_pos: np.ndarray = None,  # residue_mask
                          omit_AA_mask: np.ndarray = None,
                          pssm_coef: np.ndarray = None,
                          pssm_bias: np.ndarray = None,
                          pssm_log_odds_mask: np.ndarray = None,
                          bias_by_res: np.ndarray = None,
                          tied_beta: np.ndarray = None,
                          **kwargs) -> dict[str, torch.Tensor]:
    # omit_AAs_np = kwargs.get('omit_AAs_np', None)
    # bias_AAs_np = kwargs.get('bias_AAs_np', None)
    # pssm_multi = kwargs.get('pssm_multi', None)
    # pssm_log_odds_flag = kwargs.get('pssm_log_odds_flag', None)
    # pssm_bias_flag = kwargs.get('pssm_bias_flag', None)
    # tied_pos = kwargs.get('tied_pos', None)
    # bias_by_res = kwargs.get('bias_by_res', None)
    """Set up all data to torch.Tensors for proteinmpnn design

    Args:
        device: The device to load tensors to
        X: The array specifying the parameter X of ProteinMPNN
        S: The array specifying the parameter S of ProteinMPNN
        chain_mask: The array specifying the parameter chain_mask of ProteinMPNN
        chain_encoding: The array specifying the parameter chain_encoding of ProteinMPNN
        residue_idx: The array specifying the parameter residue_idx of ProteinMPNN
        mask: The array specifying the parameter mask of ProteinMPNN
        chain_M_pos: The array specifying the parameter residue_mask of ProteinMPNN
        # residue_mask: The array specifying the parameter residue_mask of ProteinMPNN
        omit_AA_mask: The array specifying the parameter omit_AA_mask of ProteinMPNN
        pssm_coef: The array specifying the parameter pssm_coef of ProteinMPNN
        pssm_bias: The array specifying the parameter pssm_bias of ProteinMPNN
        pssm_log_odds_mask: The array specifying the parameter pssm_log_odds_mask of ProteinMPNN
        bias_by_res: The array specifying the parameter bias_by_res of ProteinMPNN
        tied_beta: The array specifying the parameter tied_beta of ProteinMPNN
    Returns:
        The torch.Tensor proteinmpnn parameters
    """
    if device is None:
        raise ValueError('Must provide the desired device to load proteinmpnn')

    # Convert all numpy arrays to pytorch
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    residue_mask = torch.from_numpy(chain_M_pos).to(dtype=torch.float32, device=device)  # residue_mask
    chain_mask = torch.from_numpy(chain_mask).to(dtype=torch.float32, device=device)
    chain_encoding = torch.from_numpy(chain_encoding).to(dtype=torch.long, device=device)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    omit_AA_mask = torch.from_numpy(omit_AA_mask).to(dtype=torch.float32, device=device)
    tied_beta = torch.from_numpy(tied_beta).to(dtype=torch.float32, device=device)
    pssm_coef = torch.from_numpy(pssm_coef).to(dtype=torch.float32, device=device)
    pssm_bias = torch.from_numpy(pssm_bias).to(dtype=torch.float32, device=device)
    pssm_log_odds_mask = torch.from_numpy(pssm_log_odds_mask).to(dtype=torch.float32, device=device)
    bias_by_res = torch.from_numpy(bias_by_res).to(dtype=torch.float32, device=device)
    # torch.from_numpy(omit_aas).to(dtype=torch.float32, device=device)

    return dict(X=X,
                S=S,
                chain_mask=chain_mask,
                chain_encoding=chain_encoding,
                residue_idx=residue_idx,
                mask=mask,
                chain_M_pos=residue_mask,
                omit_AA_mask=omit_AA_mask,
                pssm_coef=pssm_coef,
                pssm_bias=pssm_bias,
                pssm_log_odds_mask=pssm_log_odds_mask,
                tied_beta=tied_beta,
                bias_by_res=bias_by_res
                )


def score_sequences(S: torch.Tensor, log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Score ProteinMPNN sequences using Negative log likelihood probabilities

    Args:
        S: The sequence tensor
        log_probs: The logarithmic probabilities as found by a forward pass through ProteinMPNN
        mask: Any positions that are masked in the design task
    Returns:
        The loss calculated over the log probabilites compared to the sequence tensor
    """
    criterion = torch.nn.NLLLoss(reduction='none')
    # Measure log_probs loss with respect to the sequence. Make each sequence and log probs stacked along axis=0
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)),
        S.contiguous().view(-1)
    ).view(S.size())
    # Revert the shape to the original sequence shape
    return torch.sum(loss*mask, dim=-1) / torch.sum(mask, dim=-1)
