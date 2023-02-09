import haiku as hk
import jax
import jax.numpy as jnp

from symdesign.third_party.alphafold.alphafold.common import residue_constants
from symdesign.third_party.alphafold.alphafold.model.modules_multimer import AlphaFoldIteration


class AlphaFoldInitialGuess(hk.Module):
  """AlphaFold model with recycling.

  Jumper et al. (2021) Suppl. Alg. 2 "Inference"
  """

  def __init__(self, config, name='alphafold'):
    super().__init__(name=name)
    self.config = config
    self.global_config = config.global_config

  def __call__(
      self,
      batch,
      is_training,
      compute_loss=False,
      ensemble_representations=False,
      return_representations=False):
    """Run the AlphaFold model.

    Arguments:
      batch: Dictionary with inputs to the AlphaFold model.
      is_training: Whether the system is in training or inference mode.
      compute_loss: Whether to compute losses (requires extra features
        to be present in the batch and knowing the true structure).
      ensemble_representations: Whether to use ensembling of representations.
      return_representations: Whether to also return the intermediate
        representations.

    Returns:
      When compute_loss is True:
        a tuple of loss and output of AlphaFoldIteration.
      When compute_loss is False:
        just output of AlphaFoldIteration.

      The output of AlphaFoldIteration is a nested dictionary containing
      predictions from the various heads.
    """
    # SYMDESIGN - Attempt to extract a previous position passed as initialization
    prev_pos = batch.pop('prev_pos', None)

    impl = AlphaFoldIteration(self.config, self.global_config)
    batch_size, num_residues = batch['aatype'].shape

    def get_prev(ret):
      new_prev = {
          'prev_pos':
              ret['structure_module']['final_atom_positions'],
          'prev_msa_first_row': ret['representations']['msa_first_row'],
          'prev_pair': ret['representations']['pair'],
      }
      return jax.tree_map(jax.lax.stop_gradient, new_prev)

    def do_call(prev,
                recycle_idx,
                compute_loss=compute_loss):
      if self.config.resample_msa_in_recycling:
        num_ensemble = batch_size // (self.config.num_recycle + 1)
        def slice_recycle_idx(x):
          start = recycle_idx * num_ensemble
          size = num_ensemble
          return jax.lax.dynamic_slice_in_dim(x, start, size, axis=0)
        ensembled_batch = jax.tree_map(slice_recycle_idx, batch)
      else:
        num_ensemble = batch_size
        ensembled_batch = batch

      non_ensembled_batch = jax.tree_map(lambda x: x, prev)

      return impl(
          ensembled_batch=ensembled_batch,
          non_ensembled_batch=non_ensembled_batch,
          is_training=is_training,
          compute_loss=compute_loss,
          ensemble_representations=ensemble_representations)

    prev = {}
    emb_config = self.config.embeddings_and_evoformer
    if emb_config.recycle_pos:
      # SYMDESIGN
      if prev_pos:
        prev['prev_pos'] = prev_pos
      else:
        prev['prev_pos'] = jnp.zeros(
          [num_residues, residue_constants.atom_type_num, 3])
      # SYMDESIGN
    if emb_config.recycle_features:
      prev['prev_msa_first_row'] = jnp.zeros(
          [num_residues, emb_config.msa_channel])
      prev['prev_pair'] = jnp.zeros(
          [num_residues, num_residues, emb_config.pair_channel])

    if self.config.num_recycle:
      if 'num_iter_recycling' in batch:
        # Training time: num_iter_recycling is in batch.
        # The value for each ensemble batch is the same, so arbitrarily taking
        # 0-th.
        num_iter = batch['num_iter_recycling'][0]

        # Add insurance that we will not run more
        # recyclings than the model is configured to run.
        num_iter = jnp.minimum(num_iter, self.config.num_recycle)
      else:
        # Eval mode or tests: use the maximum number of iterations.
        num_iter = self.config.num_recycle

      body = lambda x: (x[0] + 1,  # pylint: disable=g-long-lambda
                        get_prev(do_call(x[1], recycle_idx=x[0],
                                         compute_loss=False)))
      if hk.running_init():
        # When initializing the Haiku module, run one iteration of the
        # while_loop to initialize the Haiku modules used in `body`.
        _, prev = body((0, prev))
      else:
        _, prev = hk.while_loop(
            lambda x: x[0] < num_iter,
            body,
            (0, prev))
    else:
      num_iter = 0

    ret = do_call(prev=prev, recycle_idx=num_iter)
    if compute_loss:
      ret = ret[0], [ret[1]]

    if not return_representations:
      del (ret[0] if compute_loss else ret)['representations']  # pytype: disable=unsupported-operands
    return ret
