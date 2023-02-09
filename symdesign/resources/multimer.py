import haiku as hk
import jax
import jax.numpy as jnp

from symdesign.third_party.alphafold.alphafold.common import residue_constants
from symdesign.third_party.alphafold.alphafold.model import prng
from symdesign.third_party.alphafold.alphafold.model import utils
from symdesign.third_party.alphafold.alphafold.model.modules_multimer import AlphaFoldIteration


class AlphaFoldInitialGuess(hk.Module):
  """AlphaFold-Multimer model with recycling.
  """

  def __init__(self, config, name='alphafold'):
    super().__init__(name=name)
    self.config = config
    self.global_config = config.global_config

  def __call__(
      self,
      batch,
      is_training,
      return_representations=False,
      safe_key=None):
    # SYMDESIGN - Attempt to extract a previous position passed as initialization
    prev_pos = batch.pop('prev_pos', None)
    # SYMDESIGN
    c = self.config
    impl = AlphaFoldIteration(c, self.global_config)

    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())
    elif isinstance(safe_key, jnp.ndarray):
      safe_key = prng.SafeKey(safe_key)

    assert isinstance(batch, dict)
    num_res = batch['aatype'].shape[0]

    def get_prev(ret):
      new_prev = {
          'prev_pos':
              ret['structure_module']['final_atom_positions'],
          'prev_msa_first_row': ret['representations']['msa_first_row'],
          'prev_pair': ret['representations']['pair'],
      }
      return jax.tree_map(jax.lax.stop_gradient, new_prev)

    def apply_network(prev, safe_key):
      recycled_batch = {**batch, **prev}
      return impl(
          batch=recycled_batch,
          is_training=is_training,
          safe_key=safe_key)

    prev = {}
    emb_config = self.config.embeddings_and_evoformer
    if emb_config.recycle_pos:
        # SYMDESIGN
        if prev_pos:
          prev['prev_pos'] = prev_pos
        else:
          prev['prev_pos'] = jnp.zeros(
            [num_res, residue_constants.atom_type_num, 3])
        # SYMDESIGN
    if emb_config.recycle_features:
      prev['prev_msa_first_row'] = jnp.zeros(
          [num_res, emb_config.msa_channel])
      prev['prev_pair'] = jnp.zeros(
          [num_res, num_res, emb_config.pair_channel])

    if self.config.num_recycle:
      if 'num_iter_recycling' in batch:
        # Training time: num_iter_recycling is in batch.
        # Value for each ensemble batch is the same, so arbitrarily taking 0-th.
        num_iter = batch['num_iter_recycling'][0]

        # Add insurance that even when ensembling, we will not run more
        # recyclings than the model is configured to run.
        num_iter = jnp.minimum(num_iter, c.num_recycle)
      else:
        # Eval mode or tests: use the maximum number of iterations.
        num_iter = c.num_recycle

      def distances(points):
        """Compute all pairwise distances for a set of points."""
        return jnp.sqrt(jnp.sum((points[:, None] - points[None, :])**2,
                                axis=-1))

      def recycle_body(x):
        i, _, prev, safe_key = x
        safe_key1, safe_key2 = safe_key.split() if c.resample_msa_in_recycling else safe_key.duplicate()  # pylint: disable=line-too-long
        ret = apply_network(prev=prev, safe_key=safe_key2)
        return i+1, prev, get_prev(ret), safe_key1

      def recycle_cond(x):
        i, prev, next_in, _ = x
        ca_idx = residue_constants.atom_order['CA']
        sq_diff = jnp.square(distances(prev['prev_pos'][:, ca_idx, :]) -
                             distances(next_in['prev_pos'][:, ca_idx, :]))
        mask = batch['seq_mask'][:, None] * batch['seq_mask'][None, :]
        sq_diff = utils.mask_mean(mask, sq_diff)
        # Early stopping criteria based on criteria used in
        # AF2Complex: https://www.nature.com/articles/s41467-022-29394-2
        diff = jnp.sqrt(sq_diff + 1e-8)  # avoid bad numerics giving negatives
        less_than_max_recycles = (i < num_iter)
        has_exceeded_tolerance = (
            (i == 0) | (diff > c.recycle_early_stop_tolerance))
        return less_than_max_recycles & has_exceeded_tolerance

      if hk.running_init():
        num_recycles, _, prev, safe_key = recycle_body(
            (0, prev, prev, safe_key))
      else:
        num_recycles, _, prev, safe_key = hk.while_loop(
            recycle_cond,
            recycle_body,
            (0, prev, prev, safe_key))
    else:
      # No recycling.
      num_recycles = 0

    # Run extra iteration.
    ret = apply_network(prev=prev, safe_key=safe_key)

    if not return_representations:
      del ret['representations']
    ret['num_recycles'] = num_recycles

    return ret
