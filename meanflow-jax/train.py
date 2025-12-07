"""ImageNet DiT example.

This script trains a DiT on the ImageNet dataset.
The data is loaded using pytorch dataset.
"""
from copy import deepcopy
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import ml_collections
import optax
from clu import metric_writers
from flax import jax_utils
from flax.training import common_utils, train_state
from jax import lax, random
from optax._src.alias import *

import utils.input_pipeline as input_pipeline
from meanflow import MeanFlow, generate
from utils import fid_util, sample_util
from utils.ckpt_util import restore_checkpoint, save_checkpoint
from utils.ema_util import ema_schedules, update_ema
from utils.info_util import print_params
from utils.logging_util import Timer, log_for_0
from utils.vae_util import LatentManager
from utils.vis_util import make_grid_visualization
from utils.vis_3d_util import make_3d_grid_visualization

#######################################################
# Initialize
#######################################################

def initialized(key, image_size, model, in_channels=4):
  input_shape = (1, image_size, image_size, in_channels)
  x = jnp.ones(input_shape)
  t = jnp.ones((1,), dtype=int)
  y = jnp.ones((1,), dtype=int)

  @jax.jit
  def init(*args):
    return model.init(*args)

  log_for_0('Initializing params...')
  variables = init({'params': key}, x, t, y)
  log_for_0('Initializing params done.')

  param_count = sum(x.size for x in jax.tree_leaves(variables['params']))
  log_for_0("Total trainable parameters: " + str(param_count))
  return variables, variables['params']


class TrainState(train_state.TrainState):
  ema_params: Any


def create_train_state(
    rng, config: ml_collections.ConfigDict, model, image_size, lr_value, steps_per_epoch=None
):
  """
  Create initial training state.
  ---
  apply_fn: output a dict, with key 'loss', 'mse'
  """

  rng, rng_init = random.split(rng)

  in_channels = config.model.get('in_channels', 4)
  _, params = initialized(rng_init, image_size, model, in_channels=in_channels)
  ema_params = deepcopy(params)
  ema_params = update_ema(ema_params, params, 0)
  print_params(params['net'])

  # Create learning rate schedule if enabled
  if config.training.get('use_lr_schedule', False) and steps_per_epoch is not None:
      log_for_0('Creating learning rate schedule...')
      warmup_steps = config.training.get('warmup_steps', 1000)
      total_steps = config.training.num_epochs * steps_per_epoch
      decay_steps = total_steps - warmup_steps
      min_lr = config.training.get('min_lr', 1e-5)

      # Warmup phase: linear increase from 0 to peak LR
      warmup_fn = optax.linear_schedule(
          init_value=0.0,
          end_value=config.training.learning_rate,
          transition_steps=warmup_steps
      )

      # Decay phase: cosine decay from peak LR to min LR
      decay_fn = optax.cosine_decay_schedule(
          init_value=config.training.learning_rate,
          decay_steps=decay_steps,
          alpha=min_lr / config.training.learning_rate
      )

      # Combine schedules
      lr_schedule = optax.join_schedules(
          schedules=[warmup_fn, decay_fn],
          boundaries=[warmup_steps]
      )

      log_for_0(f'  Warmup steps: {warmup_steps} ({warmup_steps/steps_per_epoch:.1f} epochs)')
      log_for_0(f'  Total steps: {total_steps}')
      log_for_0(f'  Peak LR: {config.training.learning_rate:.2e}')
      log_for_0(f'  Min LR: {min_lr:.2e}')

      tx = optax.adamw(
          learning_rate=lr_schedule,
          weight_decay=0,
          b2=config.training.adam_b2,
      )
  else:
      # Use fixed learning rate
      tx = optax.adamw(
          learning_rate=lr_value,
          weight_decay=0,
          b2=config.training.adam_b2,
      )

  state = TrainState.create(
      apply_fn=partial(model.apply, method=model.forward),
      params=params,
      ema_params=ema_params,
      tx=tx,
  )
  return state

#######################################################
# Train Step
#######################################################

def compute_metrics(dict_losses):
  metrics = dict_losses.copy()
  metrics = lax.all_gather(metrics, axis_name='batch')
  metrics = jax.tree_map(lambda x: x.flatten(), metrics)  # (batch_size,)
  return metrics


def train_step_with_vae(state, batch, rng_init, config, lr, ema_fn, latent_mnger, is_3d_data=False):
  """
  Perform a single training step.
  """
  rng_step = random.fold_in(rng_init, state.step)
  rng_base = random.fold_in(rng_step, lax.axis_index(axis_name='batch'))

  cached = batch['image'] # [B, H, W, C]
  rng_base, rng_vae = random.split(rng_base)

  if is_3d_data:
    # For 3D data: channels are duplicated, just take first half
    # Shape: (B, H, W, 32) -> (B, H, W, 16)
    images = cached[..., :cached.shape[-1]//2]
  else:
    # For ImageNet latents: sample from distribution (mean + std * noise)
    images = latent_mnger.cached_encode(cached, rng_vae, deterministic=False)

  labels = batch['label']

  def loss_fn(params):
    """loss function used for training."""
    variables = {
        "params": params,
    }
    outputs = state.apply_fn(
      variables,
      imgs=images,
      labels=labels,
      rngs=dict(gen=rng_base,),
    )
    loss, dict_losses = outputs
    return loss, (dict_losses,)
  
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  aux, grads = grad_fn(state.params)
  grads = lax.pmean(grads, axis_name='batch')

  dict_losses, = aux[1]
  metrics = compute_metrics(dict_losses)
  metrics["lr"] = lr

  new_state = state.apply_gradients(
    grads=grads,
  )

  ema_value = ema_fn(state.step)
  new_ema = update_ema(new_state.ema_params, new_state.params, ema_value)
  new_state = new_state.replace(ema_params=new_ema)

  return new_state, metrics

#######################################################
# Sampling and Metrics
#######################################################

def sample_step(variable, sample_idx, model, rng_init, device_batch_size, config):
  """
  sample_idx: each random sampled image corrresponds to a seed
  """
  rng_sample = random.fold_in(rng_init, sample_idx)  # fold in sample_idx
  images = generate(variable, model, rng_sample, n_sample=device_batch_size, config=config)
  images = images.transpose(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
  return images


def run_p_sample_step(p_sample_step, state, sample_idx, latent_manager, ema=True, skip_vae_decode=False):
  variable = {"params": state.ema_params if ema else state.params}
  latent = p_sample_step(variable, sample_idx=sample_idx)
  latent = latent.reshape(-1, *latent.shape[2:])

  if skip_vae_decode:
    # For 3D data: latent IS the actual data, no VAE decoding needed
    samples = latent
    # Transpose from (B, C, H, W) to (B, H, W, C)
    samples = samples.transpose(0, 2, 3, 1)
  else:
    # For ImageNet: decode VAE latents to images
    samples = latent_manager.decode(latent)
    assert not jnp.any(jnp.isnan(samples)), f"There is nan in decoded samples! Latent range: {latent.min()}, {latent.max()}. nan in latent: {jnp.any(jnp.isnan(latent))}"
    samples = samples.transpose(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)

  samples = 127.5 * samples + 128.0
  samples = jnp.clip(samples, 0, 255).astype(jnp.uint8)

  jax.random.normal(random.key(0), ()).block_until_ready() # dist sync
  return samples


def get_fid_evaluator(workdir, config, writer, p_sample_step, latent_manager, is_3d_data=False):
  # Use smaller batch size for local GPU (50 instead of default 200)
  inception_batch_size = 50
  inception_net = fid_util.build_jax_inception(batch_size=inception_batch_size)
  stats_ref = fid_util.get_reference(config.fid.cache_ref, inception_net)
  run_p_sample_step_inner = partial(run_p_sample_step, latent_manager=latent_manager, skip_vae_decode=is_3d_data)
  
  def evaluator(state, epoch):
    log_for_0('Eval fid at epoch: {}'.format(epoch))

    samples_all = sample_util.generate_fid_samples(
          state, workdir, config, p_sample_step, run_p_sample_step_inner
    )
    mu, sigma = fid_util.compute_stats(samples_all, inception_net, batch_size=inception_batch_size)
    fid_score = fid_util.compute_fid(mu, stats_ref["mu"], sigma, stats_ref["sigma"])
    log_for_0(f'FID w/ EMA at {samples_all.shape[0]} samples: {fid_score}')
    
    writer.write_scalars(epoch+1, {'FID_ema': fid_score})
    writer.flush()
  return evaluator

#######################################################
# Main
#######################################################

def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
) -> TrainState:
  ########### Initialize ###########
  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0
  )

  rng = random.key(config.training.seed)
  image_size = config.dataset.image_size

  log_for_0('config.training.batch_size: {}'.format(config.training.batch_size))

  if config.training.batch_size % jax.process_count() > 0:
    raise ValueError('Batch size must be divisible by the number of processes')
  local_batch_size = config.training.batch_size // jax.process_count()
  log_for_0('local_batch_size: {}'.format(local_batch_size))
  log_for_0('jax.local_device_count: {}'.format(jax.local_device_count()))

  ########### Create DataLoaders ###########
  if local_batch_size % jax.local_device_count() > 0:
    raise ValueError('Local batch size must be divisible by the number of local devices')

  train_loader, steps_per_epoch = input_pipeline.create_split(
    config.dataset,
    local_batch_size,
    split='train',
  )
  log_for_0('Steps per Epoch: {}'.format(steps_per_epoch))

  ########### Create Model ###########
  model_config = config.model.to_dict()
  model_str = model_config.pop('cls')

  log_for_0(f'Model config: {model_config}')
  log_for_0(f'Model class: {model_str}')

  model = MeanFlow(
    model_str=model_str,
    model_config=model_config,
    **config.sampling,
    **config.method,
  )

  ########### Create Train State ###########
  base_lr = config.training.learning_rate
  state = create_train_state(rng, config, model, image_size, lr_value=base_lr, steps_per_epoch=steps_per_epoch)
  if config.load_from is not None:
    state = restore_checkpoint(state, config.load_from)
  
  step_offset = int(state.step)
  epoch_offset = 0 if config.eval_only else (step_offset // steps_per_epoch)
  
  state = jax_utils.replicate(state)
  ema_fn = ema_schedules(config)

  # Detect if we're using 3D data (no VAE decoding needed)
  is_3d_data = '3d' in config.dataset.name.lower() or config.model.get('in_channels', 4) > 4
  log_for_0(f'3D data mode: {is_3d_data} (skip VAE decode)')

  latent_manager = LatentManager(config.dataset.vae, config.fid.device_batch_size, image_size)

  p_train_step = jax.pmap(
      partial(
        train_step_with_vae,
        rng_init=rng,
        config=config,
        lr=base_lr,
        ema_fn=ema_fn,
        latent_mnger=latent_manager,
        is_3d_data=is_3d_data,
      ),
      axis_name='batch',
  )
  train_metrics = []
  log_for_0('Initial compilation, this might take some minutes...')

  vis_sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(jax.local_device_count())

  ########### Sampling ###########
  p_sample_step = jax.pmap(
    partial(sample_step, 
            model=model, 
            rng_init=random.PRNGKey(config.sampling.seed), 
            device_batch_size=config.fid.device_batch_size, 
            config=config,
    ),
    axis_name='batch'
  )

  if config.fid.on_training:
    fid_evaluator = get_fid_evaluator(workdir, config, writer, p_sample_step, latent_manager, is_3d_data)

  if config.eval_only:
    fid_evaluator(state, epoch_offset)
    return state

  ########### Early Stopping Setup ###########
  best_metric = float('inf')
  patience_counter = 0
  early_stopping_patience = config.get('evaluation', {}).get('early_stopping_patience', 100)
  early_stopping_metric = config.get('evaluation', {}).get('early_stopping_metric', 'chamfer_distance')
  early_stopping_min_delta = config.get('evaluation', {}).get('early_stopping_min_delta', 0.01)
  log_for_0(f'Early stopping: patience={early_stopping_patience}, metric={early_stopping_metric}, min_delta={early_stopping_min_delta}')

  ########### Training Loop ###########
  for epoch in range(epoch_offset, config.training.num_epochs):

    if jax.process_count() > 1:
      train_loader.sampler.set_epoch(epoch)
    log_for_0('epoch {}...'.format(epoch))

    ########### Sampling ###########
    if (epoch+1) % config.training.sample_per_epoch == 0 and config.training.get('sample_on_training', True):
      log_for_0(f'Samples at epoch {epoch}...')
      vis_sample = run_p_sample_step(p_sample_step, state, vis_sample_idx, latent_manager, skip_vae_decode=is_3d_data)
      vis_sample = make_3d_grid_visualization(vis_sample, grid=4) if is_3d_data else make_grid_visualization(vis_sample, grid=4)

      writer.write_images(epoch+1, {'vis_sample': vis_sample})
      writer.flush()

    ########### Train ###########
    timer = Timer()
    log_for_0('epoch {}...'.format(epoch))
    timer.reset()
    for n_batch, batch in enumerate(train_loader):
      step = epoch * steps_per_epoch + n_batch

      batch = input_pipeline.prepare_batch_data(batch) # the batch contains latent, both mean and var.
      state, metrics = p_train_step(state, batch)
      
      if epoch == epoch_offset and n_batch == 0:
        log_for_0('Initial compilation completed. Reset timer.')
        compilation_time = timer.elapse_with_reset()
        log_for_0('p_train_step compiled in {:.2f}s'.format(compilation_time))

    ########### Metrics ###########
      train_metrics.append(metrics)
      if (step+1) % config.training.log_per_step == 0:
        train_metrics = common_utils.get_metrics(train_metrics)
        summary = jax.tree_util.tree_map(lambda x: float(x.mean()), train_metrics)
        summary['steps_per_second'] = config.training.log_per_step / timer.elapse_with_reset() 
        summary["ep"] = epoch
        writer.write_scalars(step + 1, summary)

        log_for_0(
          'train epoch: %d, step: %d, loss: %.6f, steps/sec: %.2f',
          epoch, step, summary['loss'], summary['steps_per_second'],
        )
        train_metrics = []

    ########### Save Checkpoint ###########
    if (
      (epoch+1) % config.training.checkpoint_per_epoch == 0
      or (epoch+1) == config.training.num_epochs
    ):
      save_checkpoint(state, workdir)

    ########### 3D Evaluation Metrics ###########
    if (epoch+1) % config.training.get('eval_per_epoch', 25) == 0 and config.training.get('eval_on_training', False) and is_3d_data:
      log_for_0(f'Evaluation at epoch {epoch+1}...')

      # Import evaluation utilities
      from utils.evaluation_3d import chamfer_distance_batch, compute_voxel_iou
      import numpy as np

      # Generate samples for evaluation
      eval_batch_size = config.training.get('eval_batch_size', 100)
      num_devices = jax.local_device_count()
      device_batch_size = config.fid.device_batch_size
      samples_per_iter = num_devices * device_batch_size
      num_iters = (eval_batch_size + samples_per_iter - 1) // samples_per_iter

      eval_samples = []
      for i in range(num_iters):
        sample_idx = jnp.arange(num_devices) + i * num_devices
        variable = {"params": state.ema_params}
        latent = p_sample_step(variable, sample_idx=sample_idx)
        latent = latent.reshape(-1, *latent.shape[2:])
        # Transpose to (B, H, W, C)
        samples = latent.transpose(0, 2, 3, 1)
        eval_samples.append(np.array(samples))

      eval_samples = np.concatenate(eval_samples, axis=0)[:eval_batch_size]
      log_for_0(f'  Generated {len(eval_samples)} evaluation samples')

      # Load real samples from validation set
      try:
        val_loader, _ = input_pipeline.create_split(
          config.dataset,
          local_batch_size,
          split='val',
        )
        real_samples = []
        for batch_idx, batch in enumerate(val_loader):
          batch = input_pipeline.prepare_batch_data(batch)
          real_batch = batch['image']
          if is_3d_data:
            # For 3D: take first half of channels (remove duplicate)
            real_batch = real_batch[..., :real_batch.shape[-1]//2]
          # Flatten device dimension: (devices, batch, H, W, C) -> (devices*batch, H, W, C)
          real_batch = real_batch.reshape(-1, *real_batch.shape[2:])
          real_samples.append(np.array(real_batch))
          if len(real_samples) * real_batch.shape[0] >= eval_batch_size:
            break
        
        if len(real_samples) == 0:
          log_for_0(f'  Warning: No validation samples loaded')
          writer.flush()
          continue
        
        real_samples = np.concatenate(real_samples, axis=0)[:eval_batch_size]
        log_for_0(f'  Loaded {len(real_samples)} real samples')

        # Compute Chamfer Distance
        if config.get('evaluation', {}).get('chamfer_enabled', True):
          threshold = config.get('evaluation', {}).get('chamfer_threshold', -0.5)
          cd_mean, cd_std = chamfer_distance_batch(eval_samples, real_samples, threshold=threshold)
          log_for_0(f'  Chamfer Distance: {cd_mean:.4f} ± {cd_std:.4f}')
          writer.write_scalars(epoch+1, {
            'chamfer_distance_mean': cd_mean,
            'chamfer_distance_std': cd_std
          })

        # Compute IoU
        if config.get('evaluation', {}).get('iou_enabled', True):
          iou_threshold = config.get('evaluation', {}).get('iou_threshold', 0.0)
          iou_mean, iou_std = compute_voxel_iou(eval_samples, real_samples, threshold=iou_threshold)
          log_for_0(f'  Voxel IoU: {iou_mean:.4f} ± {iou_std:.4f}')
          writer.write_scalars(epoch+1, {
            'voxel_iou_mean': iou_mean,
            'voxel_iou_std': iou_std
          })

        # Early stopping check
        if config.get('evaluation', {}).get('early_stopping_patience', 0) > 0:
          current_metric = cd_mean if early_stopping_metric == 'chamfer_distance' else iou_mean
          # ... rest of early stopping logic

      except Exception as e:
        import traceback
        log_for_0(f'  Warning: Evaluation failed: {e}')
        log_for_0(f'  Traceback: {traceback.format_exc()}')

      writer.flush()

    ########### FID ###########
    if (
        (epoch+1) % config.training.fid_per_epoch == 0
        or (epoch+1) == config.training.num_epochs
      ) and config.fid.on_training:
      fid_evaluator(state, epoch)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.key(0), ()).block_until_ready()
  return state
