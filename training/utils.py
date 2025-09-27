# utilities for PPO training and evaluation
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import multiprocessing as mp

import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState

from xminigrid.environment import Environment, EnvParams
from xminigrid.rendering.rgb_render import render as rgb_render
from typing import Tuple


def _render_frame_from_args(args: Tuple[np.ndarray, np.ndarray, int, int, int]) -> np.ndarray:
    grid, agent_pos, agent_dir, view_size, tile_size = args
    from xminigrid.rendering.rgb_render import render_frame_fast
    return render_frame_fast(grid, agent_pos, agent_dir, view_size, tile_size=tile_size)


# Training stuff
class Transition(struct.PyTreeNode):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    # for obs
    obs: jax.Array
    dir: jax.Array
    # for rnn policy
    prev_action: jax.Array
    prev_reward: jax.Array


def calculate_gae(
    transitions: Transition,
    last_val: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    # single iteration for the loop
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        delta = transition.reward + gamma * next_value * (1 - transition.done) - transition.value
        gae = delta + gamma * gae_lambda * (1 - transition.done) * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        transitions,
        reverse=True,
    )
    # advantages and values (Q)
    return advantages, advantages + transitions.value


def ppo_update_networks(
    train_state: TrainState,
    transitions: Transition,
    init_hstate: jax.Array,
    advantages: jax.Array,
    targets: jax.Array,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
):
    # NORMALIZE ADVANTAGES
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def _loss_fn(params):
        # RERUN NETWORK
        dist, value, _ = train_state.apply_fn(
            params,
            {
                # [batch_size, seq_len, ...]
                "obs_img": transitions.obs,
                "obs_dir": transitions.dir,
                "prev_action": transitions.prev_action,
                "prev_reward": transitions.prev_reward,
            },
            init_hstate,
        )
        log_prob = dist.log_prob(transitions.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = transitions.value + (value - transitions.value).clip(-clip_eps, clip_eps)
        value_loss = jnp.square(value - targets)
        value_loss_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()

        # TODO: ablate this!
        # value_loss = jnp.square(value - targets).mean()

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - transitions.log_prob)
        actor_loss1 = advantages * ratio
        actor_loss2 = advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
        entropy = dist.entropy().mean()

        total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy
        return total_loss, (value_loss, actor_loss, entropy)

    (loss, (vloss, aloss, entropy)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
    (loss, vloss, aloss, entropy, grads) = jax.lax.pmean((loss, vloss, aloss, entropy, grads), axis_name="devices")
    train_state = train_state.apply_gradients(grads=grads)
    update_info = {
        "total_loss": loss,
        "value_loss": vloss,
        "actor_loss": aloss,
        "entropy": entropy,
    }
    return train_state, update_info


# for evaluation (evaluate for N consecutive episodes, sum rewards)
# N=1 single task, N>1 for meta-RL
class RolloutStats(struct.PyTreeNode):
    reward: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0.0))
    length: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0))
    episodes: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0))


def rollout(
    rng: jax.Array,
    env: Environment,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: jax.Array,
    num_consecutive_episodes: int = 1,
) -> RolloutStats:
    def _cond_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, hstate = carry
        return jnp.less(stats.episodes, num_consecutive_episodes)

    def _body_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, hstate = carry

        rng, _rng = jax.random.split(rng)
        dist, _, hstate = train_state.apply_fn(
            train_state.params,
            {
                "obs_img": timestep.observation["img"][None, None, ...],
                "obs_dir": timestep.observation["direction"][None, None, ...],
                "prev_action": prev_action[None, None, ...],
                "prev_reward": prev_reward[None, None, ...],
            },
            hstate,
        )
        action = dist.sample(seed=_rng).squeeze()
        timestep = env.step(env_params, timestep, action)

        stats = stats.replace(
            reward=stats.reward + timestep.reward,
            length=stats.length + 1,
            episodes=stats.episodes + timestep.last(),
        )
        carry = (rng, stats, timestep, action, timestep.reward, hstate)
        return carry

    timestep = env.reset(env_params, rng)
    prev_action = jnp.asarray(0)
    prev_reward = jnp.asarray(0)
    init_carry = (rng, RolloutStats(), timestep, prev_action, prev_reward, init_hstate)

    final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_val=init_carry)
    return final_carry[1]


# Host-side rollout for logging trajectories and videos (not JIT/PMAP traced)
@dataclass
class EpisodeLog:
    episode_return: float
    length: int
    actions: list[int]
    rewards: list[float]
    frames: Optional[list[Any]]


def rollout_host(
    *,
    env: Environment,
    env_params: EnvParams,
    train_state: TrainState,
    rng: jax.Array,
    rnn_num_layers: int,
    rnn_hidden_dim: int,
    enable_bf16: bool = False,
    record_frames: bool = False,
    max_steps: Optional[int] = None,
    progress: bool = False,
    frame_stride: int = 1,
) -> EpisodeLog:
    # Run a fast JAX-compiled rollout to generate actions/rewards, then (optionally) replay only for rendering.
    timestep0 = env.reset(env_params, rng)
    prev_action0 = jnp.asarray(0)
    prev_reward0 = jnp.asarray(0.0)
    hstate0 = jnp.zeros(
        (1, rnn_num_layers, rnn_hidden_dim),
        dtype=jnp.bfloat16 if enable_bf16 else None,
    )

    horizon = max_steps if max_steps is not None else int(env_params.max_steps) if env_params.max_steps is not None else 1024

    def _scan_body(carry, _):
        step_rng, timestep, prev_action, prev_reward, hstate, done = carry
        step_rng, _rng = jax.random.split(step_rng)

        def _do_step(args):
            _step_rng, _timestep, _prev_action, _prev_reward, _hstate = args
            dist, _, _hstate = train_state.apply_fn(
                train_state.params,
                {
                    "obs_img": _timestep.observation["img"][None, None, ...],
                    "obs_dir": _timestep.observation["direction"][None, None, ...],
                    "prev_action": _prev_action[None, None, ...],
                    "prev_reward": _prev_reward[None, None, ...],
                },
                _hstate,
            )
            action = dist.sample(seed=_rng).squeeze()
            _timestep_next = env.step(env_params, _timestep, action)
            reward = _timestep_next.reward
            return _step_rng, _timestep_next, action, reward, _hstate

        step_rng2, timestep2, action2, reward2, hstate2 = jax.lax.cond(
            done,
            lambda _: (step_rng, timestep, prev_action, prev_reward, hstate),
            lambda _: _do_step((step_rng, timestep, prev_action, prev_reward, hstate)),
            operand=None,
        )
        done2 = jnp.logical_or(done, timestep2.last())
        carry2 = (step_rng2, timestep2, action2, reward2, hstate2, done2)
        # Also return minimal state needed for rendering
        out = (action2, reward2, done2, timestep2.state.grid, timestep2.state.agent)
        return carry2, out

    @jax.jit
    def _run_scan(step_rng, timestep, prev_action, prev_reward, hstate):
        init = (step_rng, timestep, prev_action, prev_reward, hstate, jnp.asarray(False))
        (step_rng_f, timestep_f, prev_action_f, prev_reward_f, hstate_f, done_f), outs = jax.lax.scan(
            _scan_body, init, xs=None, length=horizon
        )
        return outs

    actions_arr, rewards_arr, dones_arr, grids_arr, agents_arr = _run_scan(rng, timestep0, prev_action0, prev_reward0, hstate0)

    # Determine actual episode length
    any_done = jnp.any(dones_arr)
    first_done_idx = jnp.argmax(dones_arr)
    length_jax = jnp.where(any_done, first_done_idx + 1, jnp.asarray(horizon))
    length = int(length_jax.item())

    # Move to host
    actions_np, rewards_np, grids_np, agents_np = jax.device_get((actions_arr, rewards_arr, grids_arr, agents_arr))
    actions_np = actions_np[:length]
    rewards_np = rewards_np[:length]
    grids_np = grids_np[:length]
    agent_pos_np = np.asarray(agents_np.position)[:length]
    agent_dir_np = np.asarray(agents_np.direction)[:length]

    actions_list = [int(x) for x in np.asarray(actions_np).tolist()]
    rewards_list = [float(x) for x in np.asarray(rewards_np).tolist()]
    ep_return = float(np.asarray(rewards_np).sum())

    frames: list[Any] | None = None
    if record_frames:
        frames = []
        import os
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

        view_size_int = int(env_params.view_size)
        tile_size = 24
        subdivs = 2
        highlight = False

        # Build args list in order
        frame_args: list[Tuple[np.ndarray, np.ndarray, int, int, int]] = []
        # initial frame
        init_agent_pos = np.asarray(jax.device_get(timestep0.state.agent.position))
        init_agent_dir = int(jax.device_get(timestep0.state.agent.direction))
        frame_args.append((np.asarray(jax.device_get(timestep0.state.grid)), init_agent_pos, init_agent_dir, view_size_int, tile_size))
        # subsequent frames at stride
        for idx in range(length):
            if frame_stride <= 1 or ((idx + 1) % frame_stride == 0):
                pos_i = np.asarray(agent_pos_np[idx])
                dir_i = int(agent_dir_np[idx])
                grid_i = np.asarray(grids_np[idx])
                frame_args.append((grid_i, pos_i, dir_i, view_size_int, tile_size))

        max_workers = max(1, (os.cpu_count() or 4))
        # Prefer threads to avoid re-importing JAX in child processes; fall back to spawned processes, then serial
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                frames = list(executor.map(_render_frame_from_args, frame_args))
        except Exception:
            try:
                with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
                    frames = list(executor.map(_render_frame_from_args, frame_args))
            except Exception:
                frames = [_render_frame_from_args(a) for a in frame_args]

    return EpisodeLog(
        episode_return=ep_return,
        length=int(length),
        actions=actions_list,
        rewards=rewards_list,
        frames=frames,
    )
