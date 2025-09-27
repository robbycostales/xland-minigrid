import argparse
import os
import sys
import tempfile
import subprocess
import shutil
from dataclasses import asdict
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import wandb
 
import orbax.checkpoint as ocp  # type: ignore[attr-defined]
from flax.training.train_state import TrainState

"""Ensure project root is on sys.path for `from training...` imports when running from scripts/"""
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import xminigrid
from xminigrid.wrappers import DirectionObservationWrapper, GymAutoResetWrapper
from training.nn import ActorCriticRNN
from training.utils import rollout_host
def _write_video_ffmpeg(path: str, frames: list[np.ndarray], fps: int) -> None:
    if not frames or len(frames) <= 1:
        return
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg.")

    h, w, c = frames[0].shape
    assert c == 3, "Expected RGB frames (H, W, 3)"

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(int(max(1, fps))),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "0",
        path,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None
    try:
        for frame in frames:
            if frame.shape != (h, w, 3) or frame.dtype != np.uint8:
                frame = np.asarray(frame, dtype=np.uint8)
                if frame.shape != (h, w, 3):
                    raise ValueError("All frames must have the same HxW and 3 channels")
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg failed to write video")



def _load_latest_model_artifact(entity: str, project: str, run_id: str) -> str:
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    if len(artifacts) == 0:
        raise RuntimeError("No model artifacts found for the specified run.")
    # Prefer ones that start with our naming convention
    preferred = [a for a in artifacts if a.name.split("/")[-1].startswith("final-weights-")]
    art = preferred[-1] if len(preferred) > 0 else artifacts[-1]
    download_dir = art.download()
    return download_dir


def _restore_checkpoint(download_dir: str) -> dict[str, Any]:
    checkpointer = ocp.PyTreeCheckpointer()
    ckpt = checkpointer.restore(download_dir)
    if not isinstance(ckpt, dict) or "config" not in ckpt or "params" not in ckpt:
        raise RuntimeError("Checkpoint format invalid: expected dict with 'config' and 'params'.")
    return ckpt


def _build_env_and_model(cfg: dict[str, Any]) -> tuple[Any, Any, TrainState]:
    # env
    env, env_params = xminigrid.make(cfg["env_id"])
    env = GymAutoResetWrapper(env)
    env = DirectionObservationWrapper(env)

    # optional image observations wrapper
    if cfg.get("img_obs", False):
        from xminigrid.experimental.img_obs import RGBImgObservationWrapper

        env = RGBImgObservationWrapper(env)

    # single-task benchmark ruleset override
    if cfg.get("benchmark_id") is not None and cfg.get("ruleset_id") is not None and "XLand" in cfg["env_id"]:
        bench = xminigrid.load_benchmark(cfg["benchmark_id"])  # type: ignore[arg-type]
        env_params = env_params.replace(ruleset=bench.get_ruleset(cfg["ruleset_id"]))

    # network
    network = ActorCriticRNN(
        num_actions=env.num_actions(env_params),
        obs_emb_dim=cfg["obs_emb_dim"],
        action_emb_dim=cfg["action_emb_dim"],
        rnn_hidden_dim=cfg["rnn_hidden_dim"],
        rnn_num_layers=cfg["rnn_num_layers"],
        head_hidden_dim=cfg["head_hidden_dim"],
        img_obs=cfg.get("img_obs", False),
        dtype=jnp.bfloat16 if cfg.get("enable_bf16", False) else None,
    )

    # Create a minimal TrainState for apply_fn/params
    dummy_tx = optax.sgd(0.0)
    init_hstate = network.initialize_carry(batch_size=1)
    shapes: dict[str, Any] = env.observation_shape(env_params)  # type: ignore[assignment]
    init_obs = {
        "obs_img": jnp.zeros((1, 1, *shapes["img"])),
        "obs_dir": jnp.zeros((1, 1, shapes["direction"])),
        "prev_action": jnp.zeros((1, 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((1, 1)),
    }
    params = network.init(jax.random.key(0), init_obs, init_hstate)
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=dummy_tx)
    return env, env_params, train_state


def _load_params_into_state(state: TrainState, params: Any) -> TrainState:
    return state.replace(params=params)


def evaluate_single(
    *,
    env: Any,
    env_params: Any,
    train_state: TrainState,
    cfg: dict[str, Any],
    out_dir: str,
    episodes: int,
    fps: int,
    frame_skip: int,
    max_steps: int,
    save_trajectories: bool,
):
    videos_dir = os.path.join(out_dir, "videos")
    traj_dir = os.path.join(out_dir, "trajectories")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    outer_pbar = None
    try:
        from tqdm.auto import tqdm  # type: ignore

        outer_pbar = tqdm(total=episodes, desc="Episodes", dynamic_ncols=True)
    except Exception:
        outer_pbar = None

    for ep in range(episodes):
        rng = jax.random.key(cfg.get("eval_seed", 42) + ep)
        ep_log = rollout_host(
            env=env,
            env_params=env_params,
            train_state=train_state,
            rng=rng,
            rnn_num_layers=cfg["rnn_num_layers"],
            rnn_hidden_dim=cfg["rnn_hidden_dim"],
            enable_bf16=cfg.get("enable_bf16", False),
            record_frames=fps > 0,
            max_steps=max_steps if max_steps > 0 else None,
            progress=True,
            frame_stride=frame_skip if (fps > 0 and frame_skip and frame_skip > 1) else 1,
        )

        if outer_pbar is not None:
            try:
                outer_pbar.update(1)
                outer_pbar.set_postfix({"len": ep_log.length, "ret": f"{ep_log.episode_return:.2f}"})
            except Exception:
                pass

        if fps > 0 and ep_log.frames:
            frames_to_write = ep_log.frames
            if len(frames_to_write) <= 1:
                continue
            path = os.path.join(videos_dir, f"episode_{ep:04d}.mp4")
            _write_video_ffmpeg(path, frames_to_write, fps)
        if save_trajectories:
            np.savez(
                os.path.join(traj_dir, f"episode_{ep:04d}.npz"),
                actions=np.asarray(ep_log.actions),
                rewards=np.asarray(ep_log.rewards),
            )

    if outer_pbar is not None:
        try:
            outer_pbar.close()
        except Exception:
            pass


def evaluate_meta(
    *,
    env: Any,
    env_params: Any,
    train_state: TrainState,
    cfg: dict[str, Any],
    out_dir: str,
    num_envs: int,
    episodes_per_env: int,
    fps: int,
    frame_skip: int,
    max_steps: int,
    save_trajectories: bool,
):
    bench = xminigrid.load_benchmark(cfg["benchmark_id"])  # type: ignore[arg-type]
    videos_dir = os.path.join(out_dir, "videos")
    traj_dir = os.path.join(out_dir, "trajectories")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    ruleset_rng = jax.random.split(jax.random.key(cfg.get("eval_seed", 42) + 777), num=num_envs)
    eval_rulesets = cast(Any, jax.vmap(bench.sample_ruleset)(ruleset_rng))

    outer_pbar = None
    try:
        from tqdm.auto import tqdm  # type: ignore

        outer_pbar = tqdm(total=num_envs * episodes_per_env, desc="Meta Episodes", dynamic_ncols=True)
    except Exception:
        outer_pbar = None

    for env_idx in range(num_envs):
        ruleset_i = jtu.tree_map(lambda x: x[env_idx], eval_rulesets)
        env_params_i = env_params.replace(ruleset=ruleset_i)
        for ep in range(episodes_per_env):
            rng = jax.random.key(cfg.get("eval_seed", 42) + env_idx * 1000 + ep)
            ep_log = rollout_host(
                env=env,
                env_params=env_params_i,
                train_state=train_state,
                rng=rng,
                rnn_num_layers=cfg["rnn_num_layers"],
                rnn_hidden_dim=cfg["rnn_hidden_dim"],
                enable_bf16=cfg.get("enable_bf16", False),
                record_frames=fps > 0,
                max_steps=max_steps if max_steps > 0 else None,
                progress=True,
                frame_stride=frame_skip if (fps > 0 and frame_skip and frame_skip > 1) else 1,
            )
            if outer_pbar is not None:
                try:
                    outer_pbar.update(1)
                    outer_pbar.set_postfix({"env": env_idx, "ep": ep, "len": ep_log.length, "ret": f"{ep_log.episode_return:.2f}"})
                except Exception:
                    pass
            if fps > 0 and ep_log.frames:
                frames_to_write = ep_log.frames
                if len(frames_to_write) <= 1:
                    continue
                path = os.path.join(videos_dir, f"env{env_idx:03d}_ep{ep:03d}.mp4")
                _write_video_ffmpeg(path, frames_to_write, fps)
            if save_trajectories:
                np.savez(
                    os.path.join(traj_dir, f"env{env_idx:03d}_ep{ep:03d}.npz"),
                    actions=np.asarray(ep_log.actions),
                    rewards=np.asarray(ep_log.rewards),
                )

    if outer_pbar is not None:
        try:
            outer_pbar.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, default="shalab")
    parser.add_argument("--project", type=str, default="xland-minigrid")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="./eval_outputs")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=0, help="Cap steps per episode (0 = unlimited)")
    parser.add_argument("--frame-skip", type=int, default=1, help="Write every k-th frame to video (k >= 1). Effective FPS = fps/k")
    parser.add_argument("--save-trajectories", action="store_true")
    # meta-only
    parser.add_argument("--meta-num-envs", type=int, default=8)
    parser.add_argument("--meta-episodes-per-env", type=int, default=2)
    args = parser.parse_args()

    # 1) Download latest model artifact
    download_dir = _load_latest_model_artifact(args.entity, args.project, args.run_id)

    # 2) Restore checkpoint
    ckpt = _restore_checkpoint(download_dir)
    cfg = ckpt["config"]

    # 3) Build env and model, load params
    env, env_params, state = _build_env_and_model(cfg)
    state = _load_params_into_state(state, ckpt["params"])

    # 4) Evaluate
    os.makedirs(args.out_dir, exist_ok=True)
    is_meta = cfg.get("benchmark_id") is not None and "XLand" in cfg["env_id"]
    if is_meta:
        evaluate_meta(
            env=env,
            env_params=env_params,
            train_state=state,
            cfg=cfg,
            out_dir=args.out_dir,
            num_envs=args.meta_num_envs,
            episodes_per_env=args.meta_episodes_per_env,
            fps=args.fps,
            frame_skip=args.frame_skip,
            max_steps=args.max_steps,
            save_trajectories=args.save_trajectories,
        )
    else:
        evaluate_single(
            env=env,
            env_params=env_params,
            train_state=state,
            cfg=cfg,
            out_dir=args.out_dir,
            episodes=args.episodes,
            fps=args.fps,
            frame_skip=args.frame_skip,
            max_steps=args.max_steps,
            save_trajectories=args.save_trajectories,
        )

    print("Done. Videos written to:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()


