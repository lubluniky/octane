#!/usr/bin/env python3
"""
Python RL Baseline Benchmarks for comparison with RocketRL (Rust)

This script benchmarks common RL operations using NumPy and PyTorch
to compare against the Rust implementation.

Requirements:
    pip install numpy torch gymnasium stable-baselines3 tqdm
"""

import time
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Check for optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, skipping torch benchmarks")

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    print("Gymnasium not available, using custom env")


@dataclass
class BenchmarkResult:
    name: str
    mean_time_us: float
    std_time_us: float
    iterations: int
    throughput: float  # ops/sec


class TradingEnvPython:
    """Simple trading environment matching RocketRL's TradingEnv"""

    def __init__(self, num_timesteps: int = 10000, lookback: int = 20):
        self.num_timesteps = num_timesteps
        self.lookback = lookback
        self.num_features = 8
        self.obs_dim = lookback * self.num_features + 2

        # Generate synthetic data
        np.random.seed(42)
        self.prices = np.random.randn(num_timesteps, self.num_features).astype(np.float32)

        self.reset()

    def reset(self) -> np.ndarray:
        self.current_step = self.lookback
        self.balance = 10000.0
        self.position = 0.0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        start = max(0, self.current_step - self.lookback)
        obs = self.prices[start:self.current_step].flatten()
        # Pad if needed
        if len(obs) < self.lookback * self.num_features:
            obs = np.pad(obs, (self.lookback * self.num_features - len(obs), 0))
        return np.concatenate([obs, [self.position, self.balance / 10000.0 - 1.0]]).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action).flatten()
        self.position = float(np.clip(action[0], -1.0, 1.0))
        self.current_step += 1

        reward = float(np.random.randn() * 0.01)  # Simplified reward
        done = self.current_step >= self.num_timesteps - 1

        if done:
            self.reset()

        return self._get_obs(), reward, done, False, {}


class VectorizedEnvPython:
    """Vectorized environment wrapper"""

    def __init__(self, env_fn, num_envs: int):
        self.envs = [env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs

    def reset(self) -> np.ndarray:
        return np.stack([env.reset() for env in self.envs])

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
        results = [env.step(actions[i:i+1]) for i, env in enumerate(self.envs)]
        obs = np.stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        terminated = np.array([r[2] for r in results], dtype=np.float32)
        truncated = np.array([r[3] for r in results], dtype=np.float32)
        infos = [r[4] for r in results]

        # Auto-reset
        for i, (term, trunc) in enumerate(zip(terminated, truncated)):
            if term or trunc:
                obs[i] = self.envs[i].reset()

        return obs, rewards, terminated, truncated, infos


def benchmark_function(func, warmup: int = 10, iterations: int = 100) -> BenchmarkResult:
    """Benchmark a function and return statistics"""
    # Warmup
    for _ in range(warmup):
        func()

    # Actual timing
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to microseconds

    times = np.array(times)
    return BenchmarkResult(
        name="",
        mean_time_us=float(np.mean(times)),
        std_time_us=float(np.std(times)),
        iterations=iterations,
        throughput=1e6 / float(np.mean(times))  # ops/sec
    )


def run_benchmarks() -> Dict[str, BenchmarkResult]:
    results = {}

    print("=" * 60)
    print("Python RL Baseline Benchmarks")
    print("=" * 60)

    # 1. Single Environment Step
    print("\n[1/6] Single Environment Step...")
    env = TradingEnvPython()
    env.reset()
    action = np.zeros(1, dtype=np.float32)

    def single_step():
        obs, reward, done, trunc, info = env.step(action)
        if done:
            env.reset()
        return obs

    result = benchmark_function(single_step, iterations=1000)
    result.name = "single_env_step"
    results["single_env_step"] = result
    print(f"   Mean: {result.mean_time_us:.2f} μs (±{result.std_time_us:.2f})")

    # 2. Vectorized Environment Steps
    print("\n[2/6] Vectorized Environment Steps...")
    for num_envs in [1, 8, 32, 128, 512, 1024]:
        vec_env = VectorizedEnvPython(TradingEnvPython, num_envs)
        vec_env.reset()
        actions = np.zeros((num_envs, 1), dtype=np.float32)

        def vec_step():
            return vec_env.step(actions)

        iters = max(10, 1000 // num_envs)
        result = benchmark_function(vec_step, iterations=iters)
        result.name = f"vecenv_step_{num_envs}"
        results[f"vecenv_step_{num_envs}"] = result
        print(f"   {num_envs:4d} envs: {result.mean_time_us:.2f} μs, throughput: {num_envs * result.throughput:.0f} steps/sec")

    # 3. Environment Reset
    print("\n[3/6] Environment Reset...")
    env = TradingEnvPython()
    result = benchmark_function(env.reset, iterations=1000)
    result.name = "env_reset"
    results["env_reset"] = result
    print(f"   Mean: {result.mean_time_us:.2f} μs (±{result.std_time_us:.2f})")

    # 4. NumPy Tensor Operations
    print("\n[4/6] NumPy Matrix Operations...")
    for size in [64, 256, 1024]:
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)

        def matmul():
            return np.matmul(a, b)

        result = benchmark_function(matmul, iterations=100)
        result.name = f"numpy_matmul_{size}"
        results[f"numpy_matmul_{size}"] = result
        print(f"   {size}x{size}: {result.mean_time_us:.2f} μs")

    # 5. Softmax (NumPy)
    print("\n[5/6] Softmax Operations (NumPy)...")
    for batch_size in [32, 128, 512]:
        logits = np.random.randn(batch_size, 64).astype(np.float32)

        def softmax():
            exp_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)

        result = benchmark_function(softmax, iterations=1000)
        result.name = f"numpy_softmax_{batch_size}"
        results[f"numpy_softmax_{batch_size}"] = result
        print(f"   Batch {batch_size}: {result.mean_time_us:.2f} μs")

    # 6. GAE Computation
    print("\n[6/6] GAE Computation...")
    for buffer_size in [256, 1024, 2048]:
        num_envs = 128
        rewards = np.random.randn(buffer_size, num_envs).astype(np.float32)
        values = np.random.randn(buffer_size, num_envs).astype(np.float32)
        dones = np.zeros((buffer_size, num_envs), dtype=np.float32)
        gamma = 0.99
        gae_lambda = 0.95

        def compute_gae():
            advantages = np.zeros_like(rewards)
            last_gae = np.zeros(num_envs, dtype=np.float32)

            for t in reversed(range(buffer_size)):
                if t == buffer_size - 1:
                    next_value = np.zeros(num_envs, dtype=np.float32)
                else:
                    next_value = values[t + 1]

                delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
                last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
                advantages[t] = last_gae

            return advantages

        result = benchmark_function(compute_gae, iterations=100)
        result.name = f"gae_{buffer_size}"
        results[f"gae_{buffer_size}"] = result
        print(f"   Buffer {buffer_size}: {result.mean_time_us:.2f} μs")

    # PyTorch benchmarks if available
    if TORCH_AVAILABLE:
        print("\n" + "=" * 60)
        print("PyTorch Benchmarks (for GPU comparison)")
        print("=" * 60)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")

        for size in [64, 256, 1024]:
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)

            def torch_matmul():
                result = torch.matmul(a, b)
                if device.type != "cpu":
                    torch.mps.synchronize() if device.type == "mps" else torch.cuda.synchronize()
                return result

            result = benchmark_function(torch_matmul, iterations=100)
            result.name = f"torch_matmul_{size}_{device.type}"
            results[f"torch_matmul_{size}_{device.type}"] = result
            print(f"   {size}x{size} ({device.type}): {result.mean_time_us:.2f} μs")

    return results


def save_results(results: Dict[str, BenchmarkResult], filename: str = "python_benchmarks.json"):
    """Save benchmark results to JSON"""
    data = {
        name: {
            "mean_time_us": r.mean_time_us,
            "std_time_us": r.std_time_us,
            "iterations": r.iterations,
            "throughput": r.throughput
        }
        for name, r in results.items()
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    results = run_benchmarks()
    save_results(results)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nPython baseline benchmarks complete.")
    print("Run Rust benchmarks with: cargo bench")
    print("Then run visualization with: python benchmarks/visualize.py")
