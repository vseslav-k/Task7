import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter


# ----------------------------
# SharedAdam
# ----------------------------
class SharedAdam(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # Put Adam state tensors into shared memory
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # Use 0-d tensor for step (safer than shape [1])
                state["step"] = torch.zeros((), dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


class Transition:
    def __init__(self, obs, action, reward, resultingObs, done):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.resultingObs = resultingObs
        self.done = done


class QNet(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class AsyncAgent:
    def __init__(
        self,
        env: gym.Env,
        id: int,
        lock,
        init_exploration_rate: float,
        exploration_rate_decay: float,
        min_exploration_rate: float,
        online_nn: QNet,
        target_nn: QNet,
        optimizer: SharedAdam,
        future_reward_discount_factor: float = 0.95,
        q_target_update_rate: int = 200,
    ):
        self.id = id
        self.env = env
        self.future_reward_discount_factor = future_reward_discount_factor

        self.exploration_rate = init_exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.min_exploration_rate = min_exploration_rate

        self.q_target_update_rate = q_target_update_rate
        self.update_counter = 0

        self.lock = lock

        self.q_online_net = online_nn
        self.q_target_net = target_nn
        self.q_target_net.eval()

        self.device = next(self.q_online_net.parameters()).device

        self.optimizer = optimizer
        self.loss_fn = nn.SmoothL1Loss()

        self.training_error = []

        # Only worker 0 does the initial sync, under lock
        if self.id == 0:
            with self.lock:
                self.sync_online_and_target()

    def obs_to_tensor(self, obs):
        return torch.tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)

    def getAction(self, obs=None):
        if (obs is None) or (random.random() < self.exploration_rate):
            return self.env.action_space.sample()
        with torch.no_grad():
            obs_t = self.obs_to_tensor(obs)
            q_values = self.q_online_net(obs_t)
            return int(torch.argmax(q_values, dim=1).item())

    def Q_online(self, obs, action):
        obs_t = self.obs_to_tensor(obs)
        q_values = self.q_online_net(obs_t)
        return q_values[0, int(action)]

    def sync_online_and_target(self):
        self.q_target_net.load_state_dict(self.q_online_net.state_dict())

    def calc_target(self, transition):
        if transition.done:
            return torch.tensor(float(transition.reward), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_obs_t = self.obs_to_tensor(transition.resultingObs)
            q_next_all = self.q_target_net(next_obs_t)
            max_next_q = torch.max(q_next_all, dim=1).values[0]
            return float(transition.reward) + self.future_reward_discount_factor * max_next_q

    def calc_loss(self, transition):
        q_pred = self.Q_online(transition.obs, transition.action)
        y = self.calc_target(transition)
        return self.loss_fn(q_pred, y)

    def decay_exploration_rate(self):
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate - self.exploration_rate_decay,
        )

    def update_Q_online(self, transition):
        loss = self.calc_loss(transition)

        with self.lock:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_online_net.parameters(), max_norm=10.0)
            self.optimizer.step()

        if self.id == 0:
            self.training_error.append(float(loss.item()))
        return float(loss.item())

    def update(self, obs, action, reward, resultingObs, done):
        self.update_counter += 1
        loss_val = self.update_Q_online(Transition(obs, action, reward, resultingObs, done))

        if (self.update_counter % self.q_target_update_rate == 0) and (self.id == 0):
            with self.lock:
                self.sync_online_and_target()

        return loss_val


# ----------------------------
# Worker process
# ----------------------------
def worker_fn(
    wid: int,
    online_nn: QNet,
    target_nn: QNet,
    optimizer: SharedAdam,
    lock,
    global_step,
    max_global_steps: int,
    seed: int,
    init_eps: float,
    eps_decay_per_episode: float,
    min_eps: float,
    gamma: float,
    q_target_update_rate: int,
    max_ep_steps: int,
    log_dir: str,
):
    random.seed(seed + 1000 * wid)
    np.random.seed(seed + 1000 * wid)
    torch.manual_seed(seed + 1000 * wid)

    env = gym.make("CartPole-v1")

    agent = AsyncAgent(
        env=env,
        id=wid,
        lock=lock,
        init_exploration_rate=init_eps,
        exploration_rate_decay=eps_decay_per_episode,
        min_exploration_rate=min_eps,
        online_nn=online_nn,
        target_nn=target_nn,
        optimizer=optimizer,
        future_reward_discount_factor=gamma,
        q_target_update_rate=q_target_update_rate,
    )

    # Only worker 0 writes TensorBoard to avoid file contention
    writer = SummaryWriter(log_dir=log_dir) if wid == 0 else None

    ep = 0
    recent_returns = []  # worker0-only usage is fine

    while True:
        with global_step.get_lock():
            if global_step.value >= max_global_steps:
                break

        obs, _ = env.reset(seed=seed + wid + ep)
        ep_return = 0.0
        ep_steps = 0
        last_loss = None

        for _ in range(max_ep_steps):
            with global_step.get_lock():
                if global_step.value >= max_global_steps:
                    break
                global_step.value += 1
                gs = global_step.value

            action = agent.getAction(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            last_loss = agent.update(obs, action, reward, next_obs, done)

            ep_return += reward
            ep_steps += 1
            obs = next_obs

            if done:
                break

        agent.decay_exploration_rate()
        ep += 1

        if wid == 0:
            recent_returns.append(ep_return)
            if len(recent_returns) > 20:
                recent_returns.pop(0)
            avg20 = sum(recent_returns) / len(recent_returns)

            # TensorBoard logs (episode-level)
            writer.add_scalar("episode/return", ep_return, ep)
            writer.add_scalar("episode/return_avg20", avg20, ep)
            writer.add_scalar("episode/length", ep_steps, ep)
            writer.add_scalar("episode/epsilon", agent.exploration_rate, ep)

            # Step-level loss (log at episode end, using latest loss)
            if last_loss is not None:
                writer.add_scalar("train/loss", last_loss, gs)

            # Parameter norm for stability monitoring
            with torch.no_grad():
                total_norm_sq = 0.0
                for p in agent.q_online_net.parameters():
                    total_norm_sq += p.data.norm(2).item() ** 2
                param_l2 = total_norm_sq ** 0.5
            writer.add_scalar("model/q_online_param_l2", param_l2, ep)

            if ep % 10 == 0:
                print(
                    f"[worker0] global_step={gs} ep={ep} return={ep_return:.1f} "
                    f"avg20={avg20:.1f} steps={ep_steps} eps={agent.exploration_rate:.3f} "
                    f"loss={(last_loss if last_loss is not None else float('nan')):.4f}"
                )

    env.close()
    if writer is not None:
        writer.flush()
        writer.close()


# ----------------------------
# Main: spawn 8 workers
# ----------------------------
def main():
    mp.set_start_method("spawn", force=True)

    # Avoid each process using many intra-op threads (helps CPU scaling)
    torch.set_num_threads(1)

    env_tmp = gym.make("CartPole-v1")
    obs_dim = int(np.prod(env_tmp.observation_space.shape))
    action_dim = env_tmp.action_space.n
    env_tmp.close()

    # Create global shared models on CPU
    online_nn = QNet(obs_dim, action_dim)
    target_nn = QNet(obs_dim, action_dim)

    online_nn.share_memory()
    target_nn.share_memory()

    # Sync target once in parent
    target_nn.load_state_dict(online_nn.state_dict())
    target_nn.eval()

    optimizer = SharedAdam(online_nn.parameters(), lr=1e-3)

    lock = mp.Lock()
    global_step = mp.Value("i", 0)

    # Config
    num_workers = 8
    max_global_steps = 200_000
    max_ep_steps = 500
    seed = 42
    gamma = 0.99

    init_eps = 1.0
    min_eps = 0.05
    eps_decay_per_episode = (init_eps - min_eps) / 300.0

    q_target_update_rate = 1000

    log_dir = "runs/cartpole_async"

    procs = []
    for wid in range(num_workers):
        p = mp.Process(
            target=worker_fn,
            args=(
                wid,
                online_nn,
                target_nn,
                optimizer,
                lock,
                global_step,
                max_global_steps,
                seed,
                init_eps,
                eps_decay_per_episode,
                min_eps,
                gamma,
                q_target_update_rate,
                max_ep_steps,
                log_dir,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()