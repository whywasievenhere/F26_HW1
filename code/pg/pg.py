#! python3

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np  # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PolicyGradient(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        lr_actor=1e-3,
        lr_critic=1e-3,
        mode="REINFORCE",
        n=0,
        gamma=0.99,
        device="cpu",
    ):
        super(PolicyGradient, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.mode = mode
        self.n = n
        self.gamma = gamma

        self.device = device

        hidden_layer_size = 256

        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, action_size),
            # BEGIN STUDENT SOLUTION
            nn.LogSoftmax(dim=-1),
            # END STUDENT SOLUTION
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            nn.Linear(hidden_layer_size, 1)
            # END STUDENT SOLUTION
        )

        # initialize networks, optimizers, move networks to device
        # BEGIN STUDENT SOLUTION
        self.to(self.device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        # END STUDENT SOLUTION

    def forward(self, state):
        return (self.actor(state), self.critic(state))

    def get_action(self, state, stochastic):
        # if stochastic, sample using the action probabilities, else get the argmax
        # BEGIN STUDENT SOLUTION
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                np.asarray(state, dtype=np.float32), device=self.device
            ).unsqueeze(0)
            log_probs = self.actor(state_tensor)
            if stochastic:
                # Categorical re-normalizes with log_softmax, which is
                # idempotent on the already-log-softmaxed actor output.
                action = torch.distributions.Categorical(logits=log_probs).sample()
            else:
                action = torch.argmax(log_probs, dim=-1)
        return int(action.item())
        # END STUDENT SOLUTION

    def __discounted_reverse_cumsum(self, rewards_tensor):
 
        T = rewards_tensor.shape[0]
        steps = torch.arange(T, dtype=torch.float64, device=rewards_tensor.device)
        disc = self.gamma**steps
        discounted = disc * rewards_tensor.to(torch.float64)

        fwd_cumsum = torch.cumsum(discounted, dim=0)
        total_sum = fwd_cumsum[-1]
        C = total_sum - fwd_cumsum + discounted
        
        return C, disc

    def calculate_n_step_bootstrap(self, rewards_tensor, values):
        # calculate n step bootstrap
        # BEGIN STUDENT SOLUTION
        T = rewards_tensor.shape[0]
        device = rewards_tensor.device
        n = self.n if self.n and self.n > 0 else T

        n = min(n, T)
        C, disc = self.__discounted_reverse_cumsum(rewards_tensor)

        values_flat = values.detach().reshape(-1).to(torch.float64)
        pad = torch.zeros(n, dtype=torch.float64, device=device)
        C_padded = torch.cat([C, pad])
        values_padded = torch.cat([values_flat, pad])
        idx = torch.arange(T, device=device) + n
        reward_window = (C - C_padded[idx]) / disc
        bootstrap = (self.gamma**n) * values_padded[idx]
        return (reward_window + bootstrap).to(rewards_tensor.dtype)
        # END STUDENT SOLUTION

    def train(self, states=None, actions=None, rewards=None):
        if actions is None and rewards is None and (
            states is None or isinstance(states, bool)
        ):
            return super().train(True if states is None else states)

        # train the agent using states, actions, and rewards
        # BEGIN STUDENT SOLUTION

        states_tensor = torch.as_tensor(np.asarray(states, dtype=np.float32), device=self.device)
        actions_tensor = torch.as_tensor(np.asarray(actions, dtype=np.int64), device=self.device)
        rewards_tensor = torch.as_tensor(np.asarray(rewards, dtype=np.float32), device=self.device)

        log_probs = self.actor(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        if self.mode == "A2C":
            values = self.critic(states_tensor).squeeze(-1)
            returns = self.calculate_n_step_bootstrap(rewards_tensor, values)
            advantages = returns - values.detach()

        actor_loss = -(advantages.detach() * log_probs).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        if values is not None:
            critic_loss = F.mse_loss(values, returns.detach())
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

        # END STUDENT SOLUTION

    def run(self, env, max_steps, num_episodes, train):
        total_rewards = []

        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        for _ in range(num_episodes):
            # No per-episode seed: the env is seeded once per trial, so
            # successive episodes advance the same RNG stream.
            state, _ = env.reset()
            states, actions, rewards = [], [], []
            episode_reward = 0.0

            for _ in range(max_steps):
                # Stochastic while training, greedy while evaluating.
                action = self.get_action(state, stochastic=train)
                next_state, reward, terminated, truncated, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(float(reward))
                episode_reward += float(reward)

                state = next_state
                if terminated or truncated:
                    break

            # undiscounted return
            total_rewards.append(episode_reward)

            if train:
                self.train(states, actions, rewards)
        # END STUDENT SOLUTION
        return total_rewards


def graph_agents(
    graph_name,
    agents,
    env,
    max_steps,
    num_episodes,
    num_test_episodes,
    graph_every,
):
    print(f"Starting: {graph_name}")

    if agents[0].n != 0:
        graph_name += "_" + str(agents[0].n)

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    graph_dir = Path(__file__).resolve().parent / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    num_evals = num_episodes // graph_every
    D = np.zeros((len(agents), num_evals))

    for trial_idx, agent in enumerate(agents):
        for eval_idx in range(num_evals):
            agent.train()
            agent.run(env, max_steps, graph_every, train=True)

            agent.train(False)
            test_rewards = agent.run(env, max_steps, num_test_episodes, train=False)
            D[trial_idx, eval_idx] = np.mean(test_rewards)

    average_total_rewards = D.mean(axis=0)
    min_total_rewards = D.min(axis=0)
    max_total_rewards = D.max(axis=0)
    # END STUDENT SOLUTION

    # plot the total rewards
    xs = [(i + 1) * graph_every for i in range(len(average_total_rewards))]
    fig, ax = plt.subplots()
    plt.fill_between(xs, min_total_rewards, max_total_rewards, alpha=0.1)
    ax.plot(xs, average_total_rewards)
    ax.set_ylim(-max_steps * 0.01, max_steps * 1.1)
    ax.set_title(graph_name, fontsize=10)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Total Reward")
    fig.savefig(graph_dir / f"{graph_name}.png")
    plt.close(fig)
    print(f"Finished: {graph_name}")


def parse_args():
    mode_choices = ["REINFORCE", "REINFORCE_WITH_BASELINE", "A2C"]

    parser = argparse.ArgumentParser(description="Train an agent.")
    parser.add_argument(
        "--mode",
        type=str,
        default="REINFORCE",
        choices=mode_choices,
        help="Mode to run the agent in",
    )
    parser.add_argument("--n", type=int, default=0, help="The n to use for n step A2C")
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of runs to average over for graph",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=3500, help="Number of episodes to train for"
    )
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=20,
        help="Number of episodes to test for every eval step",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum number of steps in the environment",
    )
    parser.add_argument(
        "--env_name", type=str, default="CartPole-v1", help="Environment name"
    )
    parser.add_argument(
        "--graph_every", type=int, default=100, help="Graph every x episodes"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # init args, agents, and call graph_agents on the initialized agents
    # BEGIN STUDENT SOLUTION
    env = gym.make(args.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"

    agents = [ PolicyGradient( state_size, action_size, mode=args.mode, n=args.n, device=device) for _ in range(args.num_runs) ]
    graph_agents( args.mode, agents, env, args.max_steps, args.num_episodes, args.num_test_episodes, args.graph_every)
    env.close()
    # END STUDENT SOLUTION


if "__main__" == __name__:
    main()
