import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os

from env import HESSMicrogridEnv,TrainConfig,EnvConfig
from models import DQNAgent,ReplayBuffer



def train_and_plot(df: pd.DataFrame, episodes: int, save_path: str = "dqn_jupyter_model.pt"):
    
    
    ecfg = EnvConfig()
    tcfg = TrainConfig()

    env = HESSMicrogridEnv(df, ecfg)
    obs_dim = 7  # The observation space now has 7 dimensions
    act_dim = env.action_space.n
    agent = DQNAgent(obs_dim, act_dim, tcfg)
    buf = ReplayBuffer(tcfg.replay_size, obs_dim)

    best_eval = -np.inf
    total_steps = 0
    ep_rewards = []
    ep_losses = []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_ret = 0.0
        ep_loss_sum = 0.0
        done = False
        term = False
        trunc = False

        while not (term or trunc):
            if total_steps < tcfg.start_steps:
                action = env.action_space.sample()
            else:
                action = agent.act(obs)
            
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            buf.push(obs, action, reward, next_obs, float(done))

            obs = next_obs
            ep_ret += reward
            total_steps += 1

            if total_steps > tcfg.train_after and total_steps % tcfg.train_freq == 0:
                loss = agent.update(buf, tcfg.batch_size)
                ep_loss_sum += loss
            
            if total_steps % tcfg.target_sync == 0:
                agent.sync_target()

        ep_rewards.append(ep_ret)
        ep_losses.append(ep_loss_sum / env.ep_len)

        print(f"Episode {ep:04d} | Return {ep_ret:8.2f} | Loss {ep_losses[-1]:8.4f} | Epsilon {agent.eps:.3f}")

        if ep_ret > best_eval:
            best_eval = ep_ret
            torch.save(agent.q.state_dict(), save_path)
            print(f"New best model saved with return: {best_eval:.2f}")

    print("Training complete.")
    
    # Plotting the metrics
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ep_rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(ep_losses)
    plt.title('Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()





def evaluate_and_plot(df, model_path="dqn_jupyter_model.pt", save_dir="results_eval"):
    os.makedirs(save_dir, exist_ok=True)

    ecfg = EnvConfig()
    tcfg = TrainConfig()

    env = HESSMicrogridEnv(df, ecfg)
    obs_dim = 7
    act_dim = env.action_space.n
    agent = DQNAgent(obs_dim, act_dim, tcfg)

    # Load trained weights
    agent.q.load_state_dict(torch.load(model_path, map_location=tcfg.device))
    agent.q.eval()

    # Storage for episode data
    batt_socs, sc_socs, loads, renews, prices, actions, rewards, grid_imports = [], [], [], [], [], [], [], []

    obs, _ = env.reset()
    done, term, trunc = False, False, False
    total_reward = 0.0

    while not (term or trunc):
        action = agent.act(obs)
        next_obs, reward, term, trunc, info = env.step(action)

        # Log values
        row = df.loc[env.ptr + env.t - 1]
        batt_socs.append(env.batt_soc)
        sc_socs.append(env.sc_soc)
        loads.append(row[ecfg.col_load_kw])
        renews.append(row[ecfg.col_renewable_kw])
        prices.append(row[ecfg.col_price])
        actions.append(action)
        rewards.append(reward)

        # Recompute grid import for logging
        p_batt_kw, p_sc_kw = env.action_list[action]
        grid_import_kw = max(0, row[ecfg.col_load_kw] - row[ecfg.col_renewable_kw] - max(0, p_batt_kw) - max(0, p_sc_kw))
        grid_imports.append(grid_import_kw)

        total_reward += reward
        obs = next_obs

    print(f"Evaluation completed. Total reward = {total_reward:.2f}")
    # Save evaluation results for comparison
    eval_results = pd.DataFrame({
        "Battery_SOC_kWh": batt_socs,
        "SC_SOC_kWh": sc_socs,
        "Grid_Import_kW": grid_imports,
        "Reward": rewards,
        "Load_kW": loads,
        "Renewable_kW": renews,
        "Price": prices,
        "Action": actions
    })
    eval_results.to_csv(f"{save_dir}/dqn_eval_results.csv", index=False)
    print(f"Saved evaluation results to {save_dir}/dqn_eval_results.csv")


    # =======================
    # Plotting
    # =======================
    timesteps = range(len(batt_socs))

    plt.figure(figsize=(12,6))
    plt.plot(timesteps, batt_socs, label="Battery SOC")
    plt.plot(timesteps, sc_socs, label="SC SOC")
    plt.xlabel("Timestep")
    plt.ylabel("State of Charge")
    plt.title("Battery & Supercapacitor SOC")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/soc_curves.png")
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(timesteps, loads, label="Load Demand (kW)")
    plt.plot(timesteps, renews, label="Renewable Supply (kW)")
    plt.plot(timesteps, grid_imports, label="Grid Import (kW)")
    plt.xlabel("Timestep")
    plt.ylabel("Power (kW)")
    plt.title("Load vs Renewable vs Grid Import")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/supply_demand.png")
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(timesteps, prices, label="Electricity Price")
    plt.xlabel("Timestep")
    plt.ylabel("USD/kWh")
    plt.title("Electricity Price Signal")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/prices.png")
    plt.show()

    plt.figure(figsize=(12,6))
    plt.step(timesteps, actions, label="Agent Action")
    plt.xlabel("Timestep")
    plt.ylabel("Action Index")
    plt.title("Actions Chosen by DQN Agent")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/actions.png")
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(timesteps, np.cumsum(rewards), label="Cumulative Reward")
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Reward")
    plt.title("Reward Accumulation During Evaluation")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/rewards.png")
    plt.show()


if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/preprocessed_hess.csv')

    #Run the training data
    train_and_plot(df, episodes=50)
    
    # Evaluate the training model
    evaluate_and_plot(df, model_path="dqn_jupyter_model.pt", save_dir="results_eval")

    
    
    