# ⚡ HESS-RL: Hybrid Energy Storage System with Reinforcement Learning

This project implements a **Hybrid Energy Storage System (HESS)** for microgrid energy management using **Deep Q-Learning (DQN)**.  
The goal is to optimize the interaction between **battery storage** and **supercapacitors**, reducing grid imports and minimizing operational costs, while comparing performance with a **rule-based baseline controller**.

---

## 🚀 Features
- Custom **Gymnasium environment** for HESS microgrid.
- **Deep Q-Network (DQN)** implementation in PyTorch.
- **Replay buffer** for stable training.
- **Rule-based controller (RBC)** for baseline comparison.
- Evaluation plots:  
  - Battery & Supercapacitor SOC  
  - Load vs Renewable vs Grid Import  
  - Electricity Price  
  - Agent Actions  
  - Reward Accumulation  

---

## 📂 Project Structure

```hess-rl/
│── data/ <- Preprocessed dataset (CSV)
│── src/ <- Core source code
│ │── env.py <- Gym environment for HESS microgrid
│ │── models.py <- DQN agent, replay buffer, and Q-network
│ │── ruler.py <- Rule-based baseline control
│ │── main.py <- Training & evaluation script
│ │── init.py
│── results_eval/ <- Saved evaluation plots and results
│── dqn_jupyter_model.pt <- Saved trained model (PyTorch)
│── README.md <- Project documentation
```


## 🔧 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Tito-Curtis/Reinforcement-Learning/hess-rl.git
   cd hess-rl
   pip install -r requirements.txt
   python src/main.py
  ```

