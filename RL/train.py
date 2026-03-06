import torch
import torch.nn as nn
import torch.optim as optim
from packing_env import PackingEnv
from policy_network import PolicyNetwork
import numpy as np
import os
import matplotlib.pyplot as plt

# ----------------------------
# 환경 세팅
# ----------------------------
basket_size = (200, 200, 200)
env = PackingEnv(basket_size, grid=40)

# ----------------------------
# 모델 + optimizer
# ----------------------------
state_dim = 3
model = PolicyNetwork(state_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ----------------------------
# 체크포인트 로드
# ----------------------------
checkpoint_path = "checkpoint.pth"
start_episode = 0

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_episode = checkpoint["episode"] + 1
    print(f"Checkpoint loaded, starting from episode {start_episode}")

# ----------------------------
# 랜덤 객체 생성 함수
# ----------------------------
def generate_random_objects(num_objects):
    objects = []
    for _ in range(num_objects):
        xsize = np.random.randint(20, 60)    # mm 단위 가로
        ysize = np.random.randint(20, 60)    # mm 단위 세로
        zsize = np.random.randint(10, 40)    # mm 단위 높이
        durability = np.random.randint(1, 6) # 1~5
        objects.append({
            "size": (xsize, ysize, zsize),
            "durability": durability
        })
    return objects

# ----------------------------
# 학습 루프
# ----------------------------
episodes = 50000  # 에피소드 수 늘림
reward_history = []  # reward 기록용 리스트

for ep in range(start_episode, episodes):
    objects_per_episode = np.random.randint(3, 9)  # 3~8개 랜덤 객체
    objects = generate_random_objects(objects_per_episode)
    env.set_objects(objects)

    state = env.reset()
    done = False
    total_reward = 0

    # ----------------------------
    # Exploration Noise 조절
    # ----------------------------
    # 초기에는 크게, 후반에는 작게
    noise_std = max(0.05 * (1 - ep / episodes), 0.01)  # 학습 후반부 noise 감소

    while not done:
        heightmap = torch.tensor(env.heightmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        features = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        action_mean = model(heightmap, features)
        action_dist = torch.distributions.Normal(action_mean, torch.tensor(noise_std))
        action_sample = action_dist.sample()
        action_sample = torch.clamp(action_sample, 0.0, 1.0)
        log_prob = action_dist.log_prob(action_sample).sum(dim=-1)

        next_state, reward, done, _ = env.step(action_sample.detach().numpy()[0])
        total_reward += reward

        reward_norm = (reward - 0) / (10 + 1e-5)  # 최대 reward 10 가정
        loss = -log_prob * reward_norm
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

    reward_history.append(total_reward)

    # ----------------------------
    # 로그 출력
    # ----------------------------
    if ep % 50 == 0:
        print(f"episode: {ep}, reward: {total_reward:.3f}")

    # ----------------------------
    # 체크포인트 저장
    # ----------------------------
    if ep % 200 == 0:
        torch.save({
            "episode": ep,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved at episode {ep}")

# ----------------------------
# 학습 완료 후 reward 그래프
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(reward_history, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward History")
plt.grid(True)
plt.legend()
plt.show()