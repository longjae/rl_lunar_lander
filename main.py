import gymnasium as gym
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

lr = 0.0005
gamma = 0.99
buffer_limit = 50000
batch_size = 32

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float32), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float32), \
               torch.tensor(done_mask_lst, dtype=torch.float32)

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else :
            return out.argmax().item()

def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)

        # DQN
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)

        target = r + gamma * max_q_prime * done_mask

        # MSE Loss
        loss = F.mse_loss(q_a.to(torch.float32), target.to(torch.float32))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main(train:bool=True):
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    if train:
        print("train model")
        env = gym.make("LunarLander-v2")
    if not train:
        print("load model")
        saved_weights_path = "./dqn_model.pth"
        q.load_state_dict(torch.load(saved_weights_path))
        env = gym.make("LunarLander-v2", render_mode="human")
    memory = ReplayBuffer()
    
    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=lr)
    
    for n_epi in range(20):
        epsilon = max(0.01, 0.08-0.01*(n_epi/200))
        s, _ = env.reset()
        done = False
        
        while not done:
            a = int(q.sample_action(torch.from_numpy(s).float(), epsilon))
            s_prime, r, terminated, truncated, info = env.step(a)
            done = (terminated or truncated)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime
            
            score += r
            if done:
                break
        # print("===================")
        # print("state:", s)
        # print("action:", a)
        # print("reward:", r)
        # print("state_prime:", s_prime)
        # print("done_mask:", done_mask)
        # print("===================")
            
        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)
        
        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    if train:
        torch.save(q.state_dict(), "./dqn_model.pth")
        print("DQN model has saved.")
    env.close()

if __name__ == "__main__":
    main(True)