import gym  # OpenAI Gym에서 환경을 로드합니다.
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 하이퍼파라미터 설정
lr = 0.0005  # 학습률
gamma = 0.99  # 할인율
buffer_limit = 50000  # Replay 버퍼 크기 제한
batch_size = 32  # 미니배치 크기

# Replay 버퍼 클래스 정의
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)  # deque를 이용해 버퍼 생성

    def put(self, transition):
        self.buffer.append(transition)  # 버퍼에 transition 추가

    def sample(self, n):
        '''
        아래 코드는 ReplayBuffer 클래스 내에 있는 sample 메서드로, 버퍼에서 무작위로 n개의 transition(상태, 행동, 보상, 다음 상태, 종료 여부)을 샘플링하여 미니배치를 구성하는 역할을 합니다.
        - `self.buffer`: ReplayBuffer 클래스에 있는 버퍼(deque)입니다.
        - `n`: 샘플링할 transition의 개수를 나타냅니다.

        여기서 샘플링된 각 transition은 's' (상태), 'a' (행동), 'r' (보상), 's_prime' (다음 상태), 'done_mask' (종료 여부)로 구성됩니다.

        샘플링된 transition은 미니배치로 사용되기 위해 각각의 요소를 분리하여 리스트(`s_lst`, `a_lst`, `r_lst`, `s_prime_lst`, `done_mask_lst`)에 추가합니다. 이 때, 'a', 'r', 'done_mask'는 리스트로 추가됩니다. 이는 PyTorch의 Tensor에 맞추기 위함입니다.

        그리고 마지막에는 분리된 리스트를 PyTorch의 Tensor로 변환하여 반환합니다. 반환되는 값은 상태(`s_lst`), 행동(`a_lst`), 보상(`r_lst`), 다음 상태(`s_prime_lst`), 종료 여부(`done_mask_lst`)를 각각 Tensor로 변환하여 반환합니다. 이때 데이터 타입은 torch.float32로 설정됩니다. 

        이러한 샘플링 과정은 DQN 학습 시에 미니배치를 생성하고, 해당 미니배치를 활용하여 네트워크를 학습시키는 데 사용됩니다.
        '''
        mini_batch = random.sample(self.buffer, n)  # 버퍼에서 무작위로 n개의 transition 샘플링
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
        return len(self.buffer)  # 버퍼 크기 반환

# Q-network 모델 정의
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(8, 256)  # 입력 크기 8, 첫 번째 은닉층 뉴런 수 256
        self.fc2 = nn.Linear(256, 256)  # 두 번째 은닉층 뉴런 수 256
        self.fc3 = nn.Linear(256, 4)  # 출력 크기 4 (행동 공간의 크기)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU 활성화 함수를 적용한 첫 번째 은닉층
        x = F.relu(self.fc2(x))  # ReLU 활성화 함수를 적용한 두 번째 은닉층
        x = self.fc3(x)  # 출력층
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)  # 입실론-탐욕적 정책에 따라 무작위 행동 선택
        else:
            return out.argmax().item()  # Q-value가 가장 높은 행동 선택

# DQN 모델 학습 함수
def train(q, q_target, memory, optimizer):
    '''
    위 코드는 DQN 모델을 학습하는 train 함수입니다. 주요한 작업들을 다음과 같이 수행합니다:

        1. **미니배치 샘플링**: `memory.sample(batch_size)`를 호출하여 replay buffer로부터 크기가 `batch_size`인 미니배치를 샘플링합니다. 이 미니배치는 상태(`s`), 행동(`a`), 보상(`r`), 다음 상태(`s_prime`), 종료 여부(`done_mask`)를 포함합니다.

        2. **Q-value 계산**: 현재 Q-network를 사용하여 상태(`s`)를 입력으로 받아 Q-value(`q_out`)를 계산합니다. 이 때, 선택된 행동(`a`)에 대한 Q-value를 가져옵니다(`q_a`). 이는 해당 상태에서 취한 행동에 대한 예측된 Q-value입니다.

        3. **타깃 값 계산**: DQN 알고리즘에 따라 타깃 Q-value를 계산합니다. 다음 상태(`s_prime`)를 입력으로 받아서 Target Q-network를 통해 가장 큰 Q-value를 취한 후, 해당 값에 할인율(`gamma`)을 곱하고 종료 여부(`done_mask`)를 적용하여 타깃 Q-value(`target`)를 얻습니다.

        4. **손실 계산 및 역전파**: MSE Loss 함수를 사용하여 예측된 Q-value(`q_a`)와 타깃 Q-value(`target`) 간의 손실을 계산합니다. 그리고 이를 통해 네트워크를 학습하기 위해 역전파를 수행합니다.

        5. **옵티마이저 업데이트**: 최적화된 그래디언트를 사용하여 네트워크의 가중치를 업데이트합니다.

    이러한 단계들은 DQN의 핵심 알고리즘을 따르며, Q-network와 Target Q-network를 사용하여 TD(시간차) 학습을 수행하고, 손실을 최소화하여 네트워크를 학습시키는 과정입니다.
    '''
    for i in range(10):  # 10번의 반복 학습
        s, a, r, s_prime, done_mask = memory.sample(batch_size)  # 미니배치 샘플링

        q_out = q(s)  # 현재 Q-network로 예측한 Q-value
        q_a = q_out.gather(1, a)  # 취한 행동에 대한 Q-value

        # DQN 알고리즘에 따라 타깃 값 계산
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask

        # MSE 손실 계산
        loss = F.mse_loss(q_a.to(torch.float32), target.to(torch.float32))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 메인 함수
def main(train_flag:bool=True):
    q = Qnet()  # Q-network 생성
    q_target = Qnet()  # Target Q-network 생성
    q_target.load_state_dict(q.state_dict())  # Target Q-network 초기화

    if train_flag:
        print("모델 학습 중")
        env = gym.make("LunarLander-v2")  # LunarLander-v2 환경 로드
        episode = 100  # 학습 에피소드 수
    if not train_flag:
        print("모델 불러오기")
        saved_weights_path = "./dqn_model.pth"  # 저장된 모델 경로
        q.load_state_dict(torch.load(saved_weights_path))  # 저장된 모델 불러오기
        env = gym.make("LunarLander-v2", render_mode="human")  # 시각화를 위한 환경 로드
        episode = 10  # 테스트 에피소드 수

    memory = ReplayBuffer()  # Replay 버퍼 생성
    print_interval = 20  # 정보 출력 주기
    score = 0.0  # 점수 초기화
    optimizer = optim.Adam(q.parameters(), lr=lr)  # Adam 옵티마이저 설정

    for n_epi in range(episode):  # 에피소드 반복
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # 입실론 감소
        s, _ = env.reset()  # 환경 초기화
        done = False

        while not done:
            a = int(q.sample_action(torch.from_numpy(s).float(), epsilon))  # 행동 선택
            s_prime, r, terminated, truncated, info = env.step(a)  # 환경 상호작용
            done = (terminated or truncated)  # 종료 여부 체크
            done_mask = 0.0 if done else 1.0  # 종료 마스크 설정
            memory.put((s, a, r / 100.0, s_prime, done_mask))  # Replay 버퍼에 transition 추가
            s = s_prime  # 다음 상태로 이동

            score += r  # 점수 누적
            if done:
                break

        if memory.size() >

 2000:  # 버퍼 크기가 일정 이상일 때 학습
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:  # 일정 주기마다 정보 출력
            q_target.load_state_dict(q.state_dict())  # Target Q-network 업데이트
            print("에피소드: {}, 점수: {:.1f}, 버퍼 크기: {}, 입실론: {:.1f}%".format(
                n_epi, score / print_interval, memory.size(), epsilon * 100))
            score = 0.0

    if train_flag:
        torch.save(q.state_dict(), "./dqn_model.pth")  # 학습한 모델 저장
        print("DQN 모델이 저장되었습니다.")
    env.close()  # 환경 종료

if __name__ == "__main__":
    main(train_flag=True)  # 학습 플래그가 True일 경우 학습 진행
