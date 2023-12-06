---

# LunarLander Environment

## 소개
OpenAI Gym의 LunarLander 환경은 강화 학습에서 비행선을 안전하게 착륙시키는 문제를 다루는 환경입니다. 로켓을 조종하여 초기 위치와 속도에서 출발하여 착륙 지점에 안전하게 착륙해야 합니다. 이 환경은 공중에서의 제어와 물리적 상호작용을 기반으로 합니다.

## 목표
이 환경의 목표는 로켓이 착륙 지점에 안전하게 착륙하도록 에이전트를 훈련하는 것입니다. 로켓은 다양한 행동을 통해 보상을 받으며, 착륙 지점에 부드럽게 착륙하거나 비행 중에 너무 많은 움직임을 피하는 등의 목표를 달성해야 합니다.

## 상태(State) 정보
LunarLander 환경은 다음과 같은 상태 정보를 제공합니다:
- 로켓의 x, y 좌표 및 속도
- 로켓의 각도 및 각 속도
- 왼쪽 및 오른쪽 다리의 접촉 여부

## 행동(Action)
로켓은 다음과 같은 행동을 선택할 수 있습니다:
- 아무 행동도 하지 않기
- 왼쪽으로 엔진을 켜기
- 오른쪽으로 엔진을 켜기
- 양쪽 엔진 모두 켜기

## 보상(Reward)
로켓이 착륙 지점에 안전하게 착륙할 때 마다 양수의 보상을 받으며, 로켓이 착륙을 실패하거나 너무 많이 움직이는 경우 음수의 보상을 받습니다.

## 설치
OpenAI Gym이 설치되어 있지 않은 경우, 다음 명령을 사용하여 설치합니다:
```bash
pip install gym
```

## 사용법
```python
import gym

env = gym.make('LunarLander-v2')
observation = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # 임의의 행동 선택 (랜덤)
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()

env.close()
```

## 환경 버전 정보
- OpenAI Gym: v0.29.1
- LunarLander: v2

---
