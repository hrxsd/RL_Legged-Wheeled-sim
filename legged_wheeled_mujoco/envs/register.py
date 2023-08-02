from gym.envs.registration import register
import numpy as np

register(
    id="Biped-v0", # 环境id
    entry_point="envs.biped:BipedEnv", # 环境类入口
    max_episode_steps=2000, # 一个episode的最大步数
    reward_threshold=6000.0, # 完成任务的奖励阈值
)


