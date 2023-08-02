import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os
import random 

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

# 定义一个仿真环境
class BipedEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    # 初始化环境参数
    def __init__(
        self,
        xml_file=os.path.join(os.path.join(os.path.dirname(__file__),
                                'asset', "Legged_wheel3.xml")),
        ctrl_cost_weight=0.005,
        healthy_reward=1.0,
        healthy_z_range=0.25,
        reset_noise_scale=0.1,
    ):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property # 计算健康奖励
    def healthy_reward(self):
        return (
            float(self.is_healthy) * self._healthy_reward
        )

    # 计算控制成本
    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property  # 是否倾倒
    def is_healthy(self):
        min_z = self._healthy_z_range
        is_healthy = (self.get_body_com("base_link")[2] > min_z) and (not self.bump_base())
        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy
        return done

    # 碰到大腿以上
    def bump_base(self):
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            geom1 = self.sim.model.geom_id2name(contact.geom1)
            geom2 = self.sim.model.geom_id2name(contact.geom2)
            if (geom1 in ['base1', 'base2', 'base3', 'base4', 'left_thigh1', 'left_thigh2', 'left_thigh3',
                          'right_thigh1', 'right_thigh2', 'right_thigh3']) or (
                    geom2 in ['base1', 'base2', 'base3', 'base4', 'left_thigh1', 'left_thigh2', 'left_thigh3',
                              'right_thigh1', 'right_thigh2', 'right_thigh3']):
                return True
        return False
    
    # 执行仿真中的一步
    def step(self, action):
        xy_target = self.data.get_geom_xpos("end")[:2].copy()  # 获取目标位置
        self.xy_target = xy_target
        xy_position_before = self.get_body_com("base_link")[:2].copy()  # 更新前位置
        xy_distance_before = abs(xy_target - xy_position_before)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("base_link")[:2].copy()   # 更新后位置
        xy_distance_after = abs(xy_target - xy_position_after)
        xy_velocity = -(xy_distance_after - xy_distance_before) / self.dt  # 速度
        x_velocity, y_velocity = xy_velocity
        ctrl_cost = self.control_cost(action)  # 控制损失
        quad_impact_cost = 0.5e-6 * np.square(self.sim.data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)

        reward = 10*x_velocity + 10*y_velocity + self.healthy_reward

        done = self.done
        # 判断是否到达终点
        if done is False:
            if np.sum(xy_distance_after) < 0.1:
                done = True
                reward = reward + 100
        observation = self._get_obs()
        info = {}
        return observation, reward, done, info

    # 获取当前状态的观察值
    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy() #身体部位
        velocity = self.sim.data.qvel.flat.copy() #速度

        observations = np.concatenate((position, velocity, self.xy_target)) #目标位置

        return observations

    # 重置模型
    def reset_model(self):
        self.sim.data.geom_xpos[self.model.geom_name2id("end")][0] = self._reset_noise_scale * random.random()
        self.sim.data.geom_xpos[self.model.geom_name2id("end")][1] = self._reset_noise_scale * random.random()
        observation = self._get_obs()
        return observation

    # 可视化查看器
    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
