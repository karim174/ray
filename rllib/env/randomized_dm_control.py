try:
    from dm_control import suite
except ImportError:
    suite = None
import numpy as np
import random
from ray.rllib.env.dm_control_wrapper import DMCEnv, MultiDMCEnv
from ray.rllib.env.dm_control_wrapper import _flatten_obs
from numpy.random import default_rng


class RDMCEnv(DMCEnv):
    def __init__(self,
                 interval,
                 *args,
                 **kwargs):
        '''

        :param args:
        :param kwargs:

        This environment contains a randomized version of a certain intraclass system
        '''

        super().__init__(*args, **kwargs)

        self.rng = default_rng()
        self.bodies = self._env.physics.named.model.body_mass._axes[0]._names[1:]
        self.orig_body_mass = self._env.physics.named.model.body_mass[self.bodies]
        self.orig_geom_size = self._env.physics.named.model.geom_size[self.bodies]
        sym_bodies = set(
            body.split('_', 1)[1] if body.find("_") != -1 else body for body in self.bodies)  # or seen_add(body))
        change_mask = self.rng.binomial(1, p=0.5, size=len(sym_bodies))

        range = interval.split()
        range = map(float, range)
        range = list(range)
        self.steps = np.arange(*range)
        if np.where(self.steps == 0)[0].shape[0] != 0:
            self.steps = np.concatenate([self.steps[:int(np.where(self.steps == 0)[0])],
                                         self.steps[int(np.where(self.steps == 0)[0] + 1):]])
        change_values = self.rng.choice(self.steps, len(sym_bodies))
        change_factor = 1 + change_values * change_mask
        sym_bdy_cng_dict = dict(zip(sym_bodies, change_factor))
        bm_bdy_cng = np.array(
            [sym_bdy_cng_dict[body.split('_', 1)[1]] if body.find("_") != -1 else sym_bdy_cng_dict[body] for body in
             self.bodies])

        bodies_mass = self.orig_body_mass * bm_bdy_cng
        orig_size = self.orig_geom_size.copy()
        orig_size[orig_size == 0] = 1
        orig_vol = np.prod(orig_size, axis=-1)
        self.approx_inv_density = orig_vol/self.orig_body_mass

        approx_inv_density = np.array([1 if i == 1 else self.approx_inv_density[ind] for ind, i in enumerate(bm_bdy_cng)])
        bodies_sizes = self.orig_geom_size * (1 + approx_inv_density * (bm_bdy_cng - 1))[:, None]

        with self._env.physics.reset_context():
            self._env.physics.named.model.body_mass[self.bodies] = bodies_mass
            self._env.physics.named.model.geom_size[self.bodies] = bodies_sizes

    def reset(self):
        # return everything to normal parameters
        with self._env.physics.reset_context():
            self._env.physics.named.model.body_mass[self.bodies] = self.orig_body_mass
            self._env.physics.named.model.geom_size[self.bodies] = self.orig_geom_size

        time_step = self._env.reset()
        sym_bodies = set(
            body.split('_', 1)[1] if body.find("_") != -1 else body for body in self.bodies)  # or seen_add(body))
        change_mask = self.rng.binomial(1, p=0.5, size=len(sym_bodies))
        change_values = self.rng.choice(self.steps, len(sym_bodies))
        change_factor = 1 + change_values * change_mask
        sym_bdy_cng_dict = dict(zip(sym_bodies, change_factor))
        bm_bdy_cng = np.array(
            [sym_bdy_cng_dict[body.split('_', 1)[1]] if body.find("_") != -1 else sym_bdy_cng_dict[body] for body in
             self.bodies])

        bodies_mass = self.orig_body_mass * bm_bdy_cng
        approx_inv_density = np.array(
            [1 if i == 1 else self.approx_inv_density[ind] for ind, i in enumerate(bm_bdy_cng)])
        bodies_sizes = self.orig_geom_size * (1 + (approx_inv_density * (bm_bdy_cng-1)))[:, None]

        with self._env.physics.reset_context():
            self._env.physics.named.model.body_mass[self.bodies] = bodies_mass
            self._env.physics.named.model.geom_size[self.bodies] = bodies_sizes

        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs


class ARDMCEnv(MultiDMCEnv):
    def __init__(self,
                 interval,
                 *args,
                 **kwargs):
        '''

        :param args:
        :param kwargs:

        This environment contains a randomized version of a certain intraclass system
        '''

        self.interval = interval
        self.args = args
        self.kwargs = kwargs
        self.rng = default_rng()


        super().__init__(*args, **kwargs)

        self.bodies = []
        self.orig_body_mass = []
        self.orig_geom_size = []
        self.approx_inv_density = []
        for i, _ in enumerate(self.tasks):

            self.bodies.append(self._env[i].physics.named.model.body_mass._axes[0]._names[1:])

            self.orig_body_mass.append(self._env[i].physics.named.model.body_mass[self.bodies[i]])
            self.orig_geom_size.append(self._env[i].physics.named.model.geom_size[self.bodies[i]])

            orig_size = self.orig_geom_size[i].copy()
            orig_size[orig_size == 0] = 1
            orig_vol = np.prod(orig_size, axis=-1)
            self.approx_inv_density.append(orig_vol / self.orig_body_mass[i])

        range = interval.split()
        range = map(float, range)
        range = list(range)
        self.steps = np.arange(*range)
        if np.where(self.steps == 0)[0].shape[0] != 0:
            self.steps = np.concatenate([self.steps[:int(np.where(self.steps == 0)[0])],
                                         self.steps[int(np.where(self.steps == 0)[0] + 1):]])


        self.env_id = self.rng.choice(len(self.tasks))
        sym_bodies = set(
            body.split('_', 1)[1] if body.find("_") != -1 else body for body in self.bodies[self.env_id])
        change_mask = self.rng.binomial(1, p=0.5, size=len(sym_bodies))
        change_values = self.rng.choice(self.steps, len(sym_bodies))
        change_factor = 1 + change_values * change_mask
        sym_bdy_cng_dict = dict(zip(sym_bodies, change_factor))
        bm_bdy_cng = np.array(
            [sym_bdy_cng_dict[body.split('_', 1)[1]] if body.find("_") != -1 else sym_bdy_cng_dict[body] for body in
             self.bodies[self.env_id]])

        bodies_mass = self.orig_body_mass[self.env_id] * bm_bdy_cng

        approx_inv_density = np.array([1 if i == 1 else self.approx_inv_density[self.env_id][ind] for ind, i in enumerate(bm_bdy_cng)])
        bodies_sizes = self.orig_geom_size[self.env_id] * (1 + approx_inv_density * (bm_bdy_cng - 1))[:, None]

        with self._env[self.env_id].physics.reset_context():
            self._env[self.env_id].physics.named.model.body_mass[self.bodies[self.env_id]] = bodies_mass
            self._env[self.env_id].physics.named.model.geom_size[self.bodies[self.env_id]] = bodies_sizes

    def reset(self):
        # return everything to normal parameters

        with self._env[self.env_id].physics.reset_context():
            self._env[self.env_id].physics.named.model.body_mass[self.bodies[self.env_id]] = self.orig_body_mass[self.env_id]
            self._env[self.env_id].physics.named.model.geom_size[self.bodies[self.env_id]] = self.orig_geom_size[self.env_id]

        self.env_id = self.rng.choice(len(self.tasks))
        time_step = self._env[self.env_id].reset()
        sym_bodies = set(
            body.split('_', 1)[1] if body.find("_") != -1 else body for body in self.bodies[self.env_id])  # or seen_add(body))
        change_mask = self.rng.binomial(1, p=0.5, size=len(sym_bodies))
        change_values = self.rng.choice(self.steps, len(sym_bodies))
        change_factor = 1 + change_values * change_mask
        sym_bdy_cng_dict = dict(zip(sym_bodies, change_factor))
        bm_bdy_cng = np.array(
            [sym_bdy_cng_dict[body.split('_', 1)[1]] if body.find("_") != -1 else sym_bdy_cng_dict[body] for body in
             self.bodies[self.env_id]])

        bodies_mass = self.orig_body_mass[self.env_id] * bm_bdy_cng
        approx_inv_density = np.array(
            [1 if i == 1 else self.approx_inv_density[self.env_id][ind] for ind, i in enumerate(bm_bdy_cng)])
        bodies_sizes = self.orig_geom_size[self.env_id] * (1 + (approx_inv_density * (bm_bdy_cng - 1)))[:, None]

        with self._env[self.env_id].physics.reset_context():
            self._env[self.env_id].physics.named.model.body_mass[self.bodies[self.env_id]] = bodies_mass
            self._env[self.env_id].physics.named.model.geom_size[self.bodies[self.env_id]] = bodies_sizes
        time_step = self._env[self.env_id].reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs