try:
    from dm_control import suite
except ImportError:
    suite = None
import numpy as np

from ray.rllib.env.dm_control_wrapper import DMCEnv
from ray.rllib.env.dm_control_wrapper import _flatten_obs
from numpy.random import default_rng


class RDMCEnv(DMCEnv):
    def __init__(self,
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
        steps = np.arange(-0.25, 3.25, 0.25)
        steps = np.concatenate([steps[:int(np.where(steps == 0)[0])], steps[int(np.where(steps == 0)[0] + 1):]])
        change_values = self.rng.choice(steps, len(sym_bodies))
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
        steps = np.arange(-0.25, 3.25, 0.25)
        steps = np.concatenate([steps[:int(np.where(steps == 0)[0])], steps[int(np.where(steps == 0)[0] + 1):]])
        change_values = self.rng.choice(steps, len(sym_bodies))
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

        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs
