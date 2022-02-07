import logging

import random
import numpy as np

from ray.rllib.agents import with_common_config
from ray.rllib.agents.dreamer.dreamer_torch_policy import DreamerTorchPolicy
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.execution.common import STEPS_SAMPLED_COUNTER, \
    LEARNER_INFO, _get_shared_metrics
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.agents.dreamer.dreamer_model import DreamerModel
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.utils.typing import SampleBatchType

logger = logging.getLogger(__name__)

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # PlaNET Model LR
    "td_model_lr": 6e-4,
    # Actor LR
    "actor_lr": 8e-5,
    # Critic LR
    "critic_lr": 8e-5,
    # Grad Clipping
    "grad_clip": 100.0,
    # Discount
    "discount": 0.99,
    # Lambda
    "lambda": 0.95,
    # Clipping is done inherently via policy tanh.
    "clip_actions": False,
    # Training iterations per data collection from real env
    "dreamer_train_iters": 100,
    # Horizon for Enviornment (1000 for Mujoco/DMC)
    "horizon": 1000,
    # Number of episodes to sample for Loss Calculation
    "batch_size": 50,
    # Length of each episode to sample for Loss Calculation
    "batch_length": 50,
    # Imagination Horizon for Training Actor and Critic
    "imagine_horizon": 15,
    # Free Nats
    "free_nats": 3.0,
    # KL Coeff for the Model Loss
    "kl_coeff": 1.0,
    # Reinforce Coeff for the Model Loss
    "rei_coeff": 1.0,
    # Entropy Coeff for the Model Loss
    "ent_coeff": 0.1,
    # Weight change Coeff for the Model Loss
    "wc_coeff": 0.1,
    # Distributed Dreamer not implemented yet
    "num_workers": 0,
    # Prefill Timesteps
    "prefill_timesteps": 5000,
    # This should be kept at 1 to preserve sample efficiency
    "num_envs_per_worker": 1,
    # Exploration Gaussian
    "explore_noise": 0.3,
    # Batch mode
    "batch_mode": "complete_episodes",
    # Custom Model
    "dreamer_model": {
        "custom_model": DreamerModel,
        # RSSM/PlaNET parameters
        "deter_size": 100,
        "stoch_size": 50,
        # CNN Decoder Encoder
        "depth_size": 32,
        # General Network Parameters
        "hidden_size": 150,  # orig was 400 but not effective
        # Action STD
        "action_init_std": 5.0,
        "hyper_size": 12,
        "n_z": 9,
        "decay": 0.9,
        "add_mask": False,
        "dembed_in_state": True,
        "w_cng_reg": True,
        "ext_context": 5,
        "memory_tau": 10,
        'num_transformer_units': 6,
        'num_heads': 6,
        'atten_size': 64
    },

    "env_config": {
        # Repeats action send by policy for frame_skip times in env
        "frame_skip": 2,
    }
})
# __sphinx_doc_end__
# yapf: enable


class EpisodicBuffer(object):
    def __init__(self, max_length: int = 1000,
                 length: int = 50, ext_context: int = 1, memory_tau: int = 10):
        """Data structure that stores episodes and samples chunks
        of size length from episodes

        Args:
            max_length: Maximum episodes it can store
            length: Episode chunking lengh in sample()
        """

        # Stores all episodes into a list: List[SampleBatchType]
        self.episodes = []
        self.mems = []
        self.max_length = max_length
        self.timesteps = 0
        self.length = length
        self.memory_tau = memory_tau
        self.ext_context = ext_context
        self.sample_length = self.length + self.ext_context

    def add(self, batch: SampleBatchType):
        """Splits a SampleBatch into episodes and adds episodes
        to the episode buffer

        Args:
            batch: SampleBatch to be added
        """

        self.timesteps += batch.count
        episodes = batch.split_by_episode()
        for i, e in enumerate(episodes):
            episodes[i] = self.preprocess_episode(e)
        self.episodes.extend(episodes)

        if len(self.episodes) > self.max_length:
            delta = len(self.episodes) - self.max_length
            # Drop oldest episodes
            self.episodes = self.episodes[delta:]

    def preprocess_episode(self, episode: SampleBatchType):
        """Batch format should be in the form of (s_t, a_(t-1), r_(t-1))
        When t=0, the resetted obs is paired with action and reward of 0.

        Args:
            episode: SampleBatch representing an episode
        """

        obs = episode["obs"]
        new_obs = episode["new_obs"]
        action = episode["actions"]
        reward = episode["rewards"]
        mem_o = np.array(episode["state_out_6"])
        mem_o  = np.reshape(mem_o, (*mem_o.shape[:2], self.memory_tau, -1))[:, :, -1, :]

        #one for the memories and treat it as if it is a regular thing
        act_shape = action.shape
        act_reset = np.array([0.0] * act_shape[-1])[None]
        rew_reset = np.array(0.0)[None]
        mem_o_reset = np.zeros_like(mem_o[0])[None]
        obs_end = np.array(new_obs[act_shape[0] - 1])[None]

        batch_obs = np.concatenate([obs, obs_end], axis=0)
        batch_action = np.concatenate([act_reset, action], axis=0)
        batch_rew = np.concatenate([rew_reset, reward], axis=0)
        batch_mem = np.concatenate([mem_o_reset, mem_o], axis=0)
        new_batch = {
            "obs": batch_obs,
            "rewards": batch_rew,
            "actions": batch_action,
            "mems": batch_mem
        }
        return SampleBatch(new_batch)

    def sample(self, batch_size: int):
        """Samples [batch_size, length] from the list of episodes

        Args:
            batch_size: batch_size to be sampled
        """
        episodes_buffer = []
        ep_t_ids = []
        while len(episodes_buffer) < batch_size:
            rand_index = random.randint(0, len(self.episodes) - 1)
            episode = self.episodes[rand_index]

            if episode.count < self.length:
                continue
            available = episode.count - self.length #this is for obs
            index = int(random.randint(0, available))
            #add ext_context to the data
            episode_slice = episode.slice(index, index + self.length)
            mem_dim = episode_slice['mems'].shape[-1]
            #print(f'start from {index} till  {index + self.length}')
            #for k, v in episode_slice.items():
            #    print(f' for the {k} shape is {v.shape}')
            ep_t_ids.append([rand_index, index])
            episode_slice=self.add_cntx_tau(rand_index, index, episode_slice)
            #for k, v in episode_slice.items():
            #    print(f' for the {k} shape is {v.shape}')

            episodes_buffer.append(episode_slice)

        batch = {}
        for k in episodes_buffer[0].keys():
            batch[k] = np.stack([e[k] for e in episodes_buffer], axis=0)
            #print(batch[k].shape)

        return ep_t_ids, SampleBatch(batch)

    def add_cntx_tau(self, episode_id, index, episode_slice):
        """completes the episode slice with data from buffer or pad to the left.

                Args:
                    episode_id: episode to index from
                    index: the index of the slice in the buffer
                    episode_slice: sample batch to pad or compliment
                """

        batch = {}
        for k, v in episode_slice.items():
            to_add = self.ext_context if k != 'mems' else self.memory_tau
            start_id_buffer = max(index - to_add, 0)
            from_buffer = index - start_id_buffer
            zero_pad = to_add - from_buffer
            init_time = v.shape[0]
            #print(f'for the {k} in episode {episode_id}: get from {start_id_buffer} till {index} in buffer for {from_buffer} elements and '
            #      f'add {zero_pad} to cover for missing items in {to_add}')
            if k != 'mems':
                batch[k] = v
            if from_buffer:
                ext_from_buffer = self.episodes[episode_id][k][start_id_buffer:index]
                batch[k] = np.concatenate([ext_from_buffer, v], axis = 0) if k != 'mems' else ext_from_buffer

            if zero_pad:
                v_shp = v.shape
                pad_shp = (zero_pad, *v_shp[1:]) if len(v_shp)>1 else (zero_pad, )
                #print(pad_shp)
                pad  = np.zeros(pad_shp)
                batch[k] = np.concatenate([pad,
                                          batch[k]], axis = 0) if k in batch else pad
        return batch

def total_sampled_timesteps(worker):
    return worker.policy_map[DEFAULT_POLICY_ID].global_timestep


class DreamerIteration:
    def __init__(self, worker, episode_buffer, dreamer_train_iters, batch_size,
                 act_repeat, smoothing=0.002):
        self.worker = worker
        self.episode_buffer = episode_buffer
        self.dreamer_train_iters = dreamer_train_iters
        self.repeat = act_repeat
        self.batch_size = batch_size
        self.smoothing = smoothing

    def __call__(self, samples):
        #print(samples.keys)
        # Dreamer Training Loop
        for n in range(self.dreamer_train_iters):
            print(n)
            ep_t_ids , batch = self.episode_buffer.sample(self.batch_size)
            if n == self.dreamer_train_iters - 1:
                batch["log_gif"] = True
            fetches = self.worker.learn_on_batch(batch)
            updated_mems=fetches['default_policy']["updated_mems"][:, -self.episode_buffer.length:, ...]
            if updated_mems is not None:
                for i, (ep, index) in enumerate(ep_t_ids):
                    prev_mems = self.episode_buffer.episodes[ep]["mems"][index: index + self.episode_buffer.length]
                    self.episode_buffer.episodes[ep]["mems"][index: index+self.episode_buffer.length] = \
                    self.smoothing*updated_mems[i] + (1-self.smoothing)*prev_mems
        # Custom Logging
        policy_fetches = self.policy_stats(fetches)
        if "log_gif" in policy_fetches:
            gif = policy_fetches["log_gif"]
            policy_fetches["log_gif"] = self.postprocess_gif(gif)

        # Metrics Calculation
        metrics = _get_shared_metrics()
        metrics.info[LEARNER_INFO] = fetches
        metrics.counters[STEPS_SAMPLED_COUNTER] = self.episode_buffer.timesteps
        metrics.counters[STEPS_SAMPLED_COUNTER] *= self.repeat
        res = collect_metrics(local_worker=self.worker)
        res["info"] = metrics.info
        res["info"].update(metrics.counters)
        res["timesteps_total"] = metrics.counters[STEPS_SAMPLED_COUNTER]

        self.episode_buffer.add(samples)
        return res

    def postprocess_gif(self, gif: np.ndarray):
        gif = np.clip(255 * gif, 0, 255).astype(np.uint8)
        B, T, C, H, W = gif.shape
        frames = gif.transpose((1, 2, 3, 0, 4)).reshape((1, T, C, H, B * W))
        return frames

    def policy_stats(self, fetches):
        return fetches["default_policy"]["learner_stats"]


def execution_plan(workers, config):
    # Special Replay Buffer for Dreamer agent
    episode_buffer = EpisodicBuffer(length=config["batch_length"],
                                    ext_context=config["dreamer_model"]["ext_context"],
                                    memory_tau=config["dreamer_model"]["memory_tau"]
                                    ) # add the batch sizes to remove the last

    local_worker = workers.local_worker()

    # Prefill episode buffer with initial exploration (uniform sampling)
    while total_sampled_timesteps(local_worker) < config["prefill_timesteps"]:
        samples = local_worker.sample()
        episode_buffer.add(samples)

    batch_size = config["batch_size"]
    dreamer_train_iters = config["dreamer_train_iters"]
    act_repeat = config["action_repeat"]

    rollouts = ParallelRollouts(workers)
    rollouts = rollouts.for_each(
        DreamerIteration(local_worker, episode_buffer, dreamer_train_iters,
                         batch_size, act_repeat))
    return rollouts


def get_policy_class(config):
    return DreamerTorchPolicy


def validate_config(config):
    config["action_repeat"] = config["env_config"]["frame_skip"]
    if config["framework"] != "torch":
        raise ValueError("Dreamer not supported in Tensorflow yet!")
    if config["batch_mode"] != "complete_episodes":
        raise ValueError("truncate_episodes not supported yet!")
    if config["num_workers"] != 0:
        raise ValueError("Distributed Dreamer not supported yet!")
    if config["clip_actions"]:
        raise ValueError("Clipping is done inherently via policy tanh!")
    if config["action_repeat"] > 1:
        config["horizon"] = config["horizon"] / config["action_repeat"]


DREAMERTrainer = build_trainer(
    name="Dreamer",
    default_config=DEFAULT_CONFIG,
    default_policy=DreamerTorchPolicy,
    get_policy_class=get_policy_class,
    execution_plan=execution_plan,
    validate_config=validate_config)
