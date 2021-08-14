from ray.rllib.env.dm_control_wrapper import DMCEnv
from ray.rllib.env.randomized_dm_control import RDMCEnv, ARDMCEnv
"""
8 Environments from Deepmind Control Suite
"""


def acrobot_swingup(from_pixels=True,
                    height=64,
                    width=64,
                    frame_skip=2,
                    channels_first=True):
    return DMCEnv(
        "acrobot",
        "swingup",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def walker_walk(from_pixels=True,
                height=64,
                width=64,
                frame_skip=2,
                channels_first=True):
    return DMCEnv(
        "walker",
        "walk",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def hopper_hop(from_pixels=True,
               height=64,
               width=64,
               frame_skip=2,
               channels_first=True):
    return DMCEnv(
        "hopper",
        "hop",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def hopper_stand(from_pixels=True,
                 height=64,
                 width=64,
                 frame_skip=2,
                 channels_first=True):
    return DMCEnv(
        "hopper",
        "stand",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def cheetah_run(from_pixels=True,
                height=64,
                width=64,
                frame_skip=2,
                channels_first=True):
    return DMCEnv(
        "cheetah",
        "run",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def walker_run(from_pixels=True,
               height=64,
               width=64,
               frame_skip=2,
               channels_first=True):
    return DMCEnv(
        "walker",
        "run",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def pendulum_swingup(from_pixels=True,
                     height=64,
                     width=64,
                     frame_skip=2,
                     channels_first=True):
    return DMCEnv(
        "pendulum",
        "swingup",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def cartpole_swingup(from_pixels=True,
                     height=64,
                     width=64,
                     frame_skip=2,
                     channels_first=True):
    return DMCEnv(
        "cartpole",
        "swingup",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def humanoid_walk(from_pixels=True,
                  height=64,
                  width=64,
                  frame_skip=2,
                  channels_first=True):
    return DMCEnv(
        "humanoid",
        "walk",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


#Random environments
def random_acrobot_swingup(interval='-0.25 3.25 0.25',
                    from_pixels=True,
                    height=64,
                    width=64,
                    frame_skip=2,
                    channels_first=True):
    return RDMCEnv(
        interval,
        "acrobot",
        "swingup",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def random_walker_walk(interval='-0.25 3.25 0.25',
                from_pixels=True,
                height=64,
                width=64,
                frame_skip=2,
                channels_first=True):
    return RDMCEnv(
        interval,
        "walker",
        "walk",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def random_hopper_hop(interval='-0.25 3.25 0.25',
               from_pixels=True,
               height=64,
               width=64,
               frame_skip=2,
               channels_first=True):
    return RDMCEnv(
        interval,
        "hopper",
        "hop",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def random_hopper_stand(interval='-0.25 3.25 0.25',
                 from_pixels=True,
                 height=64,
                 width=64,
                 frame_skip=2,
                 channels_first=True):
    return RDMCEnv(
        interval,
        "hopper",
        "stand",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def random_cheetah_run(interval='-0.25 3.25 0.25',
                from_pixels=True,
                height=64,
                width=64,
                frame_skip=2,
                channels_first=True):
    return RDMCEnv(
        interval,
        "cheetah",
        "run",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def random_walker_run(interval='-0.25 3.25 0.25',
                from_pixels=True,
               height=64,
               width=64,
               frame_skip=2,
               channels_first=True):
    return RDMCEnv(
        interval,
        "walker",
        "run",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def random_pendulum_swingup(interval='-0.25 3.25 0.25',
                     from_pixels=True,
                     height=64,
                     width=64,
                     frame_skip=2,
                     channels_first=True):
    return RDMCEnv(
        interval,
        "pendulum",
        "swingup",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def random_cartpole_swingup(interval='-0.25 3.25 0.25',
                     from_pixels=True,
                     height=64,
                     width=64,
                     frame_skip=2,
                     channels_first=True):
    return RDMCEnv(
        interval,
        "cartpole",
        "swingup",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def random_humanoid_walk(interval='-0.25 3.25 0.25',
                         from_pixels=True,
                  height=64,
                  width=64,
                  frame_skip=2,
                  channels_first=True):
    return RDMCEnv(
        interval,
        "humanoid",
        "walk",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def random_multi_dmc(tasks= ['walker_walk', 'cheetah_run'],
                  interval='-0.75 2.25 0.25',
                  from_pixels=True,
                  height=64,
                  width=64,
                  frame_skip=2,
                  channels_first=True):
    return ARDMCEnv(
        interval,
        tasks,
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)