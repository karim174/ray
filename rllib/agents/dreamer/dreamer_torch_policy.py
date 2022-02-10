import logging

import ray
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.agents.a3c.a3c_torch_policy import apply_grad_clipping
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.agents.dreamer.utils import FreezeParameters

torch, nn = try_import_torch()
if torch:
    from torch import distributions as td

logger = logging.getLogger(__name__)


# This is the computation graph for workers (inner adaptation steps)
def compute_dreamer_loss(obs,
                         action,
                         reward,
                         init_mems,
                         model,
                         imagine_horizon,
                         discount = 0.99,
                         lambda_= 0.95,
                         kl_coeff = 1.0,
                         ent_coeff = 0.1,
                         rei_coeff = 1.0,
                         wc_coeff = 0.01,
                         matching_coeff = 0.01,
                         kl_scale = 0.8,
                         log=False):
    """Constructs loss for the Dreamer objective
        Args:
            obs (TensorType): Observations (o_t)
            action (TensorType): Actions (a_(t-1))
            reward (TensorType): Rewards (r_(t-1))
            model (TorchModelV2): DreamerModel, encompassing all other models
            imagine_horizon (int): Imagine horizon for actor and critic loss
            discount (float): Discount
            lambda_ (float): Lambda, like in GAE
            kl_coeff (float): KL Coefficient for Divergence loss in model loss
            ent_coeff (float): Mask Entropy Coefficient for Mask Entropy loss in model loss
            rei_coeff (float): Reinforce Coefficient for Reinforce loss in model loss
            wc_coeff (float): Weight change Coefficient for limiting weight change in model loss
            matching_coeff (float): Matching Coefficient for tying transformer and hypertran ouputs
            kl_scale (float): scale for kl balancing
            log (bool): If log, generate gifs
        """
    encoder_weights = list(model.encoder.parameters())
    decoder_weights = list(model.decoder.parameters())
    reward_weights = list(model.reward.parameters())
    dynamics_weights = list(model.dynamics.parameters())
    tran_weights = list(model.trans.parameters())
    critic_weights = list(model.value.parameters())
    model_weights = list(encoder_weights + decoder_weights + reward_weights +
                         dynamics_weights + tran_weights)
    model.updated_mems = None
    model.dynamics.cell.reset_cell()
    device = (torch.device("cuda")
              if torch.cuda.is_available() else torch.device("cpu"))
    # PlaNET Model Loss
    latent = model.encoder(obs)
    #shape got from buffer for init_mems is (batch, mem_tau, layers, dim)
    out, memory_outs = model.trans(latent, init_mems.permute(2, 0, 1, 3))

    posts, priors, pred_dstates, d_embeds, mask, weight_changes = model.dynamics.observe(action, out)
    dstates_trans = out[:, -pred_dstates.size()[1]:, ...]

    with FreezeParameters(tran_weights):
        matching_loss = matching_coeff*torch.norm(dstates_trans - pred_dstates, dim=-1).mean() #(1)

    features = model.dynamics.get_feature(posts[-1], pred_dstates,
                                          d_embeds if model.dembed_in_state else None)  # added features
    features_t = model.dynamics.get_feature(posts[-1], dstates_trans,
                                            d_embeds if model.dembed_in_state else None)  # added
    image_pred = model.decoder(features)
    reward_pred = model.reward(features)
    image_loss = -image_pred.log_prob(obs[:,model.ext_context:])
    reward_loss = -reward_pred.log_prob(reward[:,model.ext_context:])

    image_pred_t = model.decoder(features_t)
    reward_pred_t = model.reward(features_t)
    image_loss_t = -image_pred_t.log_prob(obs[:, model.ext_context:])
    reward_loss_t = -reward_pred_t.log_prob(reward[:, model.ext_context:])

    prior_dist = model.dynamics.get_dist(priors[0], priors[1])
    post_dist = model.dynamics.get_dist(posts[0], posts[1])
    kl_lhs = torch.mean(
        torch.distributions.kl_divergence(post_dist.detach(), prior_dist).sum(dim=2))
    kl_rhs = torch.mean(
        torch.distributions.kl_divergence(post_dist, prior_dist.detach()).sum(dim=2))
    div = kl_scale * kl_lhs + (1-kl_scale)*kl_rhs

    if model.add_mask:
        mask_logp_w, mask_entropy_w = mask
        norm_mask = pred_dstates.size(-1) * (action.size(-1) + pred_dstates.size(-1) + posts[0].size(-1))
        mask_logp_w, mask_entropy_w = mask_logp_w / norm_mask, mask_entropy_w / norm_mask
        with torch.no_grad():
            reinf_reward = -(image_loss + reward_loss)  # weighted by the likelihood
            reinforce_reward = reinf_reward.detach().clone()  # remove the rewards from the graph and clone them
        reinforce_reward_ema = model.ema.update(reinforce_reward)
        mask_logp = mask_logp_w
        mask_reinforce_loss = -torch.mean((reinforce_reward - reinforce_reward_ema) * mask_logp)
        mask_entropy_loss = -torch.mean(mask_entropy_w)

    image_loss = torch.mean(image_loss)
    reward_loss = torch.mean(reward_loss)
    image_loss_t = torch.mean(image_loss_t)
    reward_loss_t = torch.mean(reward_loss_t)

    weight_loss = wc_coeff*weight_changes if weight_changes is not None else 0
    model_loss = kl_coeff * div + reward_loss + image_loss + weight_loss + matching_loss + reward_loss_t + image_loss_t
    if model.add_mask:
        model_loss = model_loss + ent_coeff*mask_entropy_loss + rei_coeff*mask_reinforce_loss


    # [imagine_horizon, batch_length*batch_size, feature_size]
    with torch.no_grad():
        actor_sstates = [v.detach() for v in posts]
        actor_d_embeds = d_embeds.detach()
        actor_pred_dstates = pred_dstates.detach()
        actor_acts = action[:, -pred_dstates.size()[1]:].detach()

    with FreezeParameters(model_weights):
        imag_feat = model.imagine_ahead(actor_sstates, actor_pred_dstates,
                                        actor_acts, actor_d_embeds,
                                        imagine_horizon)
    with FreezeParameters(model_weights + critic_weights):
        reward = model.reward(imag_feat).mean
        value = model.value(imag_feat).mean
    pcont = discount * torch.ones_like(reward)
    returns = lambda_return(reward[:-1], value[:-1], pcont[:-1], value[-1],
                            lambda_)
    discount_shape = pcont[:1].size()
    discount = torch.cumprod(
        torch.cat([torch.ones(*discount_shape).to(device), pcont[:-2]], dim=0),
        dim=0)
    actor_loss = -torch.mean(discount * returns)

    # Critic Loss
    with torch.no_grad():
        val_feat = imag_feat.detach()[:-1]
        target = returns.detach()
        val_discount = discount.detach()
    val_pred = model.value(val_feat)
    critic_loss = -torch.mean(val_discount * val_pred.log_prob(target))

    # Logging purposes
    prior_ent = torch.mean(prior_dist.entropy())
    post_ent = torch.mean(post_dist.entropy())

    log_gif = None
    if log:
        log_gif = log_summary(obs, action, out, image_pred, model)

    model.updated_mems = memory_outs

    return_dict = {
        "model_loss": model_loss,
        "reward_loss": reward_loss,
        "image_loss": image_loss,
        "reward_loss_t": reward_loss_t,
        "image_loss_t": image_loss_t,
        "weight_changes": weight_changes,
        "divergence": div,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "prior_ent": prior_ent,
        "post_ent": post_ent,
        "mask_ent_loss": mask_entropy_loss if model.add_mask else 0,
        "mask_reinforce_loss": mask_reinforce_loss if model.add_mask else 0
    }

    if log_gif is not None:
        return_dict["log_gif"] = log_gif
    return return_dict


# Similar to GAE-Lambda, calculate value targets
def lambda_return(reward, value, pcont, bootstrap, lambda_):
    def agg_fn(x, y):
        return y[0] + y[1] * lambda_ * x

    next_values = torch.cat([value[1:], bootstrap[None]], dim=0)
    inputs = reward + pcont * next_values * (1 - lambda_)

    last = bootstrap
    returns = []
    for i in reversed(range(len(inputs))):
        last = agg_fn(last, [inputs[i], pcont[i]])
        returns.append(last)

    returns = list(reversed(returns))
    returns = torch.stack(returns, dim=0)
    return returns


# Creates gif
def log_summary(obs, action, embed, image_pred, model):
    b_samp = min(6, obs.size(0))
    truth = obs[:b_samp, -image_pred.mean.size()[1]:] + 0.5
    recon = image_pred.mean[:b_samp]
    init, _, init_dstates, _, _, _ = model.dynamics.observe(action[:b_samp, :model.ext_context+5],
                                                            embed[:b_samp, :model.ext_context+5]) #fix

    init = [itm[:, -model.ext_context - 1:] for itm in init]
    init_dstates = init_dstates[:, -model.ext_context - 1:]
    priors, pred_dstates, d_embeds = model.dynamics.imagine(action[:b_samp, 5:], init, init_dstates) # shapes [model.ext_context+5:]
    if model.dembed_in_state:
        feats =model.dynamics.get_feature(priors[-1], pred_dstates,
                                   d_embeds if model.dembed_in_state else None)
        openl = model.decoder(feats).mean

    mod = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)

    error = (mod - truth + 1.0) / 2.0
    return torch.cat([truth, mod, error], 3)


def dreamer_loss(policy, model, dist_class, train_batch):
    log_gif = False
    if "log_gif" in train_batch:
        log_gif = True


    policy.stats_dict = compute_dreamer_loss(
        train_batch["obs"],
        train_batch["actions"],
        train_batch["rewards"],
        train_batch["mems"],
        policy.model,
        policy.config["imagine_horizon"],
        policy.config["discount"],
        policy.config["lambda"],
        policy.config["kl_coeff"],
        policy.config["ent_coeff"],
        policy.config["rei_coeff"],
        policy.config["wc_coeff"],
        policy.config["matching_coeff"],
        policy.config["kl_scale"],
        log_gif,
    )

    loss_dict = policy.stats_dict

    return (loss_dict["model_loss"], loss_dict["actor_loss"],
            loss_dict["critic_loss"])


def build_dreamer_model(policy, obs_space, action_space, config):

    policy.model = ModelCatalog.get_model_v2(
        obs_space,
        action_space,
        1,
        config["dreamer_model"],
        name="DreamerModel",
        framework="torch")
    policy.model.to(policy.model.device)

    policy.model_variables = policy.model.variables()
    policy.model_loss_baseline = 0

    return policy.model


def action_sampler_fn(policy, model, input_dict, state, explore, timestep):
    """Action sampler function has two phases. During the prefill phase,
    actions are sampled uniformly [-1, 1]. During training phase, actions
    are evaluated through DreamerPolicy and an additive gaussian is added
    to incentivize exploration.
    """
    obs = input_dict["obs"]

    # Custom Exploration
    if timestep <= policy.config["prefill_timesteps"]:
        logp = [0.0]
        # Random action in space [-1.0, 1.0]
        action = 2.0 * torch.rand(1, model.action_space.shape[0]) - 1.0
        state = model.get_initial_state()
    else:
        # Weird RLLib Handling, this happens when env rests
        if len(state[0].size()) == 3:
            # Very hacky, but works on all envs
            state = model.get_initial_state()
        action, logp, state = model.policy(obs, state, explore)
        action = td.Normal(action, policy.config["explore_noise"]).sample()
        action = torch.clamp(action, min=-1.0, max=1.0)

    policy.global_timestep += policy.config["action_repeat"]
    return action, logp, state


def dreamer_stats(policy, train_batch):
    return policy.stats_dict


def dreamer_optimizer_fn(policy, config):
    model = policy.model
    encoder_weights = list(model.encoder.parameters())
    decoder_weights = list(model.decoder.parameters())
    reward_weights = list(model.reward.parameters())
    dynamics_weights = list(model.dynamics.parameters())
    actor_weights = list(model.actor.parameters())
    critic_weights = list(model.value.parameters())
    tran_weights = list(model.trans.parameters())
    model_opt = torch.optim.Adam(
        encoder_weights + decoder_weights + reward_weights + tran_weights + dynamics_weights,
        lr=config["td_model_lr"])
    #dyn_opt = torch.optim.Adam(dynamics_weights, lr=0.1*config["td_model_lr"])
    actor_opt = torch.optim.Adam(actor_weights, lr=config["actor_lr"])
    critic_opt = torch.optim.Adam(critic_weights, lr=config["critic_lr"])

    return (model_opt, actor_opt, critic_opt)


DreamerTorchPolicy = build_torch_policy(
    name="DreamerTorchPolicy",
    get_default_config=lambda: ray.rllib.agents.dreamer.dreamer.DEFAULT_CONFIG,
    action_sampler_fn=action_sampler_fn,
    loss_fn=dreamer_loss,
    stats_fn=dreamer_stats,
    make_model=build_dreamer_model,
    optimizer_fn=dreamer_optimizer_fn,
    extra_grad_process_fn=apply_grad_clipping,
    extra_learn_fetches_fn=lambda policy: {"updated_mems": policy.model.updated_mems.permute(1,2,0,3)})
