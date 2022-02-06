import numpy as np
from typing import Any, List, Tuple, Optional, Union
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.framework import TensorType
# from ray.rllib.models.torch.misc import SlimFC
# from ray.rllib.models.torch.modules import GRUGate, SkipConnection
# from ray.rllib.agents.dreamer.attention_modules import RelativeMultiHeadAttention
from ray.rllib.agents.dreamer.attention_modules import GTrXLNet
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override


torch, nn = try_import_torch()
if torch:
    from torch import distributions as td
    from ray.rllib.agents.dreamer.utils import Linear, Conv2d, \
        ConvTranspose2d, TanhBijector
ActFunc = Any



class Reshape(nn.Module):
    """Standard module that reshapes/views a tensor
"""

    def __init__(self, shape: List):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class EMA:  # BatchTime:
    '''
    This class is mainly targeting EMA of losses in batched temporal data
    '''

    def __init__(self, decay: float = 0.9):
        self.decay = decay
        self.b_k = torch.Tensor([float('nan')])

    def update(self, new_value):
        """
        Args:
            new_value (TensorType): (batch_size, time, d1, ..., dn)
        """
        new_value_c = new_value.detach().clone()

        with torch.no_grad():
            # if len(new_value.size()) > 1:
            #    new_value_c=new_value_c.mean(tuple(range(2, len(new_value.size()))))
            # print(new_value_c)
            new_value_c = new_value_c.mean()
            if torch.isnan(self.b_k):
                self.b_k = new_value_c
                return self.b_k
            self.b_k = self.decay * self.b_k + (1 - self.decay) * new_value_c

        return self.b_k

    def reset_values(self, new_value):
        self.b_k = torch.Tensor([0.])


# Encoder, part of PlaNET
class ConvEncoder(nn.Module):
    """Standard Convolutional Encoder for Dreamer. This encoder is used
    to encode images frm an enviornment into a latent state for the
    RSSM model in PlaNET.
    """

    def __init__(self,
                 depth: int = 32,
                 act: ActFunc = None,
                 shape: List = [3, 64, 64]):
        """Initializes Conv Encoder

      Args:
        depth (int): Number of channels in the first conv layer
        act (Any): Activation for Encoder, default ReLU
        shape (List): Shape of observation input
      """
        super().__init__()
        self.act = act
        if not act:
            self.act = nn.ReLU #Relu
        self.depth = depth
        self.shape = shape

        init_channels = self.shape[0]
        self.layers = [
            Conv2d(init_channels, self.depth, 4, stride=2),
            self.act(),
            Conv2d(self.depth, 2 * self.depth, 4, stride=2),
            self.act(),
            Conv2d(2 * self.depth, 4 * self.depth, 4, stride=2),
            self.act(),
            Conv2d(4 * self.depth, 8 * self.depth, 4, stride=2),
            self.act(),
        ]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        # Flatten to [batch*horizon, 3, 64, 64] in loss function
        orig_shape = list(x.size())
        x = x.view(-1, *(orig_shape[-3:]))
        x = self.model(x)

        new_shape = orig_shape[:-3] + [32 * self.depth]
        x = x.view(*new_shape)
        return x


# Decoder, part of PlaNET
class ConvDecoder(nn.Module):
    """Standard Convolutional Decoder for Dreamer.
    This decoder is used to decode images from the latent state generated
    by the transition dynamics model. This is used in calculating loss and
    logging gifs for imagined trajectories.
    """

    def __init__(self,
                 input_size: int,
                 depth: int = 32,
                 act: ActFunc = None,
                 shape: List[int] = [3, 64, 64]):
        """Initializes a ConvDecoder instance.
        Args:
            input_size (int): Input size, usually feature size output from
                RSSM.
            depth (int): Number of channels in the first conv layer
            act (Any): Activation for Encoder, default ReLU
            shape (List): Shape of observation input
        """
        super().__init__()
        self.act = act
        if not act:
            self.act = nn.ReLU
        self.depth = depth
        self.shape = shape

        self.layers = [
            Linear(input_size, 32 * self.depth),
            Reshape([-1, 32 * self.depth, 1, 1]),
            ConvTranspose2d(32 * self.depth, 4 * self.depth, 5, stride=2),
            self.act(),
            ConvTranspose2d(4 * self.depth, 2 * self.depth, 5, stride=2),
            self.act(),
            ConvTranspose2d(2 * self.depth, self.depth, 6, stride=2),
            self.act(),
            ConvTranspose2d(self.depth, self.shape[0], 6, stride=2),
        ]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        # x is [batch, hor_length, input_size]
        orig_shape = list(x.size())
        x = self.model(x)

        reshape_size = orig_shape[:-1] + self.shape
        mean = x.view(*reshape_size)

        # Equivalent to making a multivariate diag
        return td.Independent(td.Normal(mean, 1), len(self.shape))


# Reward Model (PlaNET), and Value Function
class DenseDecoder(nn.Module):
    """FC network that outputs a distribution for calculating log_prob.
    Used later in DreamerLoss.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 layers: int,
                 units: int,
                 dist: str = "normal",
                 act: ActFunc = None):
        """Initializes FC network
        Args:
            input_size (int): Input size to network
            output_size (int): Output size to network
            layers (int): Number of layers in network
            units (int): Size of the hidden layers
            dist (str): Output distribution, parameterized by FC output
                logits.
            act (Any): Activation function
        """
        super().__init__()
        self.layrs = layers
        self.units = units
        self.act = act
        if not act:
            self.act = nn.ELU
        self.dist = dist
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        cur_size = input_size
        for _ in range(self.layrs):
            self.layers.extend([Linear(cur_size, self.units), self.act()])
            cur_size = units
        self.layers.append(Linear(cur_size, output_size))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.model(x)
        if self.output_size == 1:
            x = torch.squeeze(x)
        if self.dist == "normal":
            output_dist = td.Normal(x, 1)
        elif self.dist == "binary":
            output_dist = td.Bernoulli(logits=x)
        else:
            raise NotImplementedError("Distribution type not implemented!")
        return td.Independent(output_dist, 0)


# Represents dreamer policy
class ActionDecoder(nn.Module):
    """ActionDecoder is the policy module in Dreamer.
    It outputs a distribution parameterized by mean and std, later to be
    transformed by a custom TanhBijector in utils.py for Dreamer.
    """

    def __init__(self,
                 input_size: int,
                 action_size: int,
                 layers: int,
                 units: int,
                 dist: str = "tanh_normal",
                 act: ActFunc = None,
                 min_std: float = 1e-4,
                 init_std: float = 5.0,
                 mean_scale: float = 5.0):
        """Initializes Policy
        Args:
            input_size (int): Input size to network
            action_size (int): Action space size
            layers (int): Number of layers in network
            units (int): Size of the hidden layers
            dist (str): Output distribution, with tanh_normal implemented
            act (Any): Activation function
            min_std (float): Minimum std for output distribution
            init_std (float): Intitial std
            mean_scale (float): Augmenting mean output from FC network
        """
        super().__init__()
        self.layrs = layers
        self.units = units
        self.dist = dist
        self.act = act
        if not act:
            self.act = nn.ReLU
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.action_size = action_size

        self.layers = []
        self.softplus = nn.Softplus()

        # MLP Construction
        cur_size = input_size
        for _ in range(self.layrs):
            self.layers.extend([Linear(cur_size, self.units), self.act()])
            cur_size = self.units
        if self.dist == "tanh_normal":
            self.layers.append(Linear(cur_size, 2 * action_size))
        elif self.dist == "onehot":
            self.layers.append(Linear(cur_size, action_size))
        self.model = nn.Sequential(*self.layers)

    # Returns distribution
    def forward(self, x):
        raw_init_std = np.log(np.exp(self.init_std) - 1)
        x = self.model(x)
        if self.dist == "tanh_normal":
            mean, std = torch.chunk(x, 2, dim=-1)
            mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
            std = self.softplus(std + raw_init_std) + self.min_std
            dist = td.Normal(mean, std)
            transforms = [TanhBijector()]
            dist = td.transformed_distribution.TransformedDistribution(
                dist, transforms)
            dist = td.Independent(dist, 1)
        elif self.dist == "onehot":
            dist = td.OneHotCategorical(logits=x)
            raise NotImplementedError("Atari not implemented yet!")
        return dist


class HyperTranCell(nn.Module):
    """
    For HyperGRU the smaller network and the larger network is a GRU and the hyper network is a LSTM.
    """

    def __init__(self,
                 action_size: int,
                 deter_size: int,
                 stoch_size: int,
                 n_z: int = 12,
                 ext_context: int = 4,
                 hidden_size: int = 128,
                 add_mask: bool = True,
                 w_cng_reg: bool = True):
        """
        Args:
        input_size (int): Hidden size or $f_theta(s_{t-1}, a_{t-1})$
        hidden_size (int): deter size of the base GRU
        hyper_size (int): size of the smaller LSTM that alters the weights of the larger outer GRU.
        n_z int(int): size of the feature vectors used to alter the GRU weights.
        """

        super().__init__()
        self.add_mask = add_mask
        self.w_cng_reg = w_cng_reg

        # self.time_context = time_context
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.n_z = n_z
        self.hidden_size = hidden_size
        self.t_p_ext_context = ext_context + 1

        # combined z_hx
        ads_size = self.action_size + self.deter_size + self.stoch_size
        # print(ads_size, self.time_context)
        self.z_ads = nn.Sequential(
            nn.Linear(self.t_p_ext_context * ads_size, self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, self.n_z),
        )

        self.offset_layer = nn.Linear(self.t_p_ext_context * ads_size, 1, bias=False)  #, self.deter_size, bias=False) # bias was True
        w_m_ads = [torch.zeros(self.deter_size, ads_size, n_z) for _ in
                   range(2 if self.add_mask else 1)]  # self.t_p_ext_context*ads_size
        w_m_ads = tuple([nn.init.xavier_normal_(w) for w in w_m_ads])  # shouldn't be like that orthogonal_ xavier_normal_
        self.w_m_ads = nn.ParameterList([nn.Parameter(torch.cat(w_m_ads))])
        self.prev_w = None

    def forward(self,
                s: torch.Tensor,
                a: torch.Tensor,
                d: torch.Tensor,
                ):
        # print('debugging in hypercell x and h shapes are ', x.shape, h.shape)
        # TODO(karim): fix this
        a_context = a  # [:, -self.time_context:]
        d_context = d  # [:, -self.time_context:]
        s_context = s  # [:, -self.time_context:]
        ads_context_wt = torch.cat((a_context, d_context, s_context), dim=-1)
        ads_context = ads_context_wt.view(-1, np.prod(ads_context_wt.size()[1:]))

        z_ads = self.z_ads(
            ads_context)  # contains mask, and every feature. just chunk for the masks (last two dimensions)
        offset = self.offset_layer(ads_context)
        all_weights = torch.einsum('ijk,bk->bij', self.w_m_ads[0], z_ads)/np.sqrt(z_ads.size(1))
        # split weights ru for x and h, o for x and h, mask_weight x and h
        if self.add_mask:
            o_weights, m_weights = torch.split(all_weights,
                                               (self.deter_size, self.deter_size),
                                               dim=1)
            m_w = torch.sigmoid(m_weights)
            m_w_dist = td.Bernoulli(m_w)
            m_w_dist = td.Independent(m_w_dist, 2)
            m_w_sample = m_w_dist.sample()
        else:
            o_weights = all_weights

        curr_w = o_weights * m_w_sample if self.add_mask else o_weights
        # print(torch.einsum('bij,bj->bi',  curr_w, ads_context).size(), d_context[:, -1, :] .size(), offset.size())


        d_next =  torch.sigmoid(offset)*torch.einsum('bij,bj->bi',
                                      curr_w, ads_context_wt[:, -1, :])/np.sqrt(ads_context_wt[:, -1, :].size(1)) \
                                        + d_context[:, -1, :]
        mask_o = (m_w_dist, m_w_sample) if self.add_mask else (None, None)
        weight_change = torch.mean(
            torch.norm(self.prev_w - curr_w, dim=(1, 2))) if self.w_cng_reg and self.prev_w is not None else None
        self.prev_w = curr_w
        output = d_next, z_ads, *mask_o, weight_change
        return output

    def reset_cell(self):
        self.prev_w = None


class TSSM(nn.Module):
    """TSSM is the core recurrent part of the Hyper dreamer module. It consists of
    two networks, one (obs) to calculate posterior beliefs and states and
    the second (img) to calculate prior beliefs and states. The prior network
    takes in the previous state and action, while the posterior network takes
    in the previous state, action, and a latent embedding of the most recent
    observation.
    """

    def __init__(self,
                 action_size: int,
                 stoch_size: int = 30,
                 deter_size: int = 200,
                 hidden_size: int = 200,
                 n_z: int = 12,
                 ext_context: int = 4,
                 add_mask: bool = False,
                 w_cng_reg: bool = False,
                 act: ActFunc = None):
        """Initializes RSSM
        Args:
            action_size (int): Action space size
            stoch_size (int): Size of the distributional hidden state
            deter_size (int): Size of the deterministic hidden state
            hidden_size (int): General size of hidden layers
            n_z (int): embedding size of the Hypernet
            ext_context (int): extended timesteps to add for dynamics
            add_mask (bool): whether to add a sparsity mask or not
            w_cng_reg (bool): penalizer on weight changes
            act (Any): Activation function
        """
        super().__init__()
        self.act = act
        self.n_z = n_z

        self.add_mask = add_mask
        self.w_cng_reg = w_cng_reg

        self.ext_context = ext_context
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.n_z = n_z
        self.hidden_size = hidden_size

        if act is None:
            self.act = nn.ELU

        self.obs1 = Linear(2 * deter_size, hidden_size)
        self.obs2 = Linear(hidden_size, 2 * stoch_size)
        self.cell = HyperTranCell(action_size=self.action_size,
                                  deter_size=self.deter_size,
                                  stoch_size=self.stoch_size,
                                  n_z=self.n_z,
                                  ext_context=self.ext_context,
                                  hidden_size=self.hidden_size,
                                  add_mask=self.add_mask,
                                  w_cng_reg=self.w_cng_reg)

        self.img1 = Linear(deter_size, hidden_size)
        self.img2 = Linear(hidden_size, 2 * stoch_size)

        self.softplus = nn.Softplus

        self.device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))

    def get_initial_sstates(self, batch_size: int) -> List[TensorType]:
        """Returns the stoch state for the TSSM, which consists of mean,
        std for the stochastic state, the sampled stochastic hidden state
        (from mean, std) for a selected ext_context + 1
        Args:
            batch_size (int): Batch size for initial state
        Returns:
            List of tensors
        """
        return [
            torch.zeros(batch_size, 1 + self.ext_context,
                        self.stoch_size).to(self.device),
            torch.zeros(batch_size, 1 + self.ext_context,
                        self.stoch_size).to(self.device),
            torch.zeros(batch_size, 1 + self.ext_context,
                        self.stoch_size).to(self.device),
        ]

    def get_initial_dstates(self, batch_size: int) -> List[TensorType]:
        """the initial deterministic states for a selected ext_context.
        Args:
            batch_size (int): Batch size for initial state
        Returns:
            tensor
        """
        return torch.zeros(batch_size, 1 + self.ext_context,
                           self.deter_size).to(self.device)

    def observe(self,
                # embeds: TensorType,
                actions: TensorType,
                dstates: TensorType,
                sstates: List[TensorType] = None,
                ) -> Tuple[List[TensorType], List[TensorType], TensorType,
                           TensorType, List[TensorType], TensorType]:
        """Returns the corresponding states from the embedding from ConvEncoder
        and actions. This is accomplished by rolling out the HyperTran from
        the starting state through each index of embed, dstatates and action,
        saving all intermediate states between. The functions also returns
        the predicted dstates from the estimated dynamics in hypertran, which
        are incentivized to be similar to dstates from the transformer.
        Args:
            actions (TensorType): Actions (T+ext_context) from episodic buffer
            dstates (TensorType): from the transformer (T+ext_context) from
            episodic buffer. (acts as a spatiotemporal embeddings of current obs)
            sstates (List[TensorType]): Initial state (ext_context+1) before
            rollout

        Returns:
            Posterior sstates, prior sstates, pred dstates,
            masks, weight changes (sstates and mask are List[TensorType]
            others are tensors)
        """
        act_size = actions.size()  # B, T + ext_context, act_dim
        self.cell.reset_cell()  # reset cell pre observe

        if sstates is None:
            sstates = self.get_initial_sstates(act_size[0])

        priors_w_inits = sstates
        posts_w_inits = sstates
        weight_changes = [] if self.w_cng_reg else None

        if self.add_mask:
            masks = [[] for _ in range(2)]  # [ logp_w, entropy_w]

        pred_dstates = []
        d_embeds = []
        last = (sstates, sstates)

        next_dstates = dstates[:, self.ext_context:]
        # to train good HC bias
        init_dstate = torch.zeros(act_size[0], 1, self.deter_size).to(self.device)
        #print(f'init_dstate {init_dstate.size()} and dstates size is {dstates.size()}')
        prev_dstates = torch.cat((init_dstate, dstates), dim=1)[:, :-1, ...]

        for index in range(act_size[1] - self.ext_context):
            # Tuple of post and prior
            last_acts = actions[:, index:index + self.ext_context + 1, ...]
            last_dstates = prev_dstates[:, index:index + self.ext_context + 1, ...]

            last_post, last_prior, last_d, d_embed, mask, weight_change = \
                self.obs_step(last[0], last_acts, last_dstates, next_dstates[:, index, ...])
            priors_w_inits = [torch.cat((s, o[:, None, ...]), dim=1)
                              for s, o in zip(priors_w_inits, last_prior)]
            posts_w_inits = [torch.cat((s, o[:, None, ...]), dim=1)
                             for s, o in zip(posts_w_inits, last_post)]

            last_priors = [s[:, -self.ext_context - 1:, ...] for s in priors_w_inits]
            last_posts = [s[:, -self.ext_context - 1:, ...] for s in posts_w_inits]
            last = (last_priors, last_posts)

            if self.add_mask:
                [o.append(s) for s, o in zip(mask, masks)]

            pred_dstates.append(last_d)
            d_embeds.append(d_embed)

            if self.w_cng_reg:
                weight_changes.append(weight_change)

        # remove inits from priors and posts
        priors = [x[:, self.ext_context + 1:] for x in priors_w_inits]
        posts = [x[:, self.ext_context + 1:] for x in posts_w_inits]
        pred_dstates = torch.stack(pred_dstates, dim=1)
        d_embeds = torch.stack(d_embeds, dim=1)
        assert pred_dstates.size() == dstates[:, self.ext_context:].size()

        if self.w_cng_reg:
            weight_changes = torch.stack(weight_changes[1:], dim=0)
            weight_changes = torch.mean(weight_changes)

        if self.add_mask:
            mask = [torch.stack(x, dim=1) for x in masks]
        else:
            mask = [None for _ in range(2)]

        return posts, priors, pred_dstates, d_embeds, mask, weight_changes

    def imagine(self, actions: TensorType,
                sstates: List[TensorType] = None,
                dstates: TensorType = None) \
        -> Tuple[List[TensorType], List[TensorType], List[TensorType]]:
        """Imagines the trajectory starting from state through a list of actions.
        Similar to observe(), requires rolling out the RNN for each timestep.
        Args:
            actions (TensorType): Actions (T+ext_context)
            sstates (List[TensorType]): Starting ext_context+1 sstate before rollout
            dstates (TensorType): starting ext_context+1 dstates before rollout
        Returns:
            Prior states, hyper_states masks
        """
        act_size = actions.size()
        self.cell.reset_cell()
        if sstates is None:
            sstates = self.get_initial_sstates(act_size[0])

        if dstates is None:
            dstates = self.get_initial_dstates(act_size[0])

        priors_w_inits = sstates
        dstates_w_inits = dstates

        last_priors = sstates
        last_dstates = dstates
        d_embeds = []

        for index in range(act_size[1] - self.ext_context):
            last_acts = actions[:, index:index + self.ext_context + 1, ...]
            last_prior, last_d, d_embed, _, _ = self.img_step(last_priors,
                                                              last_acts, last_dstates)

            priors_w_inits = [torch.cat((s, o[:, None, ...]), dim=1)
                              for s, o in zip(priors_w_inits, last_prior)]
            dstates_w_inits = torch.cat((dstates_w_inits, last_d[:, None, ...])
                                        , dim=1)

            # should contain 1+ext context elements
            last_priors = [s[:, -self.ext_context - 1:, ...] for s in priors_w_inits]
            last_dstates = dstates_w_inits[:, -self.ext_context - 1:, ...]
            d_embeds.append(d_embed)

        d_embeds = torch.stack(d_embeds, dim=1)
        priors = [x[:, self.ext_context + 1:] for x in priors_w_inits]
        dstates = dstates_w_inits[:, self.ext_context + 1:]

        return priors, dstates, d_embeds

    def obs_step(
        self,
        prev_sstates: List[TensorType],
        prev_acts: TensorType,
        prev_dstates: TensorType,
        next_dstate: TensorType) \
        -> Tuple[List[TensorType], List[TensorType],
                 TensorType, TensorType, List[TensorType], TensorType]:
        """Runs through the posterior model and returns the posterior state
        Args:
            prev_sstates ((List[TensorType]): The previous ext_context+1 state (mu, sigma, sample)
            prev_acts (TensorType): The previous ext_context+1 actions
            prev_dstates (TensorType): the previous ext_context+1 dstates
            nex_dstate (TensorType): the next dstate as a spatio temporal embedding
            embed (TensorType): Embedding from ConvEncoder
        Returns:
            Post, Prior, Hyper state, Mask
      """
        prior, next_d, d_embed, mask, weight_change = self.img_step(prev_sstates,
                                                                    prev_acts, prev_dstates)
        # here we have two options either using next_d from im_step or from dstates
        x = torch.cat([next_d, next_dstate], dim=-1)  # highly correlated
        x = self.obs1(x)
        x = self.act()(x)
        x = self.obs2(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = self.softplus()(std) + 0.1
        stoch = self.get_dist(mean, std).rsample()
        post = [mean, std, stoch]
        return post, prior, next_d, d_embed, mask, weight_change

    def img_step(self, prev_sstates: List[TensorType],
                 prev_acts: TensorType,
                 prev_dstates: TensorType
                 ) \
        -> Tuple[List[TensorType], TensorType, TensorType, List[TensorType], TensorType]:
        """Runs through the prior model and returns the prior state and next
        diff
        Args:
            prev_sstates ((List[TensorType]): The previous stoch.
            states (mu, sigma, sample) 1+ext_context
            prev_acts (TensorType): The previous 1+ext_context action
            prev_dstates (TensorType): The previous 1+ext_context deter. states
        Returns:
            Prior state
        """
        # with at least 1 in the time dimension for prev s.a.d.
        next_d, d_embed, m_w_dist, m_w_sample, weight_change = \
            self.cell(prev_sstates[2], prev_acts, prev_dstates)

        x = next_d
        x = self.img1(x)
        x = self.act()(x)
        x = self.img2(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = self.softplus()(std) + 0.1
        stoch = self.get_dist(mean, std).rsample()
        if self.add_mask:
            m_w_logp, m_w_entropy = m_w_dist.log_prob(m_w_sample), m_w_dist.entropy()
        else:
            m_w_logp, m_w_entropy = None, None
        return [mean, std, stoch], \
               next_d, \
               d_embed, \
               [m_w_logp, m_w_entropy], \
               weight_change

    def get_feature(self, sample_sstate,
                    dstate, d_embed: TensorType = None) -> TensorType:
        # Constructs feature for input to reward, decoder, actor, critic
        if d_embed is not None:
            return torch.cat([sample_sstate, dstate, d_embed], dim=-1)
        else:
            return torch.cat([sample_sstate, dstate], dim=-1)

    def get_dist(self, mean: TensorType, std: TensorType) -> TensorType:
        return td.Normal(mean, std)


# Represents all models in Dreamer, unifies them all into a single interface
class DreamerModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        nn.Module.__init__(self)
        self.depth = model_config["depth_size"]
        self.deter_size = model_config["deter_size"]
        self.stoch_size = model_config["stoch_size"]
        self.hidden_size = model_config["hidden_size"]
        self.n_z = model_config['n_z']
        self.decay = model_config['decay']
        self.dembed_in_state = model_config['dembed_in_state']
        self.add_mask = model_config['add_mask']
        self.w_cng_reg = model_config['w_cng_reg']
        self.ext_context = model_config['ext_context']
        self.num_transformer_units = model_config['num_transformer_units']
        self.num_heads = model_config['num_heads']
        self.atten_size = model_config['atten_size']
        self.memory_tau = model_config['memory_tau']

        self.action_size = action_space.shape[0]
        dembed_add_dim = self.n_z if self.dembed_in_state else 0

        self.device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))

        self.encoder = ConvEncoder(self.depth).to(self.device)
        self.decoder = ConvDecoder(
            self.stoch_size + self.deter_size + dembed_add_dim,
            depth=self.depth).to(self.device)
        self.reward = DenseDecoder(
            self.stoch_size + self.deter_size + dembed_add_dim,
            1, 2,
            self.hidden_size).to(self.device)

        self.dynamics = TSSM(self.action_size,
                             stoch_size=self.stoch_size,
                             deter_size=self.deter_size,
                             hidden_size=self.hidden_size,
                             n_z=self.n_z,
                             ext_context=self.ext_context,
                             add_mask=self.add_mask,
                             w_cng_reg=self.w_cng_reg
                             ).to(self.device)
        self.trans = GTrXLNet(input_dim=32 * self.depth,
                              output_dim=self.deter_size,
                              action_dim=self.action_size,
                              attention_dim=self.atten_size,
                              num_transformer_units=self.num_transformer_units,
                              num_heads=self.num_heads,
                              memory_inference=self.memory_tau,
                              memory_training=self.memory_tau).to(self.device)

        self.actor = ActionDecoder(self.stoch_size + self.deter_size +
                                   dembed_add_dim,
                                   self.action_size, 4, self.hidden_size).to(self.device)
        self.value = DenseDecoder(self.stoch_size + self.deter_size + dembed_add_dim,
                                  1, 3,
                                  self.hidden_size).to(self.device)
        self.state = None

        self.ema = EMA(self.decay)  # EMABatchTime(self.decay)

        print(f'device in model is {self.device}')
        self.updated_mems = None
        self.get_initial_state()
        self.state_temp_dims = [state.size() if state is not None else None for state in self.state]
        self.state = None
        #print(f'shapes of states is {self.state_temp_dims}')

    def context_to_feat(self, states_w_context):
        # add all features to last dim and make the batch major dim
        states_wo_context = []
        for state in states_w_context:
            if len(state.size()) < 4:
                states_wo_context.append(
                    state.view(state.size(0), -1) if state is not None else state)
            else:
                states_wo_context.append(
                    state.view(*state.size()[:2], -1).permute(1, 0, 2) if state is not None else state)
        return states_wo_context

    def feat_to_context(self, states_wo_context):

        states_w_context = []
        for state, sz in zip(states_wo_context, self.state_temp_dims):
            #print(f'pair of state {state}, and its size {sz}')
            # states_w_context.append(state.view(*sz) if state is not None else state)
            if len(sz) <4:
                states_w_context.append(state.view(*sz) if state is not None else state)

            else:
                states_w_context.append(
                    state.permute(1, 0, 2).view(*sz) if state is not None else state)
        return states_w_context

    def policy(self, obs: TensorType, state: List[TensorType], explore=True
               ) -> Tuple[TensorType, List[float], List[TensorType]]:
        """Returns the action. Runs through the encoder, trans, observe step,
        and policy to obtain action. state should lag the obs by 1 step
        """

        #for i, s in enumerate(state):
            #print(f'In policy state {i} dim is: {s.size()})')
        if state is None:
            # with initial ext_context
            self.get_initial_state()
        else:
            #for i, s in enumerate(state):
            #    print(f'In policy state before {i} dim is: {s.size()})')
            self.state = self.feat_to_context(state)
            #for i, s in enumerate(self.state):
            #    print(f'In policy state before {i} dim is: {s.size()})')

        # split states into it's four parts
        last_posts = self.state[:3]
        last_dstates = self.state[3]
        last_actions = self.state[5]
        mems =  self.state[-1]#[-self.num_transformer_units:]
        # The knowledge of the transformer is distilled online to the
        # transition HC model to get meaningful transitions
        embed = self.encoder(obs)
        out, memory_outs = self.trans(embed[:, None, ...], mems)
        self.dynamics.cell.reset_cell()
        last_post, _, last_dstate, d_embed, _, _ = \
            self.dynamics.obs_step(last_posts, last_actions, last_dstates, out[:, 0])

        self.state = self.update_state(last_post +
                                       [last_dstate, d_embed, None, memory_outs])

        last_posts = self.state[:3]
        last_dstates = self.state[3]
        d_embed = self.state[4]


        feat = self.dynamics.get_feature(last_posts[-1][:, -1], last_dstates[:, -1], d_embed[:, -1])
        #print('features to compute actions: ', feat.max(), feat.min())

        action_dist = self.actor(feat)
        if explore:
            action = action_dist.sample()
        else:
            action = action_dist.mean
        logp = action_dist.log_prob(action)
        self.state = self.update_state(3 * [None] + [None, None, action, None])
        #print(action)
        return action, logp, self.context_to_feat(self.state)

    def imagine_ahead(self, sstates: List[TensorType], dstates: TensorType, acts: TensorType, d_embeds: TensorType,
                      horizon: int) -> TensorType:
        """Given a batch of statesrolls out more state of length horizon.

        """
        # extract extended context from each argument
        time_sample = (dstates.size(1) // (self.ext_context + 1)) * (self.ext_context + 1)
        start_prior = []
        for s in sstates:
            s = s[:, :time_sample].contiguous().detach()
            shpe = [-1] + [self.ext_context + 1] + list(s.size())[2:]
            start_prior.append(s.view(*shpe))

        start_dstates = dstates[:, :time_sample].contiguous().detach()
        shpe = [-1] + [self.ext_context + 1] + list(start_dstates.size())[2:]
        start_dstates = start_dstates.view(*shpe)

        start_acts = acts[:, :time_sample].contiguous().detach()
        shpe = [-1] + [self.ext_context + 1] + list(start_acts.size())[2:]
        start_acts = start_acts.view(*shpe)

        if self.dembed_in_state:
            start_d_embeds = d_embeds[:, :time_sample].contiguous().detach()
            shpe = [-1] + [self.ext_context + 1] + list(start_d_embeds.size())[2:]
            d_embed = start_d_embeds.view(*shpe)[:, 0, ...]
        else:
            d_embed = None

        def next_state(sstate, dstate, d_embed, prev_acts):
            # All parameters except the d_embed are with the ext_context so it must be cleared from the get_feature
            feature = self.dynamics.get_feature(sstate[-1][:, -1], dstate[:, -1],
                                                d_embed).detach()  # added
            action = self.actor(feature).rsample()
            prev_acts = torch.cat((prev_acts, action[:, None, ...]), dim=1)
            prev_acts = prev_acts[:, 1:, ...]
            self.dynamics.cell.reset_cell()
            next_sstate, next_dstate, d_embed, _, _ = self.dynamics.img_step(sstate, prev_acts, dstate)
            d_embed = d_embed if self.dembed_in_state else None
            return next_sstate, next_dstate, d_embed, prev_acts

        o_sstates = start_prior
        o_dstates = start_dstates

        last_priors = start_prior
        last_dstates = start_dstates
        last_acts = start_acts

        d_embeds = [] if self.dembed_in_state else None

        for _ in range(horizon):
            next_sstate, next_dstate, d_embed, last_acts = next_state(last_priors, last_dstates, d_embed, last_acts)

            o_sstates = [torch.cat((s, o[:, None, ...]), dim=1)
                         for s, o in zip(o_sstates, next_sstate)]
            o_dstates = torch.cat((o_dstates, next_dstate[:, None, ...]), dim=1)

            # should contain 1+ext context elements
            last_priors = [s[:, -self.ext_context - 1:, ...] for s in o_sstates]
            last_dstates = o_dstates[:, -self.ext_context - 1:, ...]
            if self.dembed_in_state:
                d_embeds.append(d_embed)

        # in this part the output features should be indexed first with time (T,B, features)
        # the inits are removed which are the first self.ext_context + 1
        d_embeds = torch.stack(d_embeds, dim=0) if self.dembed_in_state else None
        o_sstates = [x[:, self.ext_context + 1:].permute(1, 0, 2) for x in o_sstates]
        o_dstates = o_dstates[:, self.ext_context + 1:].permute(1, 0, 2)
        imag_feat = self.dynamics.get_feature(o_sstates[-1], o_dstates, d_embeds)
        return imag_feat

    def get_initial_mem(self, batch_size: int = None) -> List[TensorType]:

        if self.memory_tau > 0:
            batch_size = batch_size if batch_size else 1
            mems = torch.stack([torch.zeros(batch_size, self.memory_tau, self.atten_size).to(self.device) for _ in
                                range(self.num_transformer_units)])
            #mems = [torch.zeros(batch_size, self.memory_tau, self.atten_size).to(self.device) for _ in
            #        range(self.num_transformer_units)]
        else:
            mems = None #[None for _ in
                    #range(self.num_transformer_units)]
        return mems


    def get_initial_state(self, batch_size: int = None) -> List[TensorType]:
        """Initial state consists of the initial state of the TSSM with d_embed if active, memory and action
         The form of the state is (mu_post, sig_post, sample_post, dstate, d_embed, action, memory)
         all of ext_context+1 in the time axis except for the dembed it is 1 and for the mem it is
         mem tau
        """
        batch_size = batch_size if batch_size else 1
        self.state = self.dynamics.get_initial_sstates(batch_size) + [self.dynamics.get_initial_dstates(batch_size)] + \
                     [torch.zeros(batch_size, 1, self.n_z).to(self.device) if self.dembed_in_state else None] + \
                     [torch.zeros(batch_size, self.ext_context + 1, self.action_space.shape[0]).to(self.device)] + \
                     [self.get_initial_mem(batch_size)] #remove brackets
        return self.context_to_feat(self.state)


    def update_state(self, single_step_state) -> List[TensorType]:
        """updates the state with the previous memory tau for the memory and the previous ext context for other
        state (List[TensorType or List(TensorType)]): list of the new states from obs_step and transformer
        required to add them to the current state and roll it
                  """
        # update memory
        last_state = []
        state_no_mem = single_step_state[:-1]
        mem = single_step_state[-1]

        if mem is None:
            # if there is no update or memory is not used
            # then keep last (either the prev_memory or none)
            last_mems = self.state[-1] #self.state[-self.num_transformer_units:]
        else:
            last_mems = []
            for i, _ in enumerate(mem):
                state_cat = torch.cat((self.state[-1][i], mem[i]), dim=1)
                #state_cat = torch.cat((self.state[-self.num_transformer_units + i], mem[i]), dim=1)
                last_mems.append(state_cat[:, -self.memory_tau:, ...])
            last_mems = torch.stack(last_mems)

        for i, _ in enumerate(state_no_mem):

            if state_no_mem[i] is None:
                # None means no change, so if None, keep it.
                last_state.append(self.state[i])
                continue
            if len(self.state[i].size()) > 2:  # contains an extend context
                state_cat = torch.cat((self.state[i], state_no_mem[i][:, None, ...]), dim=1)
                last_state.append(state_cat[:, - self.state[i].size()[1]:, ...])
            else:
                last_state.append(state_no_mem[i])
        updated_state = last_state + [last_mems]
        return updated_state

    def value_function(self) -> TensorType:
        return None
