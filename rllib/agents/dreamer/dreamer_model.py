import numpy as np
from typing import Any, List, Tuple
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.framework import TensorType

torch, nn = try_import_torch()
if torch:
    from torch import distributions as td
    from ray.rllib.agents.dreamer.utils import Linear, Conv2d, \
        ConvTranspose2d, GRUCell, TanhBijector, LSTMCell

ActFunc = Any


class Reshape(nn.Module):
    """Standard module that reshapes/views a tensor
"""

    def __init__(self, shape: List):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class EMABatchTime:
    '''
    This class is mainly targeting EMA of losses in batched temporal data
    '''
    def __init__(self, decay: float = 0.9):

        self.decay = decay
        self.b_k = torch.Tensor([0.])

    def update(self, new_value):
        """
        Args:
            new_value (TensorType): (batch_size, time, d1, ..., dn)
        """
        new_value_c = new_value.detach().clone()

        with torch.no_grad():
            if len(new_value.size()) > 1:
                new_value_c=new_value_c.mean(tuple(range(2, len(new_value.size()))))
                #print(new_value_c)
            self.b_k = self.decay*self.b_k + (1-self.decay)*new_value_c

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
            self.act = nn.ReLU
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


class HyperGRUCell(nn.Module):
    """
    For HyperGRU the smaller network and the larger network is a GRU and the hyper network is a LSTM.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 75,
                 hyper_size: int = 15,
                 n_z: int = 12,
                 simple_rnn: bool = False):
        """
        Args:
        input_size (int): Hidden size or $f_theta(s_{t-1}, a_{t-a})$
        hidden_size (int): deter size of the base GRU
        hyper_size (int): size of the smaller LSTM that alters the weights of the larger outer GRU.
        n_z int(int): size of the feature vectors used to alter the GRU weights.
        """

        super().__init__()

        self.hyper = LSTMCell(hidden_size + input_size, hyper_size)
        self.simple_rnn = simple_rnn

        if simple_rnn:
            # I feel that it's a typo.
            self.z_h = nn.Linear(hyper_size, n_z)
            self.m_h = nn.Linear(hyper_size, n_z)

            self.z_x = nn.Linear(hyper_size, n_z)
            self.m_x = nn.Linear(hyper_size, n_z)

            self.z_b = nn.Linear(hyper_size, n_z, bias=False)
            self.d_b = nn.Linear(n_z, hidden_size)

            #Single parameters are listed (in a ParameterList) to be registered in the model parameters
            self.w_h = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, hidden_size, n_z))])
            self.w_m_h = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, hidden_size, n_z))])
            self.w_x = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, input_size, n_z))])
            self.w_m_x = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, input_size, n_z))])

        else:
            self.z_h = nn.Linear(hyper_size, 3 * n_z)
            self.m_h = nn.Linear(hyper_size, n_z)

            self.z_x = nn.Linear(hyper_size, 3 * n_z)
            self.m_x = nn.Linear(hyper_size, n_z)

            self.z_b = nn.Linear(hyper_size, 3 * n_z, bias=False)
            d_b = [nn.Linear(n_z, hidden_size) for _ in range(3)]
            self.d_b = nn.ModuleList(d_b)

            self.w_h = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, hidden_size, n_z)) for _ in range(3)])
            self.w_m_h = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, hidden_size, n_z))])
            self.w_x = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, input_size, n_z)) for _ in range(3)])
            self.w_m_x = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, input_size, n_z))])

        for param in self.w_h:
            nn.init.orthogonal_(param)
        for param in self.w_m_h:
            nn.init.orthogonal_(param)
        for param in self.w_x:
            nn.init.orthogonal_(param)
        for param in self.w_m_x:
            nn.init.orthogonal_(param)
        # Layer normalization
        #self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(3)])
        #self.layer_norm_c = nn.LayerNorm(hidden_size)

    def forward(self,
                 x: torch.Tensor,
                 h: torch.Tensor,
                 h_hat: torch.Tensor,
                 c_hat: torch.Tensor):
        #print('debugging in hypercell x and h shapes are ', x.shape, h.shape)
        x_hat = torch.cat((h, x), dim=-1)
        h_hat, c_hat = self.hyper(x_hat, [h_hat, c_hat])

        if self.simple_rnn:

            z_h = self.z_h(h_hat)
            m_h = self.m_h(h_hat)
            z_x = self.z_x(h_hat)
            m_x = self.m_x(h_hat)
            z_b = self.z_b(h_hat)
            m_h = torch.einsum('ijk,bk->bij', self.w_m_h[0], m_h)
            m_x = torch.einsum('ijk,bk->bij', self.w_m_x[0], m_x)
            m_h_dist = td.Bernoulli(logits=m_h)
            m_h_dist = td.Independent(m_h_dist, 2)
            m_x_dist = td.Bernoulli(logits=m_x)
            m_x_dist = td.Independent(m_x_dist, 2)
            m_h_sample = m_h_dist.sample()
            m_x_sample = m_x_dist.sample()
            h_next = torch.einsum('bij,bj->bi', m_h_sample * torch.einsum('ijk,bk->bij', self.w_h[0], z_h), h) + \
                torch.einsum('bij,bj->bi',  m_x_sample * torch.einsum('ijk,bk->bij',self.w_x[0], z_x), x) + \
                self.d_b(z_b)
            return h_next, h_hat, c_hat, m_h_dist, m_h_sample, m_x_dist, m_x_sample

        z_h = self.z_h(h_hat).chunk(3, dim=-1)
        m_h = self.m_h(h_hat)
        z_x = self.z_x(h_hat).chunk(3, dim=-1)
        m_x = self.m_x(h_hat)
        z_b = self.z_b(h_hat).chunk(3, dim=-1)

        # We calculate $r$, $u$, and $o$ in a loop
        ruo = []
        for i in range(3):

            if i != 2:
                y = torch.einsum('bij,bj->bi', torch.einsum('ijk,bk->bij', self.w_h[i], z_h[i]), h) + \
                    torch.einsum('bij,bj->bi', torch.einsum('ijk,bk->bij', self.w_x[i], z_x[i]), x) + \
                    self.d_b[i](z_b[i])
                #ruo.append(torch.sigmoid(self.layer_norm[i](y)))
                ruo.append(torch.sigmoid(y))
            else:

                m_h = torch.einsum('ijk,bk->bij', self.w_m_h[0], m_h)
                m_x = torch.einsum('ijk,bk->bij', self.w_m_x[0], m_x)
                m_h_dist = td.Bernoulli(logits=m_h)
                m_h_dist = td.Independent(m_h_dist, 2)
                m_x_dist = td.Bernoulli(logits=m_x)
                m_x_dist = td.Independent(m_x_dist, 2)
                m_h_sample = m_h_dist.sample()
                m_x_sample = m_x_dist.sample()
                #print(m_h_sample.size(), self.w_h[i].size(), z_h[i].size(), h.size(), ruo[0].size())
                y = torch.tanh(torch.einsum('bij,bj->bi',  m_h_sample*torch.einsum('ijk,bk->bij',self.w_h[i], z_h[i]), ruo[0] * h) + \
                               torch.einsum('bij,bj->bi', m_x_sample*torch.einsum('ijk,bk->bij', self.w_x[i], z_x[i]), x) + \
                               self.d_b[i](z_b[i]))

                #ruo.append(torch.tanh(self.layer_norm[i](y)))
                ruo.append(torch.tanh(y))

        r, u, o = ruo
        h_next = (1 - u) * h + u * o

        return h_next, h_hat, c_hat, m_h_dist, m_h_sample, m_x_dist, m_x_sample

# Represents TD model in PlaNET
class RSSM(nn.Module):
    """RSSM is the core recurrent part of the PlaNET module. It consists of
    two networks, one (obs) to calculate posterior beliefs and states and
    the second (img) to calculate prior beliefs and states. The prior network
    takes in the previous state and action, while the posterior network takes
    in the previous state, action, and a latent embedding of the most recent
    observation.
    """

    def __init__(self,
                 action_size: int,
                 embed_size: int,
                 stoch: int = 30,
                 deter: int = 200,
                 hidden: int = 200,
                 hyper_size: int = 15,
                 n_z: int = 12,
                 simple_rnn: bool = False,
                 act: ActFunc = None):
        """Initializes RSSM
        Args:
            action_size (int): Action space size
            embed_size (int): Size of ConvEncoder embedding
            stoch (int): Size of the distributional hidden state
            deter (int): Size of the deterministic hidden state
            hidden (int): General size of hidden layers
            hyper_size (int): The hidden size of the Hypernet
            n_z (int): embedding size of the Hypernet
            act (Any): Activation function
        """
        super().__init__()
        self.stoch_size = stoch
        self.deter_size = deter
        self.hidden_size = hidden
        self.act = act
        self.n_z = n_z
        self.hyper_size = hyper_size
        self.simple_rnn = simple_rnn
        if act is None:
            self.act = nn.ELU

        self.obs1 = Linear(embed_size + deter, hidden)
        self.obs2 = Linear(hidden, 2 * stoch)
        self.cell = HyperGRUCell(input_size=self.hidden_size, hidden_size=self.deter_size,
                                 hyper_size=self.hyper_size, n_z=self.n_z, simple_rnn=self.simple_rnn)
        self.img1 = Linear(stoch + action_size, hidden)
        self.img2 = Linear(deter, hidden)
        self.img3 = Linear(hidden, 2 * stoch)

        self.softplus = nn.Softplus

        self.device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))

    def get_initial_state(self, batch_size: int) -> List[TensorType]:
        """Returns the inital state for the RSSM, which consists of mean,
        std for the stochastic state, the sampled stochastic hidden state
        (from mean, std), and the deterministic hidden state, which is
        pushed through the GRUCell.
        Args:
            batch_size (int): Batch size for initial state
        Returns:
            List of tensors
        """
        return [
            torch.zeros(batch_size, self.stoch_size).to(self.device),
            torch.zeros(batch_size, self.stoch_size).to(self.device),
            torch.zeros(batch_size, self.stoch_size).to(self.device),
            torch.zeros(batch_size, self.deter_size).to(self.device),
        ]

    def get_initial_hyper_state(self, batch_size: int) -> List[TensorType]:
        """the hyper hidden state, and the hyper cell state.
        Args:
            batch_size (int): Batch size for initial state
        Returns:
            List of tensors
        """
        return [
            torch.zeros(batch_size, self.hyper_size).to(self.device),
            torch.zeros(batch_size, self.hyper_size).to(self.device),
        ]

    def observe(self,
                embed: TensorType,
                action: TensorType,
                state: List[TensorType] = None,
                hyper_state: List[TensorType] = None
                ) -> Tuple[List[TensorType], List[TensorType], List[TensorType], List[TensorType]]:
        """Returns the corresponding states from the embedding from ConvEncoder
        and actions. This is accomplished by rolling out the RNN from the
        starting state through each index of embed and action, saving all
        intermediate states between.
        Args:
            embed (TensorType): ConvEncoder embedding
            action (TensorType): Actions
            state (List[TensorType]): Initial state before rollout
            hyper_state (List[TensorType]): Initial hyper_states before rollout
        Returns:
            Posterior states, prior states, hyper states, and masks (all List[TensorType])
        """
        if state is None:
            state = self.get_initial_state(action.size()[0])

        if hyper_state is None:
            hyper_state = self.get_initial_hyper_state(action.size()[0])

        embed = embed.permute(1, 0, 2)
        action = action.permute(1, 0, 2)

        priors = [[] for _ in range(len(state))]
        posts = [[] for _ in range(len(state))]
        # [logp_x, entropy_x, logp_h, entropy_h]
        hyper_states = [[] for _ in range(len(hyper_state))]
        masks = [[] for _ in range(4)]  # [logp_x, entropy_x, sample_x, logp_h, entropy_h, sample_h]
        last = (state, state)
        for index in range(len(action)):
            # Tuple of post and prior
            post, prior, hyper_state, mask = self.obs_step(last[0], action[index], embed[index], hyper_state)
            last = [post, prior]
            [o.append(s) for s, o in zip(last[0], posts)]
            [o.append(s) for s, o in zip(last[1], priors)]
            [o.append(s) for s, o in zip(hyper_state, hyper_states)]
            [o.append(s) for s, o in zip(mask, masks)]

        prior = [torch.stack(x, dim=0) for x in priors]
        post = [torch.stack(x, dim=0) for x in posts]
        hyper_state = [torch.stack(x, dim=0) for x in hyper_states]
        mask = [torch.stack(x, dim=0) for x in masks]

        prior = [e.permute(1, 0, 2) for e in prior]
        post = [e.permute(1, 0, 2) for e in post]
        hyper_state = [e.permute(1, 0, 2) for e in hyper_state]
        mask = [e.permute(1, 0) for e in mask]

        return post, prior, hyper_state, mask

    def imagine(self, action: TensorType,
                state: List[TensorType] = None,
                hyper_state: List[TensorType] = None) -> Tuple[List[TensorType], List[TensorType], List[TensorType]]:
        """Imagines the trajectory starting from state through a list of actions.
        Similar to observe(), requires rolling out the RNN for each timestep.
        Args:
            action (TensorType): Actions
            state (List[TensorType]): Starting state before rollout
            hyper_state (List[TensorType]): starting hyper_state before rollout
        Returns:
            Prior states, hyper_states masks
        """
        if state is None:
            state = self.get_initial_state(action.size()[0])

        if hyper_state is None:
            hyper_state = self.get_initial_hyper_state(action.size()[0])

        action = action.permute(1, 0, 2)

        indices = range(len(action))
        priors = [[] for _ in range(len(state))]
        hyper_states = [[] for _ in range(len(hyper_state))]
        masks = [[] for _ in range(4)]  # [logp_x, entropy_x, sample_x, logp_h, entropy_h, sample_h]
        last_prior = state
        last_hyper = hyper_state
        for index in indices:
            last_prior, last_hyper, mask = self.img_step(last_prior, action[index], last_hyper)
            [o.append(s) for s, o in zip(last_prior, priors)]


        prior = [torch.stack(x, dim=0) for x in priors]
        prior = [e.permute(1, 0, 2) for e in prior]
        return prior

    def obs_step(
            self, prev_state: TensorType,
            prev_action: TensorType,
            embed: TensorType,
            hyper_state: List[TensorType])\
            -> Tuple[List[TensorType], List[TensorType], List[TensorType], List[TensorType]]:
        """Runs through the posterior model and returns the posterior state
        Args:
            prev_state (TensorType): The previous state
            prev_action (TensorType): The previous action
            embed (TensorType): Embedding from ConvEncoder
            hyper_states (List[TensorType]): previous hyper_states
        Returns:
            Post, Prior, Hyper state, Mask
      """
        prior, hyper_state, mask = self.img_step(prev_state, prev_action, hyper_state)
        x = torch.cat([prior[3], embed], dim=-1)
        x = self.obs1(x)
        x = self.act()(x)
        x = self.obs2(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = self.softplus()(std) + 0.1
        stoch = self.get_dist(mean, std).rsample()
        post = [mean, std, stoch, prior[3]]
        return post, prior, hyper_state, mask

    def img_step(self, prev_state: List[TensorType],
                 prev_action: TensorType,
                 hyper_state: List[TensorType]) -> Tuple[List[TensorType], List[TensorType], List[TensorType]]:
        """Runs through the prior model and returns the prior state
        Args:
            prev_state (TensorType): The previous state
            prev_action (TensorType): The previous action
            hyper_states (List[TensorType]): previous hyper_states
        Returns:
            Prior state
        """

        x = torch.cat([prev_state[2], prev_action], dim=-1)
        x = self.img1(x)
        x = self.act()(x)
        deter, h_hat, c_hat, m_h_dist, m_h_sample, m_x_dist, m_x_sample =\
            self.cell(x, prev_state[3], *hyper_state)  # x, h, h_hat, c_hat
        x = deter
        x = self.img2(x)
        x = self.act()(x)
        x = self.img3(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = self.softplus()(std) + 0.1
        stoch = self.get_dist(mean, std).rsample()
        m_x_logp, m_x_entropy = m_x_dist.log_prob(m_x_sample), m_x_dist.entropy()
        m_h_logp, m_h_entropy = m_h_dist.log_prob(m_h_sample), m_h_dist.entropy()

        return [mean, std, stoch, deter],\
               [h_hat, c_hat], \
               [m_x_logp, m_x_entropy, m_h_logp, m_h_entropy]
        # [m_x_logp, m_x_entropy, m_x_sample, m_h_logp, m_h_entropy, m_h_sample]


    def get_feature(self, state: List[TensorType]) -> TensorType:
        # Constructs feature for input to reward, decoder, actor, critic
        return torch.cat([state[2], state[3]], dim=-1)

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
        self.hyper_size = model_config['hyper_size']
        self.n_z = model_config['n_z']
        self.decay = model_config['decay']
        self.simple_rnn = model_config['simple_rnn']


        self.action_size = action_space.shape[0]

        self.encoder = ConvEncoder(self.depth)
        self.decoder = ConvDecoder(
            self.stoch_size + self.deter_size, depth=self.depth)
        self.reward = DenseDecoder(self.stoch_size + self.deter_size, 1, 2,
                                   self.hidden_size)
        self.dynamics = RSSM(
            self.action_size,
            32 * self.depth,
            stoch=self.stoch_size,
            deter=self.deter_size,
            hidden=self.hidden_size,
            hyper_size=self.hyper_size,
            n_z=self.n_z,
            simple_rnn=self.simple_rnn
            )
        self.actor = ActionDecoder(self.stoch_size + self.deter_size,
                                   self.action_size, 4, self.hidden_size)
        self.value = DenseDecoder(self.stoch_size + self.deter_size, 1, 3,
                                  self.hidden_size)
        self.state = None

        self.ema = EMABatchTime(self.decay)
        self.device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))

    def policy(self, obs: TensorType, state: List[TensorType], explore=True
                 ) -> Tuple[TensorType, List[float], List[TensorType]]:
        """Returns the action. Runs through the encoder, recurrent model,
        and policy to obtain action.
        """
        if state is None:
            self.initial_state()
        else:
            self.state = state

        #split states into it's three parts
        post = self.state[:4]
        hyper_state = self.state[4:-1]
        action = self.state[-1]

        embed = self.encoder(obs)
        post, _, hyper_state, _ = self.dynamics.obs_step(post, action, embed, hyper_state)
        feat = self.dynamics.get_feature(post)

        action_dist = self.actor(feat)
        if explore:
            action = action_dist.sample()
        else:
            action = action_dist.mean
        logp = action_dist.log_prob(action)

        self.state = post + hyper_state + [action]
        return action, logp, self.state

    def imagine_ahead(self, state: List[TensorType], hyper_state: List[TensorType],
                      horizon: int) -> TensorType:
        """Given a batch of states and hyperstates rolls out more state of length horizon.
        """
        start_prior = []
        start_hyper = []
        for s in state:
            s = s.contiguous().detach()
            shpe = [-1] + list(s.size())[2:]
            start_prior.append(s.view(*shpe))

        for s in hyper_state:
            s = s.contiguous().detach()
            shpe = [-1] + list(s.size())[2:]
            start_hyper.append(s.view(*shpe))


        def next_state(state, hyper_state):
            feature = self.dynamics.get_feature(state).detach()
            action = self.actor(feature).rsample()
            next_state, hyper_state, _ = self.dynamics.img_step(state, action, hyper_state)
            return next_state, hyper_state, _

        last = start_prior
        last_hyper = start_hyper
        outputs = [[] for _ in range(len(start_prior))]
        for _ in range(horizon):
            last, last_hyper, _ = next_state(last, last_hyper)
            [o.append(s) for s, o in zip(last, outputs)]
        outputs = [torch.stack(x, dim=0) for x in outputs]

        imag_feat = self.dynamics.get_feature(outputs)
        return imag_feat

    def get_initial_state(self) -> List[TensorType]:
        """Initial state consists of the initial state of the RSSM and the hyper network on top of it and action
                """
        self.state = self.dynamics.get_initial_state(1) + self.dynamics.get_initial_hyper_state(1) +\
            [torch.zeros(1, self.action_space.shape[0]).to(self.device)]
        return self.state

    def value_function(self) -> TensorType:
        return None
