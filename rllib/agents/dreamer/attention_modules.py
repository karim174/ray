from typing import Union
from typing import Any, List, Tuple, Optional, Union
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.torch_ops import sequence_mask
from ray.rllib.utils.typing import TensorType
from ray.rllib.models.torch.modules import GRUGate, SkipConnection
from typing import Any, List, Tuple, Optional, Union


torch, nn = try_import_torch()


class RelativePositionEmbedding(nn.Module):
    """Creates a [seq_length x seq_length] matrix for rel. pos encoding.
    Denoted as Phi in [2] and [3]. Phi is the standard sinusoid encoding
    matrix.
    Args:
        seq_length (int): The max. sequence length (time axis).
        out_dim (int): The number of nodes to go into the first Tranformer
            layer with.
    Returns:
        torch.Tensor: The encoding matrix Phi.
    """

    def __init__(self, out_dim, **kwargs):
        super().__init__()
        self.out_dim = out_dim
        self.device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))
        out_range = torch.arange(0, self.out_dim, 2.0).to(self.device)
        inverse_freq = 1 / (10000**(out_range / self.out_dim))
        self.register_buffer("inverse_freq", inverse_freq)

    def forward(self, seq_length):
        pos_input = torch.arange(
            seq_length - 1, -1, -1.0,
            dtype=torch.float).to(self.inverse_freq.device)
        sinusoid_input = torch.einsum("i,j->ij", pos_input, self.inverse_freq)
        pos_embeddings = torch.cat(
            [torch.sin(sinusoid_input),
             torch.cos(sinusoid_input)], dim=-1)
        return pos_embeddings[:, None, :]


class RelativeMultiHeadAttention(nn.Module):
    """A RelativeMultiHeadAttention layer as described in [3].
    Uses segment level recurrence with state reuse.
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_heads: int,
                 head_dim: int,
                 input_layernorm: bool = False,
                 output_activation: Union[str, callable] = None,
                 **kwargs):
        """Initializes a RelativeMultiHeadAttention nn.Module object.
        Args:
            in_dim (int):
            out_dim (int): The output dimension of this module. Also known as
                "attention dim".
            num_heads (int): The number of attention heads to use.
                Denoted `H` in [2].
            head_dim (int): The dimension of a single(!) attention head
                Denoted `D` in [2].
            input_layernorm (bool): Whether to prepend a LayerNorm before
                everything else. Should be True for building a GTrXL.
            output_activation (Union[str, callable]): Optional activation
                function or activation function specifier (str).
                Should be "relu" for GTrXL.
            **kwargs:
        """
        super().__init__(**kwargs)

        # No bias or non-linearity.
        self.device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))
        self._num_heads = num_heads
        self._head_dim = head_dim

        # 3=Query, key, and value inputs.
        self._qkv_layer = SlimFC(
            in_size=in_dim, out_size=3 * num_heads * head_dim, use_bias=False)

        self._linear_layer = SlimFC(
            in_size=num_heads * head_dim,
            out_size=out_dim,
            use_bias=False,
            activation_fn=output_activation)

        self._uvar = nn.Parameter(torch.zeros(num_heads, head_dim))
        self._vvar = nn.Parameter(torch.zeros(num_heads, head_dim))
        nn.init.xavier_uniform_(self._uvar)
        nn.init.xavier_uniform_(self._vvar)
        self.register_parameter("_uvar", self._uvar)
        self.register_parameter("_vvar", self._vvar)

        self._pos_proj = SlimFC(
            in_size=in_dim, out_size=num_heads * head_dim, use_bias=False)
        self._rel_pos_embedding = RelativePositionEmbedding(out_dim)

        self._input_layernorm = None
        if input_layernorm:
            self._input_layernorm = torch.nn.LayerNorm(in_dim)

    def forward(self, inputs: TensorType,
                memory: TensorType = None) -> TensorType:
        T = list(inputs.size())[1]  # length of segment (time)
        H = self._num_heads  # number of attention heads
        d = self._head_dim  # attention head dimension

        # Add previous memory chunk (as const, w/o gradient) to input.
        # Tau (number of (prev) time slices in each memory chunk).
        Tau = list(memory.shape)[1]
        inputs = torch.cat((memory.detach(), inputs), dim=1)

        # Apply the Layer-Norm.
        if self._input_layernorm is not None:
            inputs = self._input_layernorm(inputs)

        qkv = self._qkv_layer(inputs)

        queries, keys, values = torch.chunk(input=qkv, chunks=3, dim=-1)
        # Cut out Tau memory timesteps from query.
        queries = queries[:, -T:]

        queries = torch.reshape(queries, [-1, T, H, d])
        keys = torch.reshape(keys, [-1, Tau + T, H, d])
        values = torch.reshape(values, [-1, Tau + T, H, d])

        R = self._pos_proj(self._rel_pos_embedding(Tau + T))
        R = torch.reshape(R, [Tau + T, H, d])

        # b=batch
        # i and j=time indices (i=max-timesteps (inputs); j=Tau memory space)
        # h=head
        # d=head-dim (over which we will reduce-sum)
        score = torch.einsum("bihd,bjhd->bijh", queries + self._uvar, keys)
        pos_score = torch.einsum("bihd,jhd->bijh", queries + self._vvar, R)
        score = score + self.rel_shift(pos_score)
        score = score / d**0.5

        # causal mask of the same length as the sequence
        mask = sequence_mask(
            torch.arange(Tau + 1, Tau + T + 1),
            dtype=score.dtype).to(score.device)
        mask = mask[None, :, :, None]

        masked_score = score * mask + 1e30 * (mask.float() - 1.)
        wmat = nn.functional.softmax(masked_score, dim=2)

        out = torch.einsum("bijh,bjhd->bihd", wmat, values)
        shape = list(out.shape)[:2] + [H * d]
        out = torch.reshape(out, shape)

        return self._linear_layer(out)

    @staticmethod
    def rel_shift(x: TensorType) -> TensorType:
        # Transposed version of the shift approach described in [3].
        # https://github.com/kimiyoung/transformer-xl/blob/
        # 44781ed21dbaec88b280f74d9ae2877f52b492a5/tf/model.py#L31
        x_size = list(x.shape)

        x = torch.nn.functional.pad(x, (0, 0, 1, 0, 0, 0, 0, 0))
        x = torch.reshape(x, [x_size[0], x_size[2] + 1, x_size[1], x_size[3]])
        x = x[:, 1:, :, :]
        x = torch.reshape(x, x_size)

        return x


class GTrXLNet(nn.Module):
    """A GTrXL net Model described in [2].
    This is still in an experimental phase.
    Can be used as a drop-in replacement for LSTMs in PPO and IMPALA.
    For an example script, see: `ray/rllib/examples/attention_net.py`.
    To use this network as a replacement for an RNN, configure your Trainer
    as follows:
    Examples:
        >> config["model"]["custom_model"] = GTrXLNet
        >> config["model"]["max_seq_len"] = 10
        >> config["model"]["custom_model_config"] = {
        >>     num_transformer_units=1,
        >>     attention_dim=32,
        >>     num_heads=2,
        >>     memory_tau=50,
        >>     etc..
        >> }
    """

    def __init__(self,
                 input_dim: int,
                 action_dim: int,
                 output_dim: int,
                 num_transformer_units: int = 1,
                 attention_dim: int = 64,
                 num_heads: int = 2,
                 memory_inference: int = 50,
                 memory_training: int = 50,
                 head_dim: int = 32,
                 position_wise_mlp_dim: int = 32,
                 init_gru_gate_bias: float = 2.0):
        """Initializes a GTrXLNet.
        Args:
            num_transformer_units (int): The number of Transformer repeats to
                use (denoted L in [2]).
            attention_dim (int): The input and output dimensions of one
                Transformer unit.
            num_heads (int): The number of attention heads to use in parallel.
                Denoted as `H` in [3].
            memory_inference (int): The number of timesteps to concat (time
                axis) and feed into the next transformer unit as inference
                input. The first transformer unit will receive this number of
                past observations (plus the current one), instead.
            memory_training (int): The number of timesteps to concat (time
                axis) and feed into the next transformer unit as training
                input (plus the actual input sequence of len=max_seq_len).
                The first transformer unit will receive this number of
                past observations (plus the input sequence), instead.
            head_dim (int): The dimension of a single(!) attention head within
                a multi-head attention unit. Denoted as `d` in [3].
            position_wise_mlp_dim (int): The dimension of the hidden layer
                within the position-wise MLP (after the multi-head attention
                block within one Transformer unit). This is the size of the
                first of the two layers within the PositionwiseFeedforward. The
                second layer always has size=`attention_dim`.
            init_gru_gate_bias (float): Initial bias values for the GRU gates
                (two GRUs per Transformer unit, one after the MHA, one after
                the position-wise MLP).
        """

        super().__init__()
        self.num_transformer_units = num_transformer_units
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.memory_inference = memory_inference
        self.memory_training = memory_training
        self.head_dim = head_dim
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.output_dim = output_dim  # dynamics

        self.linear_layer = SlimFC(
            in_size=self.input_dim, out_size=self.attention_dim)

        self.layers = [self.linear_layer]

        attention_layers = []
        # 2) Create L Transformer blocks according to [2].
        for i in range(self.num_transformer_units):
            # RelativeMultiHeadAttention part.
            MHA_layer = SkipConnection(
                RelativeMultiHeadAttention(
                    in_dim=self.attention_dim,
                    out_dim=self.attention_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    input_layernorm=True,
                    output_activation=nn.ReLU),
                fan_in_layer=GRUGate(self.attention_dim, init_gru_gate_bias))

            # Position-wise MultiLayerPerceptron part.
            E_layer = SkipConnection(
                nn.Sequential(
                    torch.nn.LayerNorm(self.attention_dim),
                    SlimFC(
                        in_size=self.attention_dim,
                        out_size=position_wise_mlp_dim,
                        use_bias=False,
                        activation_fn=nn.ReLU),
                    SlimFC(
                        in_size=position_wise_mlp_dim,
                        out_size=self.attention_dim,
                        use_bias=False,
                        activation_fn=nn.ReLU)),
                fan_in_layer=GRUGate(self.attention_dim, init_gru_gate_bias))

            # Build a list of all attanlayers in order.
            attention_layers.extend([MHA_layer, E_layer])

        # Create a Sequential such that all parameters inside the attention
        # layers are automatically registered with this top-level model.
        self.attention_layers = nn.Sequential(*attention_layers)
        self.layers.extend(attention_layers)

        # Postprocess GTrXL output with another hidden layer.
        self.out = SlimFC(in_size=self.attention_dim, out_size=self.output_dim,
                          activation_fn=nn.ReLU)

    def forward(self, input, state: List[TensorType]) -> (TensorType, List[TensorType]):

        all_out = input
        memory_outs = []
        for i in range(len(self.layers)):
            # MHA layers which need memory passed in.
            if i % 2 == 1:
                all_out = self.layers[i](all_out, memory=state[i // 2])
            # Either self.linear_layer (initial obs -> attn. dim layer) or
            # MultiLayerPerceptrons. The output of these layers is always the
            # memory for the next forward pass.
            else:
                all_out = self.layers[i](all_out)
                memory_outs.append(all_out)

        # Discard last output (not needed as a memory since it's the last
        # layer).
        memory_outs = torch.stack(memory_outs[:-1])#memory_outs[:-1]
        out = self.out(all_out)

        return out, memory_outs
