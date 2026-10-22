# mypy: allow-untyped-defs
import copy
import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import xavier_uniform_

from .activation import MultiheadAttention  # .表示相对导入（relative import），指从当前包（package）中导入模块，即等价于from torch.nn.modules.activation import MultiheadAttention
from .container import ModuleList
from .dropout import Dropout
from .linear import Linear
from .module import Module                  # Base class for all neural network modules
from .normalization import LayerNorm


__all__ = [
    "Transformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
]


# 已看完
# 生成用于屏蔽后续位置的方阵掩码
def _generate_square_subsequent_mask(
    sz: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    #https://docs.pytorch.org/docs/2.13/generated/torch.triu.html中有:
    #   the other elements of the result tensor out are set to 0.
    #   If diagonal = 0, all elements on and above the main diagonal are retained.
    #   A positive value excludes just as many diagonals above the main diagonal
    return torch.triu(
        torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )


# 已看完
def _get_seq_len(src: Tensor, batch_first: bool) -> int | None:
    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


class Transformer(Module):
    r"""A basic transformer layer.


    This Transformer layer implements the original Transformer architecture described
    in the `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_ paper. The
    intent of this layer is as a reference implementation for foundational understanding
    and thus it contains only limited features relative to newer Transformer architectures.
    Given the fast pace of innovation in transformer-like architectures, we recommend
    exploring this `tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_
    to build an efficient transformer layer from building blocks in core or using higher
    level libraries from the `PyTorch Ecosystem <https://landscape.pytorch.org/>`_.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples:
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = F.relu,
        custom_encoder: Any | None = None,
        custom_decoder: Any | None = None,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                layer_norm_eps,
                batch_first,
                norm_first,
                bias,
                **factory_kwargs,
            )
            encoder_norm = LayerNorm(
                d_model,
                eps=layer_norm_eps,
                bias=bias,
                **factory_kwargs,
            )
            self.encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm
            )

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                layer_norm_eps,
                batch_first,
                norm_first,
                bias,
                **factory_kwargs,
            )
            decoder_norm = LayerNorm(
                d_model,
                eps=layer_norm_eps,
                bias=bias,
                **factory_kwargs,
            )
            self.decoder = TransformerDecoder(
                decoder_layer, num_decoder_layers, decoder_norm
            )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        src_is_causal: bool | None = None,
        tgt_is_causal: bool | None = None,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Take in and process masked source/target sequences.

        .. note::

            If a boolean tensor is provided for any of the [src/tgt/memory]_mask arguments, positions with a ``True`` value are
            not allowed to participate in the attention,
            which is the opposite of the definition for :attr:`attn_mask`
            in :func:`torch.nn.functional.scaled_dot_product_attention`.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the Tensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the Tensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the Tensor mask for memory keys per batch (optional).
            src_is_causal: If specified, applies a causal mask as ``src_mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``src_is_causal`` provides a hint that ``src_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            tgt_is_causal: If specified, applies a causal mask as ``tgt_mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory_mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            - src: :math:`(S, E)` for unbatched input, :math:`(S, N, E)` if `batch_first=False` or
              `(N, S, E)` if `batch_first=True`.
            - tgt: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.                                       src_mask&tgt_mask仅用于自注意力机制且可全batch共享,因其是结构性规则:位置i只能看0..i,对batch里每条都一样;
            - src_mask: :math:`(S, S)` or :math:`(N\cdot\text{num\_heads}, S, S)`.                  encoder自注意力的位置对掩码(多数任务不用,传None);
            - tgt_mask: :math:`(T, T)` or :math:`(N\cdot\text{num\_heads}, T, T)`.                  decoder自注意力的因果掩码(训练时必需,否则泄漏); 
            - memory_mask: :math:`(T, S)`.                                                          decoder的cross attention的位置对掩码(极少用);
            - src_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.       encoder自注意力时指定key中的padding; query无需指定padding,因padding的输出虽是垃圾,却无害,可由ignore_index处理;
            - tgt_key_padding_mask: :math:`(T)` for unbatched input otherwise :math:`(N, T)`.       decoder自注意力时指定key中的padding;
            - memory_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.    decoder的cross attention时指定key中的padding;

            Note: [src/tgt/memory]_mask ensures that position :math:`i` is allowed to attend the unmasked
            positions. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decoder.

            where :math:`S` is the source sequence length, :math:`T` is the target sequence length, :math:`N` is the
            batch size, :math:`E` is the feature number

        Examples:
            >>> # xdoctest: +SKIP
            >>> output = transformer_model(
            ...     src, tgt, src_mask=src_mask, tgt_mask=tgt_mask
            ... )
        """
        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError(
                "the feature number of src and tgt must be equal to d_model"
            )

        memory = self.encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=src_is_causal,
        )
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal,
        )
        return output

    @staticmethod
    def generate_square_subsequent_mask(
        sz: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        r"""Generate a square causal mask for the sequence.

        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        return _generate_square_subsequent_mask(sz, dtype=dtype, device=device)

    # 已看完
    def _reset_parameters(self) -> None:
        r"""Initialize parameters in the transformer model."""
        for p in self.parameters(): # self.parameters()返回的是nn.Parameter对象; nn.Parameter继承自torch.Tensor; 
            if p.dim() > 1: # 所有Tensor都有dim()方法; Transformer的所有p.dim()>1的parameter都被用xavier_uniform_重新初始化,如:Attention层的W_Q,W_K,W_V,W_O;FFN层的W1,W2
                xavier_uniform_(p) 
            # 而所有p.dim()<=1的parameter则保留原本的初始化,如:Attention及FFN层的上述weights对应的bias(MultiheadAttention的_reset_parameters将Attention中的bias初始化为0,FFN使用U(-1/sqrt(fan_in),1/sqrt(fan_in))初始化bias); LayerNorm的γ,β分别初始化为1,0;


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers.

    This TransformerEncoder layer implements the original architecture described
    in the `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_ paper. The
    intent of this layer is as a reference implementation for foundational understanding
    and thus it contains only limited features relative to newer Transformer architectures.
    Given the fast pace of innovation in transformer-like architectures, we recommend
    exploring this `tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_
    to build efficient layers from building blocks in core or using higher
    level libraries from the `PyTorch Ecosystem <https://landscape.pytorch.org/>`_.

    .. warning::
        All layers in the TransformerEncoder are initialized with the same parameters.
        It is recommended to manually initialize the layers after creating the TransformerEncoder instance. ???

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer: "TransformerEncoderLayer",
        num_layers: int,
        norm: Module | None = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # this attribute saves the value provided at object construction
        self.enable_nested_tensor = enable_nested_tensor    # 保留enable_nested_tensor的原始配置,当前后续未使用
        # this attribute controls whether nested tensors are used
        self.use_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

        enc_layer = "encoder_layer"
        why_not_sparsity_fast_path = ""
        if not isinstance(encoder_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{enc_layer} was not TransformerEncoderLayer"
        # TransformerEncoderLayer.forward中why_not_sparsity_fast_path没有对norm_first的检查,因为_transformer_encoder_layer_fwd对pre/post-norm都实现了;
        # 此处的检查是因为pre-norm要求融合kernel(torch._transformer_encoder_layer_fwd)内部显式维护类似于x + self._sa_block(self.norm1(x), ...)两条线,这需要单独实现和验证,于是对其在栈级保守关闭nested转换.
        #   post-norm(BERT系)实现和验证过;
        #   post-norm是单流管线,pre-norm是分叉-汇合(fork-join)结构??？;而nested输入把这个实现负担进一步放大;
        elif encoder_layer.norm_first:
            why_not_sparsity_fast_path = f"{enc_layer}.norm_first was True"
        elif not encoder_layer.self_attn.batch_first:   # 静态结构(构造后不变),在__init__中检查
            why_not_sparsity_fast_path = (
                f"{enc_layer}.self_attn.batch_first was not True"
                + "(use batch_first for better inference performance)"
            )
        elif not encoder_layer.self_attn._qkv_same_embed_dim:   # 静态结构(构造后不变),在__init__中检查
            why_not_sparsity_fast_path = (
                f"{enc_layer}.self_attn._qkv_same_embed_dim was not True"
            )
        elif encoder_layer.self_attn.in_proj_bias is None:  # 静态结构(构造后不变),在__init__中检查
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn was passed bias=False"
        elif not encoder_layer.activation_relu_or_gelu: # 静态结构(构造后不变),在__init__中检查
            why_not_sparsity_fast_path = (
                f"{enc_layer}.activation_relu_or_gelu was not True"
            )
        elif encoder_layer.norm1.eps != encoder_layer.norm2.eps:    # 静态结构(构造后不变),在__init__中检查
            why_not_sparsity_fast_path = (
                f"{enc_layer}.norm1.eps was not equal to {enc_layer}.norm2.eps"
            )
        elif encoder_layer.self_attn.num_heads % 2 == 1:    # 静态结构(构造后不变),在__init__中检查
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn.num_heads is odd"

        if enable_nested_tensor and why_not_sparsity_fast_path:
            warnings.warn(
                f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}",
                stacklevel=2,
            )
            self.use_nested_tensor = False

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool | None = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src    # 此处output与src指向同一个Tensor对象; 下面执行output = torch._nested_tensor_from_mask(/output = mod(/其它时,会让其指向其他Tensor;
        convert_to_nested = False
        first_layer = self.layers[0]    # 查第一层是"各层通常同步"假设下的低开销;
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ""
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
        do_mask_check = getattr(self, "mask_check", True)

        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = (
                "torch.backends.mha.get_fastpath_enabled() was not True"
            )
        elif not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = (
                "self.use_nested_tensor (set in init) was not True"
            )
        elif first_layer.training:  # 运行时状态(逐调用可变); 为什么不能用TransformerEncoder.training?语义权威性:决定TransformerEncoderLayer层的行为应是其自己的training属性;
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif src.dim() != 3:    # 运行时状态(逐调用可变)
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif src_key_padding_mask is None:  # 下面_nested_tensor_from_mask需要src_key_padding_mask
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        # This check avoids a call to torch._nested_tensor_from_mask_left_aligned() that
        # breaks in torch.compile.   # torch._nested_tensor_from_mask_left_aligned内部有数据依赖的控制流,在torch.compile/export下会图断裂,故编译模式下直接放弃nested转换
        elif do_mask_check and torch.compiler.is_compiling():
            why_not_sparsity_fast_path = (
                "mask_check enabled with torch.compile or torch.export"
            )                                                                   # 运行时状态(逐调用可变)
        elif do_mask_check and not torch._nested_tensor_from_mask_left_aligned( # 这张mask能否合法地把这个src转成nested:每条序列的有效token全部集中在左侧、padding全部在右侧
            src, src_key_padding_mask.logical_not()
        ):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:  # 下面_nested_tensor_from_mask无意义,但NestedTensor的out通过下面的mod(...)调用仍会传给TransformerEncoderLayer.forward
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:  # 运行时状态(逐调用可变)
            why_not_sparsity_fast_path = (
                "src_key_padding_mask and mask were both supplied"
            )
        elif torch.is_autocast_enabled():   # 运行时状态(逐调用可变)
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = [
                "cpu",
                "cuda",
                "xpu",
                torch.utils.backend_registration._privateuse1_backend_name,
            ]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = (
                    f"src device is neither one of {_supported_device_type}"
                )
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args): # 运行时状态(逐调用可变)
                why_not_sparsity_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(  ???为啥这个函数的调用使用why_not_sparsity_fast_path来判断??? # mask_check=False执行转换——不重复_nested_tensor_from_mask_left_aligned进行的校验
                    output, src_key_padding_mask.logical_not(), mask_check=False
                )
                src_key_padding_mask_for_layers = None

        seq_len = _get_seq_len(src, batch_first) # 获取输入序列的长度
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask_for_layers,
            )

        if convert_to_nested:
            output = output.to_padded_tensor(0.0, src.size())

        if self.norm is not None:
            output = self.norm(output)  # self.norm主要是为norm_first=True(pre-norm)时设计的; 在post-norm时它确实接近冗余(与最后一层的norm2功能重叠);

        return output


class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers.

    This TransformerDecoder layer implements the original architecture described
    in the `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_ paper. The
    intent of this layer is as a reference implementation for foundational understanding
    and thus it contains only limited features relative to newer Transformer architectures.
    Given the fast pace of innovation in transformer-like architectures, we recommend
    exploring this `tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_
    to build efficient layers from building blocks in core or using higher
    level libraries from the `PyTorch Ecosystem <https://landscape.pytorch.org/>`_.

    .. warning::
        All layers in the TransformerDecoder are initialized with the same parameters.
        It is recommended to manually initialize the layers after creating the TransformerDecoder instance.

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    __constants__ = ["norm"]

    def __init__(
        self,
        decoder_layer: "TransformerDecoderLayer",
        num_layers: int,
        norm: Module | None = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool | None = None,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )

        if self.norm is not None:
            output = self.norm(output) # 类似于TransformerEncoder中的:self.norm主要是为norm_first=True(pre-norm)时设计的; 在post-norm时它确实接近冗余(与最后一层的norm2功能重叠);

        return output


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.

    This TransformerEncoderLayer implements the original architecture described
    in the `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_ paper. The
    intent of this layer is as a reference implementation for foundational understanding
    and thus it contains only limited features relative to newer Transformer architectures.
    Given the fast pace of innovation in transformer-like architectures, we recommend
    exploring this `tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_
    to build efficient layers from building blocks in core or using higher
    level libraries from the `PyTorch Ecosystem <https://landscape.pytorch.org/>`_.

    TransformerEncoderLayer can handle either traditional torch.tensor inputs,
    or Nested Tensor inputs.  Derived classes are expected to similarly accept
    both input formats.  (Not all combinations of inputs are currently          虽然TransformerEncoderLayer声称可以处理Nested Tensor,但如TransformerEncoderLayer.forward的所有“输入张量类 +可选参数”的组合并不是都能正常工作
    supported by TransformerEncoderLayer while Nested Tensor is in prototype    https://docs.pytorch.org/docs/2.13/nested.html中有: Warning Nested tensors are not currently under active development. Use at your own risk.
    state.)

    If you are implementing a custom layer, you may derive it either from
    the Module or TransformerEncoderLayer class.  If your custom layer
    supports both torch.Tensors and Nested Tensors inputs, make its
    implementation a derived class of TransformerEncoderLayer. If your custom
    Layer supports only torch.Tensor inputs, derive its implementation from
    Module.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).      "关于LayerNorm"中的ϵ: small constant to prevent division by zero.
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward        "关于Post-LayerNorm与Pre-LayerNorm"
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive   还涉及MultiheadAttention层
            bias. Default: ``True``.

    Examples:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(
        ...     d_model=512, nhead=8, batch_first=True
        ... )
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation described in
        `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor    对应下面的elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args)的逻辑取反
          argument ``requires_grad``                                                                        autograd的建图规则是:一个操作被记录到计算图,当且仅当grad模式开启AND至少一个输入requires_grad=True
        - training is disabled (using ``.eval()``)              对应下面的elif self.training的逻辑取反;   "推理"只描述模型被使用这一事实,不排斥梯度:梯度可以作为分析工具,还可以驱动真正的参数更新; 建图由grad模式+requires_grad在前向时隐式控制; backward由用户显式调用触发; 参数更新由optimizer.step()独立完成;
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed          过时了  
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.

        .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:  torch/nn/modules/activation.py中MultiheadAttention部分有:FlashAttention是指at::native::scaled_dot_product_attention中的SDPBackend::flash_attention实现
         https://arxiv.org/abs/2205.14135

        此Fast path就是指下面调用torch._transformer_encoder_layer_fwd; 引用FlashAttention论文是因为其融合kernel的注意力计算(_native_multi_head_attention)采用了论文提出的IO-aware融合注意力技术(调用at::native::scaled_dot_product_attention),而整层融合进一步省去Python逐算子调度的开销;
        但其实非Fast path路径(self._sa_block)涉及的MultiheadAttention内部也会走scaled_dot_product_attention;
            TransformerEncoderLayer.forward
            │
            ├─【路径1：layer 级 fast path】条件最苛刻
            │    torch._transformer_encoder_layer_fwd          ← 整层融合（MHA+残差+LN+FFN 一次 C++ 调用）
            │      └─ aten/.../transformer.cpp
            │           └─ _native_multi_head_attention
            │                └─ SDPA → flash / efficient / cudnn / math
            │
            └─【路径2/3：慢速路径】_sa_block → MultiheadAttention.forward
                │
                ├─【路径2：MHA 级 fastpath】条件次苛刻（self-attn、batch_first、eval、无梯度…）
                │    torch._native_multi_head_attention        ← 仅MHA
                │      └─ SDPA → flash / efficient / cudnn / math
                │
                └─【路径3：纯 Python 路径】
                    F.multi_head_attention_forward
                        └─ need_weights=False 时 → F.scaled_dot_product_attention
                            └─ SDPA → flash / efficient / cudnn / math
    """

    __constants__ = ["norm_first"]

    # 已看完
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,    # torch/nn/functional.py的multi_head_attention_forward中有:在softmax之后应用了dropout;
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)    # Dropout无状态,故:若self.dropout,self.dropout1,self.dropout2共用一个实例时,计算结果等价；但分开实例能保证独立配置p;
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.   stash:verb:to put by or away as for safekeeping or future use, usually in a secret place (usually followed byaway ).
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):   # F.relu是函数,nn.ReLU是封装它的Module子类；
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    # 已看完
    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:        # forward返回值的形状与输入src完全相同
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).   应是遵循上面:src_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        why_not_sparsity_fast_path = ""
        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = (
                "torch.backends.mha.get_fastpath_enabled() was not True"
            )
        elif src.dim() != 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif self.training: # 训练语义需要:1.融合kernel(_transformer_encoder_layer_fwd)没有的反向传播实现; 2.融合kernel(_transformer_encoder_layer_fwd)不执行的dropout;
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif self.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = "self_attn was passed bias=False"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:  # MultiheadAttention.forward中调用的torch._native_multi_head_attention不涉及激活
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif self.norm1.eps != self.norm2.eps:  # MultiheadAttention.forward中调用的torch._native_multi_head_attention不涉及norm
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (                ???到此为止???
            src_key_padding_mask is not None or src_mask is not None    # NestedTensor的jagged结构已经完整编码了"哪些是padding",故无需src_key_padding_mask
        ):  # nested输入的注意力分数矩阵是锯齿(ragged)的——每个样本的query/key长度不同,不存在规则的(S, S)或(N·H, S, S)稠密矩阵;而src_mask恰好是稠密的(S, S)或(N·H, S, S)形状,无法与ragged布局建立位置对应;
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
            # Autocast(自动混合精度)自动把"适合低精度"的算子(matmul、conv等)的输入转成fp16/bf16计算,其余算子保持fp32——用更低的显存和更快的Tensor Core吞吐加速,同时维持数值稳定性.
            #     白名单机制:框架内置算子级策略——matmul/linear/conv进fp16(计算密集、精度容忍高); LayerNorm/softmax/求和类保持fp32(归约易溢出);
            #     自动插入cast:进入算子前按策略把输入cast到目标精度,输出dtype由策略决定,用户无需手动;
        elif any(   # any里面是省略了()的生成器表达式
            len(getattr(m, "_forward_hooks", {}))
            + len(getattr(m, "_forward_pre_hooks", {}))
            for m in self.modules() # 遍历本层及其所有子模块(self、self_attn、linear1、linear2、norm1、norm2、各Dropout等),只要任何一个挂了forward hook或forward pre-hook,就拒绝走融合kernel;
        ):                          # 融合kernel(下面调用的torch._transformer_encoder_layer_fwd)会绕过Python调用链而省略用户回调;
            # MultiheadAttention.forward中调用torch._native_multi_head_attention前没类似检查,因为:
	        #   MHA的fast path与slow path在hook语义上完全一致:MHA的慢速路径本身就不会触发任何子模块hook,F.multi_head_attention_forward内部全部用F.linear等函数式接口——self.out_proj这个子模块的__call__从未被调用;
            why_not_sparsity_fast_path = "forward pre-/hooks are attached to the module"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            _supported_device_type = [
                "cpu",
                "cuda",
                "xpu",
                torch.utils.backend_registration._privateuse1_backend_name,
            ]
            if torch.overrides.has_torch_function(tensor_args): # __torch_function__协议
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all(
                (x.device.type in _supported_device_type) for x in tensor_args
            ):
                why_not_sparsity_fast_path = (
                    "some Tensor argument's device is neither one of "
                    f"{_supported_device_type}"
                )
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = (  # torch._transformer_encoder_layer_fwd是C++融合kernel:MHA+残差+LayerNorm+FFN在一次调用内完成,没有对应的backward实现(反向需要的中间激活值在融合计算中被丢弃/未暴露)
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(    # 因为是在TransformerEncoderLayer即在encoder中,故此处仅涉及self-attention的计算
                    src_mask, src_key_padding_mask, src                 # 而由上面elif src.dim() != 3知此处必为batched
                )
                return torch._transformer_encoder_layer_fwd( # aten/src/ATen/native/transformers/transformer.cpp中的transformer_encoder_layer_forward直接调用:_native_multi_head_attention计算注意力
                    src,                                     # 继续参考torch/nn/modules/activation.py的MultiheadAttention部分:fastpath inference指的是在MultiheadAttention.forward中直接调用torch._native_multi_head_attention
                    self.self_attn.embed_dim,                # _transformer_encoder_layer_fwd是为推理优化:1.不保存反向所需的中间激活;2.也不实现dropout; MultiheadAttention.forward中的torch._native_multi_head_attention也是如此
                    self.self_attn.num_heads,                # SDPA(at::native::scaled_dot_product_attention)是只融合了注意力核心的kernel,且其注册了backward;
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(     # 上有:torch/nn/functional.py的multi_head_attention_forward中有:在softmax之后应用了dropout;
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

    This TransformerDecoderLayer implements the original architecture described
    in the `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_ paper. The
    intent of this layer is as a reference implementation for foundational understanding
    and thus it contains only limited features relative to newer Transformer architectures.
    Given the fast pace of innovation in transformer-like architectures, we recommend
    exploring this `tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_
    to build efficient layers from building blocks in core or using higher
    level libraries from the `PyTorch Ecosystem <https://landscape.pytorch.org/>`_.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(
        ...     d_model=512, nhead=8, batch_first=True
        ... )
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        self.multihead_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


# 已看完
def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])    # Lists may be constructed in several ways:Using a list comprehension: [x for x in iterable]
                                                                    # range(stop)

# 已看完
def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:  # 返回一个可调用对象:接收一个Tensor,返回一个Tensor
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


# 已看完
# 检测是否为标准因果掩码
# Causal（因果）有着极其严格的定义：它特指标准的、下三角全为1(或上三角全为-infty)的方阵掩码
def _detect_is_causal_mask(
    mask: Tensor | None,
    is_causal: bool | None = None, # 因为需要区分三种状态：用户明确知道是因果掩码（True）; 用户明确知道不是因果掩码（False）; 用户不确定/让系统自动检测（None）
    size: int | None = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    # # 假设我们没有引入新变量 `make_causal`
    # if is_causal is None and mask is not None:
    #     # 在这个分支里，类型检查器 100% 知道 is_causal 是 None
    #     ...
    #     if mask.size() == causal_comparison.size():
    #         is_causal = bool((mask == causal_comparison).all())
    #     else:
    #         is_causal = False
	# return is_causal  # 类型检查器推导的返回类型可能变得极度复杂
    make_causal = is_causal is True

    if is_causal is None and mask is not None:
        # 在矩阵乘法:Q X K^T中，结果矩阵的形状是 (..., L, S)；行数（-2 维）决定了有多少个 Query 需要进行掩码计算。
        # 在 90% 的传统任务中，Cross-Attention 不需要 Causal Mask。但 PyTorch 作为一个支持全世界开发者各种“魔改”的底层框架，必须假设有人会在 L != S 的 Cross-Attention 里玩出新花样（如流式多模态、Prefix 掩码）。
        # 使用 mask.size(-2)（Query 长度）是唯一能在数学逻辑和框架健壮性上同时闭环的选择。
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype
        )

        # Do not use `torch.equal` so we handle batched masks by   #历史注释问题
        # broadcasting the comparison.   
        # 如果 Cross-Attention 的 L != S，那么 mask.size() == causal_comparison.size() 确实必然返回 False，因为causal_comparison为方阵而mask不是。
        # mask在 PyTorch 底层加速算子的眼里，它不属于标准因果方阵，不能偷懒用硬件一键加速了，老老实实把这个非方阵的 mask 传入 CUDA 核心，做矩阵减法/加法吧。
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())   # mask == causal_comparison	逐元素比较，生成布尔张量(tensor)
        else:                                                       # tensor.all() 返回一个 0 维布尔张量（torch.bool 标量），表示张量中所有元素是否都为 True
            make_causal = False                                     # .all() 返回的是 torch.Tensor，不是 Python bool; bool() 显式转换为 Python 原生 bool 类型

    return make_causal
