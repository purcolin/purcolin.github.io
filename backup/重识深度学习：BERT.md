## *本文为bert相关实操知识，模型结构等放到tranformer里讲。*

# 模型初始化

[官方文档](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel)&[官方api](https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/bert/modeling_bert.py#L952)&[config的api](https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/bert/configuration_bert.py#L29)

```
class transformers.BertModel(config : BertConfig , add_pooling_layer = True)
```
其中config为类`transformers.BertConfig`
主要的参数如下：
- **vocab_size** *(int, optional, defaults to 30522)* — BERT 模型的词表大小。定义了调用 BertModel 或 TFBertModel 时传递的 `inputs_ids` 可以表示的不同 token 的数量。
- **hidden_size** *(int, optional, defaults to 768)* — encoder层和池化层的维度大小。
- **num_hidden_layers** *(int, optional, defaults to 12)* — Transformer encoder中的隐藏层数量。
- **num_attention_heads** *(int, optional, defaults to 12)* — Transformer encoder中每个注意力层的注意力头数量。
- **intermediate_size** *(int, optional, defaults to 3072)* — Transformer encoder中“中间层”（常称为前馈层）的维度大小。
- **hidden_act** *(str or Callable, optional, defaults to "gelu")* — 编码器和池化层中的非线性激活函数（函数或字符串）。如果是字符串，可以选 "gelu"、"relu"、"silu" 和 "gelu_new"。
- **hidden_dropout_prob** *(float, optional, defaults to 0.1)* — embedding、encoder和poller中所有全连接层的 dropout 概率。
- **attention_probs_dropout_prob** *(float, optional, defaults to 0.1)* — 注意力概率的 dropout 比例。
- **max_position_embeddings** *(int, optional, defaults to 512)* — 该模型可能使用的最大序列长度。通常设置为一个较大的数值以防万一（例如，512、1024 或 2048）。
- **type_vocab_size** *(int, optional, defaults to 2)* — 调用 BertModel 或 TFBertModel 时传递的 token_type_ids 的词汇表大小。
- **initializer_range** *(float, optional, defaults to 0.02)* — 用于初始化所有权重矩阵的 `truncated_normal_initializer` 的标准差。
- **layer_norm_eps** *(float, optional, defaults to 1e-12)* — layer normalization层使用的 epsilon 值。
- **position_embedding_type** *(str, optional, defaults to "absolute")* — position embedding的类型。选择 "absolute"、"relative_key"、"relative_key_query" 之一。对于positional embedding，使用 "absolute"。
>有关 "relative_key" 的更多信息，请参见《Self-Attention with Relative Position Representations (Shaw et al.)》。有关 "relative_key_query" 的更多信息，请参见《Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)》中的方法 4。
- **is_decoder** *(bool, optional, defaults to False)* — 模型是否用作解码器。如果为 False，模型用作编码器。
- **use_cache** *(bool, optional, defaults to True)* — 模型是否应返回上一个键/值的注意力（并非所有模型都使用）。仅在 `config.is_decoder=True` 时相关。
- **classifier_dropout** *(float, optional)* — 分类头的 dropout 比例。


# 模型输出

BertModel的forward()类覆写了__call__()方法，因此调用BertModel(**inputs)即会返回输出。

输入：

- input_ids: Optional[torch.Tensor] = None,
- attention_mask: Optional[torch.Tensor] = None,
- token_type_ids: Optional[torch.Tensor] = None,
- position_ids: Optional[torch.Tensor] = None,
- head_mask: Optional[torch.Tensor] = None,
- inputs_embeds: Optional[torch.Tensor] = None,
- encoder_hidden_states: Optional[torch.Tensor] = None,
- encoder_attention_mask: Optional[torch.Tensor] = None,
-  past_key_values: Optional[List[torch.FloatTensor]] = None,
- use_cache: Optional[bool] = None,
-  output_attentions: Optional[bool] = None,
- output_hidden_states: Optional[bool] = None,
-  return_dict: Optional[bool] = None,


## 如果return_dict=True
返回一个[BaseModelOutputWithPoolingAndCrossAttentions](https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions)*
-  last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):Sequence of hidden-states at the output of the last layer of the model.
-  pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
-  hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
 Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
-  attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
 - cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
  - past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)` and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
 Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding. 
## 如果return_dict=False
|位置|描述|shape|
|---|---|---|
|output[0]| 最后一层的隐藏状态 |（batch_size, sequence_length, hidden_size)|
|output[1]| 第一个token即（cls）最后一层的隐藏状态 |(batch_size, hidden_size)| 
|output[2]| 需要指定 output_hidden_states = True， 包含所有隐藏状态，第一个元素是embedding, 其余元素是各层的输出 |(batch_size, sequence_length, hidden_size)
|output[3]| 需要指定output_attentions=True，包含每一层的注意力权重，用于计算self-attention heads的加权平均值|(batch_size, layer_nums, sequence_length, sequence_legth)