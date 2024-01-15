# coding=utf-8
#
# created by kpe on 20.Mar.2019 at 16:30
#

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import params_flow as pf
from tensorflow.python import keras

# from params_flow import LayerNormalization
# from bert2.attention import AttentionLayer
# from bert2.layer import Layer



def gelu(x):
    """
    Gelu activation from arXiv:1606.08415.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * tf.pow(x, 3))
    ))
    return x * cdf


#########################################################################################################################
class Normalization(pf.Layer):
    class Params(pf.Layer.Params):
        name = "LayerNorm"

    def _construct(self, **kwargs):
        super()._construct(name=self.params.name, **kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask

class LayerNormalization(Normalization):
    """
    Layer normalization layer from arXiv:1607.06450.
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        See: https://github.com/CyberZHG/keras-layer-normalization
        See: tf.contrib.layers.layer_norm
    """
    class Params(Normalization.Params):
        epsilon         = 1e-12

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.gamma = None
        self.beta  = None
        self.supports_masking = True

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        input_spec = tf.keras.layers.InputSpec(shape=input_shape)
        self.gamma = self.add_weight(name="gamma", shape=input_shape[-1:],
                                     initializer=tf.keras.initializers.Ones(), trainable=True)
        self.beta  = self.add_weight(name="beta", shape=input_shape[-1:],
                                     initializer=tf.keras.initializers.Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):                               # pragma: no cover
        x = inputs
        if tf.__version__.startswith("2."):
            mean, var = tf.nn.moments(x, axes=-1, keepdims=True)
        else:
            mean, var = tf.nn.moments(x, axes=-1, keep_dims=True)

        #
        # this is how we would normalize, but
        #    it's commented out as it is not numerically equivalent
        #    to the tf.nn.batch_normalization implementation (used in BERT)
        #
        # normed = (x - mean)/tf.sqrt(var + self.params.epsilon)
        # res    = self.gamma * normed + self.beta
        # res = tf.nn.batch_normalization(x, mean=mean, variance=var,
        #                                 scale=self.gamma,
        #                                 offset=self.beta,
        #                                 variance_epsilon=self.params.epsilon)
        #

        # following two lines represent the tf.nn.batch_normalization implementation
        inv = self.gamma * tf.math.rsqrt(var + self.params.epsilon)
        res = x * tf.cast(inv, x.dtype) + tf.cast(self.beta - mean * inv, x.dtype)

        return res

#########################################################################################################################


class Layer(pf.Layer):
    """ Common abstract base layer for all BERT layers. """
    class Params(pf.Layer.Params):
        initializer_range = 0.02

    def create_initializer(self):
        return tf.keras.initializers.TruncatedNormal(stddev=self.params.initializer_range)

    @staticmethod
    def get_activation(activation_string):
        if not isinstance(activation_string, str):
            return activation_string
        if not activation_string:
            return None

        act = activation_string.lower()
        if act == "linear":
            return None
        elif act == "relu":
            return tf.nn.relu
        elif act == "gelu":
            return gelu
        elif act == "tanh":
            return tf.tanh
        elif act == "sigmoid":
            return tf.sigmoid
        else:
            raise ValueError("Unsupported activation: %s" % act)







#########################################################################################################################
class AttentionLayer(Layer):
    class Params(Layer.Params):
        num_heads         = None
        size_per_head     = None
        initializer_range = 0.02
        query_activation  = None
        key_activation    = None
        value_activation  = None
        attention_dropout = 0.1
        negative_infinity = -10000.0  # used for attention scores before softmax


    @staticmethod
    def create_attention_mask(from_shape, input_mask):
        """
        Creates 3D attention.
        :param from_shape:  [batch_size, from_seq_len, ...]
        :param input_mask:  [batch_size, seq_len]
        :return: [batch_size, from_seq_len, seq_len]
        """

        mask = tf.cast(tf.expand_dims(input_mask, axis=1), tf.float32)                   # [B, 1, T]
        ones = tf.expand_dims(tf.ones(shape=from_shape[:2], dtype=tf.float32), axis=-1)  # [B, F, 1]
        mask = ones * mask  # broadcast along two dimensions

        return mask  # [B, F, T]

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.query_activation = self.params.query_activation
        self.key_activation   = self.params.key_activation
        self.value_activation = self.params.value_activation

        self.query_layer = None
        self.key_layer   = None
        self.value_layer = None

        self.supports_masking = True

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        input_spec = keras.layers.InputSpec(shape=input_shape)

        dense_units = self.params.num_heads * self.params.size_per_head  # N*H
        #
        # B, F, T, N, H - batch, from_seq_len, to_seq_len, num_heads, size_per_head
        #
        self.query_layer = keras.layers.Dense(units=dense_units, activation=self.query_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="query")
        self.key_layer   = keras.layers.Dense(units=dense_units, activation=self.key_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="key")
        self.value_layer = keras.layers.Dense(units=dense_units, activation=self.value_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="value")
        self.dropout_layer = keras.layers.Dropout(self.params.attention_dropout)

        super(AttentionLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        from_shape = input_shape

        # from_shape         # [B, F, W]   [batch_size, from_seq_length, from_width] from_with = embedding_len(应该)
        # input_mask_shape   # [B, F]

        output_shape = [from_shape[0], from_shape[1], self.params.num_heads * self.params.size_per_head]
        attention_shape = [from_shape[0],self.params.num_heads,from_shape[1],from_shape[1]]
        return [output_shape,attention_shape]  # [B, F, N*H], [B, F, T]

    # noinspection PyUnusedLocal
    def call(self, inputs, mask=None, training=None, **kwargs):
        from_tensor = inputs
        to_tensor   = inputs
        if mask is None:
            sh = self.get_shape_list(from_tensor)
            mask = tf.ones(sh[:2], dtype=tf.int32)
        attention_mask = AttentionLayer.create_attention_mask(tf.shape(input=from_tensor), mask)

        #  from_tensor shape - [batch_size, from_seq_length, from_width]
        input_shape  = tf.shape(input=from_tensor)
        batch_size, from_seq_len, from_width = input_shape[0], input_shape[1], input_shape[2]
        to_seq_len = from_seq_len

        # [B, F, N*H] -> [B, N, F, H]
        def transpose_for_scores(input_tensor, seq_len):
            output_shape = [batch_size, seq_len,
                            self.params.num_heads, self.params.size_per_head]
            output_tensor = K.reshape(input_tensor, output_shape)
            return tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])  # [B,N,F,H]

        query = self.query_layer(from_tensor)  # [B,F, N*H] [batch_size, from_seq_len, N*H]
        key   = self.key_layer(to_tensor)      # [B,T, N*H]
        value = self.key_layer(to_tensor)    # [B,T, N*H]

        query = transpose_for_scores(query, from_seq_len)           # [B, N, F, H]
        key   = transpose_for_scores(key,   to_seq_len)             # [B, N, T, H]

        attention_scores = tf.matmul(query, key, transpose_b=True)  # [B, N, F, T]
        attention_scores = attention_scores / tf.sqrt(float(self.params.size_per_head))

        if attention_mask is not None:
            attention_mask = tf.expand_dims(attention_mask, axis=1)  # [B, 1, F, T]
            # {1, 0} -> {0.0, -inf}
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * self.params.negative_infinity
            attention_scores = tf.add(attention_scores, adder)  # adding to softmax -> its like removing them entirely

        # scores to probabilities
        attention_probs = tf.nn.softmax(attention_scores)           # [B, N, F, T]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout_layer(attention_probs,
                                             training=training)    # [B, N, F, T]

        # [B,T,N,H]
        value = tf.reshape(value, [batch_size, to_seq_len,
                                   self.params.num_heads, self.params.size_per_head])
        value = tf.transpose(a=value, perm=[0, 2, 1, 3])                                # [B, N, T, H]

        context_layer = tf.matmul(attention_probs, value)                               # [B, N, F, H]
        context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])                # [B, F, N, H]

        output_shape = [batch_size, from_seq_len,
                        self.params.num_heads * self.params.size_per_head]
        context_layer = tf.reshape(context_layer, output_shape)
        return [context_layer, attention_scores]                                         # [B, F, N*H]

    # noinspection PyUnusedLocal
    def compute_mask(self, inputs, mask=None):
        return mask   # [B, F]





#########################################################################################################################



class ProjectionLayer(Layer):
    class Params(Layer.Params):
        hidden_size        = None
        hidden_dropout     = 0.1
        initializer_range  = 0.02
        adapter_size       = None       # bottleneck size of the adapter - arXiv:1902.00751
        adapter_activation = "gelu"
        adapter_init_scale = 1e-3
    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.dense      = None
        self.dropout    = None
        self.layer_norm = None

        self.adapter_down = None
        self.adapter_up   = None

        self.supports_masking = True

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        assert isinstance(input_shape, list) and 2 == len(input_shape)
        out_shape, residual_shape = input_shape
        input_spec = [keras.layers.InputSpec(shape=out_shape), keras.layers.InputSpec(shape=residual_shape)]

        self.dense = keras.layers.Dense(units=self.params.hidden_size,
                                        kernel_initializer=self.create_initializer(),
                                        name="dense")
        self.dropout    = keras.layers.Dropout(rate=self.params.hidden_dropout)
        self.layer_norm = LayerNormalization(name="LayerNorm")

        if self.params.adapter_size is not None:
            self.adapter_down = keras.layers.Dense(units=self.params.adapter_size,
                                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                       stddev=self.params.adapter_init_scale),
                                                   activation=self.get_activation(self.params.adapter_activation),
                                                   name="adapter-down")
            self.adapter_up   = keras.layers.Dense(units=self.params.hidden_size,
                                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                       stddev=self.params.adapter_init_scale),
                                                   name="adapter-up")

        super(ProjectionLayer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):
        output, residual = inputs
        output = self.dense(output)
        output = self.dropout(output, training=training)

        if self.adapter_down is not None:
            adapted = self.adapter_down(output)
            adapted = self.adapter_up(adapted)
            output = tf.add(output, adapted)

        output = self.layer_norm(tf.add(output, residual))
        return output


class TransformerSelfAttentionLayer(Layer):
    class Params(ProjectionLayer.Params,
                 AttentionLayer.Params):
        hidden_size         = None
        num_heads           = None
        hidden_dropout      = None
        attention_dropout   = 0.1
        initializer_range   = 0.02

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        params = self.params
        if params.hidden_size % params.num_heads != 0:
            raise ValueError("The hidden_size:[{}] is not a multiple of num_heads:[{}]".format(params.hidden_size,
                                                                                               params.num_heads))
        self.size_per_head = params.hidden_size // params.num_heads
        assert params.size_per_head is None or self.size_per_head == params.size_per_head

        self.attention_layer     = None
        self.attention_projector = None
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        score_shape = [input_shape[0], self.params.num_heads, input_shape[1], input_shape[1]]
        return [input_shape,score_shape]

    def build(self, input_shape):
        input_spec = keras.layers.InputSpec(shape=input_shape)

        self.attention_layer = AttentionLayer.from_params(
            self.params,
            size_per_head=self.size_per_head,
            name="self_attention",
        )
        self.attention_projector = ProjectionLayer.from_params(
            self.params,
            name="output",
        )

        super(TransformerSelfAttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        layer_input = inputs

        #
        # TODO: is it OK to recompute the 3D attention mask in each attention layer
        #
        attention_head, score   = self.attention_layer(layer_input, mask=mask, training=training)
        attention_output = self.attention_projector([attention_head, layer_input], mask=mask, training=training)

        return [attention_output,score]


class SingleTransformerEncoderLayer(Layer):
    """
    Multi-headed, single layer for the Transformer from 'Attention is All You Need' (arXiv: 1706.03762).

    See also: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    """

    class Params(TransformerSelfAttentionLayer.Params,
                 ProjectionLayer.Params):
        intermediate_size       = None
        intermediate_activation = "gelu"

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        params = self.params
        if params.hidden_size % params.num_heads != 0:
            raise ValueError("The hidden_size:[{}] is not a multiple of num_heads:[{}]".format(params.hidden_size,
                                                                                               params.num_heads))
        self.size_per_head = params.hidden_size // params.num_heads

        self.self_attention_layer = None
        self.intermediate_layer   = None
        self.output_projector     = None

        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        score_shape = [input_shape[0],self.params.num_heads,input_shape[1],input_shape[1]]
        return [input_shape,score_shape]

    def build(self, input_shape):
        input_spec = keras.layers.InputSpec(shape=input_shape)  # [B, seq_len, hidden_size]

        self.self_attention_layer = TransformerSelfAttentionLayer.from_params(
            self.params,
            name="attention_layer"
        )
        self.intermediate_layer = keras.layers.Dense(
            name="intermediate",
            units=self.params.intermediate_size,
            activation=self.get_activation(self.params.intermediate_activation),
            kernel_initializer=self.create_initializer()
        )
        self.output_projector = ProjectionLayer.from_params(
            self.params,
            name="output",
        )

        super(SingleTransformerEncoderLayer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        layer_input = inputs

        attention_output, score = self.self_attention_layer(layer_input, mask=mask, training=training)

        # intermediate
        intermediate_output = self.intermediate_layer(attention_output)

        # output
        layer_output = self.output_projector([intermediate_output, attention_output], mask=mask)

        return [layer_output, score]


class TransformerEncoderLayer(Layer):
    """
    Multi-headed, multi-layer Transformer from 'Attention is All You Need' (arXiv: 1706.03762).

    Implemented for BERT, with support for ALBERT (sharing encoder layer params).

    See also: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    """

    class Params(SingleTransformerEncoderLayer.Params):
        num_layers     = None
        out_layer_ndxs = None   # [-1]

        shared_layer   = False  # False for BERT, True for ALBERT

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.encoder_layer   = None
        self.shared_layer     = None  # for ALBERT
        self.supports_masking = True

    def build(self, input_shape):
        input_spec = keras.layers.InputSpec(shape=input_shape)

        # create all transformer encoder sub-layers

        self.encoder_layer = SingleTransformerEncoderLayer.from_params(
            self.params,
            name="layer_STE",
        )

        super(TransformerEncoderLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        score_shape = [input_shape[0],self.params.num_heads,input_shape[1],input_shape[1]]
        return [input_shape,score_shape]

    def call(self, inputs, mask=None, training=None):
        final_output,final_score = self.encoder_layer(inputs, mask=mask, training=training)
        return [final_output,final_score]


