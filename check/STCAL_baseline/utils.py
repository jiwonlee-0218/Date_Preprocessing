from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras import backend as K
import params_flow as pf
from params_flow.activations import gelu


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

class MyAttention(keras.layers.Layer):
    """ Keras layer for Mutiply a Tensor to be the same shape as another Tensor.
    """

    def __init__(self, num_heads=None, size_per_head=None, **kwargs):
        self.num_heads = num_heads
        self.size_per_head = size_per_head
        super(MyAttention, self).__init__(**kwargs)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'size_per_head': self.size_per_head 
        })
        return config

    def call(self, inputs,training=None,**kwargs):
        value, attention_probs = inputs
        input_shape = tf.shape(input=value)
        batch_size, from_seq_len = input_shape[0], input_shape[1]
        to_seq_len = from_seq_len
        value = tf.reshape(value, [batch_size, to_seq_len,
                                   self.num_heads, self.size_per_head])
        value = tf.transpose(a=value, perm=[0, 2, 1, 3])  # [B, N, T, H]
        context_layer = tf.matmul(attention_probs, value)  # [B, N, F, H]
        context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])  # [B, F, N, H]
        output_shape = [batch_size, from_seq_len,
                        self.num_heads * self.size_per_head]
        context_layer = tf.reshape(context_layer, output_shape)
        return context_layer

    def compute_output_shape(self, input_shape):
        from_shape = input_shape
        output_shape = [from_shape[0], from_shape[1], self.num_heads * self.size_per_head]
        return output_shape  # [B, F, N*H], [B, F, T]

class CoAttention(keras.layers.Layer):
    """ Keras layer for Mutiply a Tensor to be the same shape as another Tensor.
    """

    def __init__(self, num_heads=None, size_per_head=None,attention_dropout=None, **kwargs):
        self.num_heads = num_heads
        self.size_per_head = size_per_head
        self.attention_dropout = attention_dropout
        super(CoAttention, self).__init__(**kwargs)


    def call(self, inputs, **kwargs):
        query, key, value = inputs
        input_shape = tf.shape(input=value)
        batch_size, from_seq_len = input_shape[0], input_shape[1]
        to_seq_len = from_seq_len

        def transpose_for_scores(input_tensor, seq_len):
            output_shape = [batch_size, seq_len,
                            self.num_heads, self.size_per_head]
            output_tensor = K.reshape(input_tensor, output_shape)
            return tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])  # [B,N,F,H]

        query = transpose_for_scores(query, from_seq_len)  # [B, N, F, H]
        key = transpose_for_scores(key, to_seq_len)  # [B, N, T, H]

        attention_scores = tf.matmul(query, key, transpose_b=True)  # [B, N, F, T]
        attention_scores = attention_scores / tf.sqrt(float(self.size_per_head))
        attention_probs = tf.nn.softmax(attention_scores)           # [B, N, F, T]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = keras.layers.Dropout(self.attention_dropout)(attention_probs)    # [B, N, F, T]

        value = tf.reshape(value, [batch_size, to_seq_len,
                                   self.num_heads, self.size_per_head])
        value = tf.transpose(a=value, perm=[0, 2, 1, 3])  # [B, N, T, H]
        context_layer = tf.matmul(attention_probs, value)  # [B, N, F, H]
        context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])  # [B, F, N, H]
        output_shape = [batch_size, from_seq_len,
                        self.num_heads * self.size_per_head]
        context_layer = tf.reshape(context_layer, output_shape)
        return context_layer

    def compute_output_shape(self, input_shape):
        from_shape = input_shape
        output_shape = [from_shape[0], from_shape[1], self.num_heads * self.size_per_head]
        return output_shape  # [B, F, N*H], [B, F, T]

class PositionEmbeddingLayer(Layer):
    class Params(Layer.Params):
        max_position_embeddings  = 512
        hidden_size              = 190

    # noinspection PyUnusedLocal
    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.embedding_table = None

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        # input_shape: () of seq_len
        if input_shape is not None:
            assert input_shape.ndims == 0
            self.input_spec = keras.layers.InputSpec(shape=input_shape, dtype='int32')
        else:
            self.input_spec = keras.layers.InputSpec(shape=(), dtype='int32')

        self.embedding_table = self.add_weight(name="embeddings",
                                               dtype=K.floatx(),
                                               shape=[self.params.max_position_embeddings, self.params.hidden_size],
                                               initializer=self.create_initializer())
        super(PositionEmbeddingLayer, self).build(input_shape)

    # noinspection PyUnusedLocal
    def call(self, inputs, **kwargs):
        # just return the embedding after verifying
        # that seq_len is less than max_position_embeddings

        seq_len = inputs

        assert_op = tf.compat.v2.debugging.assert_less_equal(seq_len, self.params.max_position_embeddings)

        with tf.control_dependencies([assert_op]):
            # slice to seq_len
            full_position_embeddings = tf.slice(self.embedding_table,
                                                [0, 0],
                                                [seq_len, -1])
        output = full_position_embeddings
        return output



class EmbeddingsProjector(Layer):
    class Params(Layer.Params):
        hidden_size                  = 190
        embedding_size               = None   # None for BERT, not None for ALBERT
        project_embeddings_with_bias = True   # in ALBERT - True for Google, False for brightmart/albert_zh

    # noinspection PyUnusedLocal
    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.projector_layer      = None   # for ALBERT
        self.projector_bias_layer = None   # for ALBERT

    def build(self, input_shape):
        emb_shape = input_shape
        self.input_spec = keras.layers.InputSpec(shape=emb_shape)
        assert emb_shape[-1] == self.params.embedding_size

        # ALBERT word embeddings projection
        self.projector_layer = self.add_weight(name="projector",
                                               shape=[self.params.embedding_size,
                                                      self.params.hidden_size],
                                               dtype=K.floatx())
        if self.params.project_embeddings_with_bias:
            self.projector_bias_layer = self.add_weight(name="bias",
                                                        shape=[self.params.hidden_size],
                                                        dtype=K.floatx())
        super(EmbeddingsProjector, self).build(input_shape)

    def call(self, inputs, **kwargs):
        input_embedding = inputs
        assert input_embedding.shape[-1] == self.params.embedding_size

        # ALBERT: project embedding to hidden_size
        output = tf.matmul(input_embedding, self.projector_layer)
        if self.projector_bias_layer is not None:
            output = tf.add(output, self.projector_bias_layer)

        return output


class BertEmbeddingsLayer(Layer):
    class Params(PositionEmbeddingLayer.Params,
                 EmbeddingsProjector.Params):
        vocab_size               = None
        use_token_type           = None
        use_position_embeddings  = True
        token_type_vocab_size    = 0 # segment_ids类别 [0,1]
        hidden_size              = 190
        hidden_dropout           = 0.1

        extra_tokens_vocab_size  = None  # size of the extra (task specific) token vocabulary (using negative token ids)

        #
        # ALBERT support - set embedding_size (or None for BERT)
        #
        embedding_size               = None   # None for BERT, not None for ALBERT
        project_embeddings_with_bias = False   # in ALBERT - True for Google, False for brightmart/albert_zh
        project_position_embeddings  = False   # in ALEBRT - True for Google, False for brightmart/albert_zh

        mask_zero                    = False

    # noinspection PyUnusedLocal
    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.word_embeddings_layer       = None
        self.extra_word_embeddings_layer = None   # for task specific tokens (negative token ids)
        self.token_type_embeddings_layer = None
        self.position_embeddings_layer   = None
        self.word_embeddings_projector_layer = None   # for ALBERT
        self.layer_norm_layer = None
        self.dropout_layer    = None

        self.support_masking = self.params.mask_zero

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if isinstance(input_shape, list):
            assert len(input_shape) == 2
            input_ids_shape, token_type_ids_shape = input_shape
            self.input_spec = [keras.layers.InputSpec(shape=input_ids_shape),
                               keras.layers.InputSpec(shape=token_type_ids_shape)]
        else:
            input_ids_shape = input_shape
            self.input_spec = keras.layers.InputSpec(shape=input_ids_shape)

        # use either hidden_size for BERT or embedding_size for ALBERT
        embedding_size = self.params.hidden_size if self.params.embedding_size is None else self.params.embedding_size

        self.word_embeddings_layer = keras.layers.Embedding(
            input_dim=self.params.vocab_size,
            output_dim=embedding_size,
            mask_zero=self.params.mask_zero,
            name="word_embeddings"
        )
        if self.params.extra_tokens_vocab_size is not None:
            self.extra_word_embeddings_layer = keras.layers.Embedding(
                input_dim=self.params.extra_tokens_vocab_size + 1,  # +1 is for a <pad>/0 vector
                output_dim=embedding_size,
                mask_zero=self.params.mask_zero,
                embeddings_initializer=self.create_initializer(),
                name="extra_word_embeddings"
            )

        # ALBERT word embeddings projection
        if self.params.embedding_size is not None:
            self.word_embeddings_projector_layer = EmbeddingsProjector.from_params(
                self.params, name="word_embeddings_projector")

        position_embedding_size = embedding_size if self.params.project_position_embeddings else self.params.hidden_size

        if self.params.use_token_type:
            self.token_type_embeddings_layer = keras.layers.Embedding(
                input_dim=self.params.token_type_vocab_size,
                output_dim=position_embedding_size,
                mask_zero=False,
                name="token_type_embeddings"
            )
        if self.params.use_position_embeddings:
            self.position_embeddings_layer = PositionEmbeddingLayer.from_params(
                self.params,
                name="position_embeddings",
                hidden_size=position_embedding_size
            )

        self.layer_norm_layer = pf.LayerNormalization(name="LayerNorm")
        self.dropout_layer    = keras.layers.Dropout(rate=self.params.hidden_dropout)

        super(BertEmbeddingsLayer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids      = inputs
            token_type_ids = None

        input_ids = tf.cast(input_ids, dtype=tf.float32)

        if self.extra_word_embeddings_layer is not None:
            token_mask   = tf.cast(tf.greater_equal(input_ids, 0), tf.int32)
            extra_mask   = tf.cast(tf.less(input_ids, 0), tf.int32)
            token_ids    = token_mask * input_ids
            extra_tokens = extra_mask * (-input_ids)
            token_output = self.word_embeddings_layer(token_ids)
            extra_output = self.extra_word_embeddings_layer(extra_tokens)
            embedding_output = tf.add(token_output,
                                      extra_output * tf.expand_dims(tf.cast(extra_mask, K.floatx()), axis=-1))
        else:
            embedding_output = input_ids # 这里做了相应的修改，去掉了word_embedding
            # embedding_output = self.word_embeddings_layer(input_ids)

        # ALBERT: for brightmart/albert_zh weights - project only token embeddings
        if not self.params.project_position_embeddings:
            if self.word_embeddings_projector_layer:
                embedding_output = self.word_embeddings_projector_layer(embedding_output)

        if token_type_ids is not None:
            token_type_ids    = tf.cast(token_type_ids, dtype=tf.int32)
            embedding_output += self.token_type_embeddings_layer(token_type_ids)

        if self.position_embeddings_layer is not None:
            seq_len  = input_ids.shape.as_list()[1]
            emb_size = embedding_output.shape[-1]

            pos_embeddings = self.position_embeddings_layer(seq_len)
            # broadcast over all dimension except the last two [..., seq_len, width]
            broadcast_shape = [1] * (embedding_output.shape.ndims - 2) + [seq_len, emb_size]
            embedding_output += tf.reshape(pos_embeddings, broadcast_shape)

        embedding_output = self.layer_norm_layer(embedding_output)
        embedding_output = self.dropout_layer(embedding_output, training=training)

        # ALBERT: for google-research/albert weights - project all embeddings
        if self.params.project_position_embeddings:
            if self.word_embeddings_projector_layer:
                embedding_output = self.word_embeddings_projector_layer(embedding_output)

        return embedding_output   # [B, seq_len, hidden_size]

    def compute_mask(self, inputs, mask=None):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids      = inputs
            token_type_ids = None

        if not self.support_masking:
            return None

        return tf.not_equal(input_ids, 0)

class ScoreEmbedding(Layer):
    class Params(Layer.Params):
        max_position_embeddings  = 512
        hidden_size              = 190

    # noinspection PyUnusedLocal
    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.embedding_table = None

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        # input_shape: () of seq_len
        if input_shape is not None:
            assert input_shape.ndims == 0
            self.input_spec = keras.layers.InputSpec(shape=input_shape, dtype='int32')
        else:
            self.input_spec = keras.layers.InputSpec(shape=(), dtype='int32')

        self.embedding_table = self.add_weight(name="embeddings",
                                               dtype=K.floatx(),
                                               shape=[10,self.params.max_position_embeddings, self.params.hidden_size],
                                               initializer=self.create_initializer())
        super(ScoreEmbedding, self).build(input_shape)

    # noinspection PyUnusedLocal
    def call(self, inputs, **kwargs):
        # just return the embedding after verifying
        # that seq_len is less than max_position_embeddings


        # slice to seq_len
        full_position_embeddings = tf.slice(self.embedding_table,
                                                [0, 0, 0],
                                                [10, self.params.hidden_size, -1])
        output = full_position_embeddings
        return output


class ScoreEmbeddingLayer(Layer):
    class Params(ScoreEmbedding.Params):
        vocab_size               = None
        use_token_type           = None
        use_position_embeddings  = True
        hidden_size              = 190
        hidden_dropout           = 0.1

    # noinspection PyUnusedLocal
    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.position_embeddings_layer   = None
        self.layer_norm_layer = None
        self.dropout_layer    = None

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if isinstance(input_shape, list):
            assert len(input_shape) == 2
            input_ids_shape, token_type_ids_shape = input_shape
            self.input_spec = [keras.layers.InputSpec(shape=input_ids_shape),
                               keras.layers.InputSpec(shape=token_type_ids_shape)]
        else:
            input_ids_shape = input_shape
            self.input_spec = keras.layers.InputSpec(shape=input_ids_shape)

        # use either hidden_size for BERT or embedding_size for ALBERT

        self.layer_norm_layer = pf.LayerNormalization(name="LayerNorm")
        self.dropout_layer    = keras.layers.Dropout(rate=self.params.hidden_dropout)

        super(ScoreEmbeddingLayer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids      = inputs

        input_ids = tf.cast(input_ids, dtype=tf.float32)


        embedding_output = input_ids # 这里做了相应的修改，去掉了word_embedding

        if self.position_embeddings_layer is not None:
            seq_len  = input_ids.shape.as_list()[2]

            pos_embeddings = self.position_embeddings_layer(seq_len)
            embedding_output += pos_embeddings

        # embedding_output = self.layer_norm_layer(embedding_output)
        embedding_output = self.dropout_layer(embedding_output, training=training)

        return embedding_output   # [B, seq_len, hidden_size]