import tensorflow as tf
import numpy as np


class LightGCN(tf.keras.Model):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.
    """
    def __init__(self, n_users, n_items, dim, n_layers=3, l2_weight=1e-7):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.n_layers = n_layers
        self.l2_weight = l2_weight

        self.user_emb = tf.keras.layers.Embedding(
            n_users, dim,
            embeddings_initializer=tf.keras.initializers.GlorotUniform(),
            name="user_embedding"
        )
        self.item_emb = tf.keras.layers.Embedding(
            n_items, dim,
            embeddings_initializer=tf.keras.initializers.GlorotUniform(),
            name="item_embedding"
        )

    def call(self, inputs, training=False):
        user_ids = inputs["users"]
        item_ids = inputs["items"]
        adj_matrix = inputs["adj_matrix"]

        # Get all embeddings
        user_embeddings = self.user_emb(tf.range(self.n_users))
        item_embeddings = self.item_emb(tf.range(self.n_items))
        all_embeddings = tf.concat([user_embeddings, item_embeddings], axis=0)

        embs = [all_embeddings]
        for _ in range(self.n_layers):
            all_embeddings = tf.sparse.sparse_dense_matmul(adj_matrix, all_embeddings)
            embs.append(all_embeddings)

        # Average embeddings across all layers
        final_embeddings = tf.reduce_mean(tf.stack(embs, axis=0), axis=0)

        final_user_embeddings = final_embeddings[:self.n_users]
        final_item_embeddings = final_embeddings[self.n_users:]

        u_emb = tf.gather(final_user_embeddings, user_ids)
        i_emb = tf.gather(final_item_embeddings, item_ids)

        logits = tf.reduce_sum(u_emb * i_emb, axis=1)
        predictions = tf.sigmoid(logits)

        return predictions

    def compute_loss(self, labels, predictions):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        rec_loss = bce(labels, predictions)

        # L2 regularization
        l2_loss = 0.0
        for var in self.trainable_variables:
            l2_loss += tf.nn.l2_loss(var)
        l2_term = self.l2_weight * l2_loss

        return rec_loss + l2_term, rec_loss, l2_term


class NGCF(tf.keras.Model):
    """
    NGCF: Neural Graph Collaborative Filtering.
    """
    def __init__(self, n_users, n_items, dim, n_layers=3, l2_weight=1e-7):
        super(NGCF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.n_layers = n_layers
        self.l2_weight = l2_weight

        self.user_emb = tf.keras.layers.Embedding(
            n_users, dim,
            embeddings_initializer=tf.keras.initializers.GlorotUniform(),
            name="user_embedding"
        )
        self.item_emb = tf.keras.layers.Embedding(
            n_items, dim,
            embeddings_initializer=tf.keras.initializers.GlorotUniform(),
            name="item_embedding"
        )

        self.w1_layers = []
        self.w2_layers = []
        for _ in range(n_layers):
            self.w1_layers.append(tf.keras.layers.Dense(dim, activation=None))
            self.w2_layers.append(tf.keras.layers.Dense(dim, activation=None))

    def call(self, inputs, training=False):
        user_ids = inputs["users"]
        item_ids = inputs["items"]
        adj_matrix = inputs["adj_matrix"]

        user_embeddings = self.user_emb(tf.range(self.n_users))
        item_embeddings = self.item_emb(tf.range(self.n_items))
        all_embeddings = tf.concat([user_embeddings, item_embeddings], axis=0)

        embs = [all_embeddings]
        for l in range(self.n_layers):
            # Neighbor aggregation
            side_embeddings = tf.sparse.sparse_dense_matmul(adj_matrix, all_embeddings)

            # W1 * (e_i + e_neighbor)
            sum_embeddings = all_embeddings + side_embeddings
            t1 = self.w1_layers[l](sum_embeddings)

            # W2 * (e_i * e_neighbor)
            mul_embeddings = all_embeddings * side_embeddings
            t2 = self.w2_layers[l](mul_embeddings)

            all_embeddings = tf.nn.leaky_relu(t1 + t2)
            embs.append(all_embeddings)

        # Concatenate layer embeddings
        final_embeddings = tf.concat(embs, axis=1)

        final_user_embeddings = final_embeddings[:self.n_users]
        final_item_embeddings = final_embeddings[self.n_users:]

        u_emb = tf.gather(final_user_embeddings, user_ids)
        i_emb = tf.gather(final_item_embeddings, item_ids)

        logits = tf.reduce_sum(u_emb * i_emb, axis=1)
        predictions = tf.sigmoid(logits)

        return predictions

    def compute_loss(self, labels, predictions):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        rec_loss = bce(labels, predictions)

        # L2 regularization
        l2_loss = 0.0
        for var in self.trainable_variables:
            l2_loss += tf.nn.l2_loss(var)
        l2_term = self.l2_weight * l2_loss

        return rec_loss + l2_term, rec_loss, l2_term


class SASRec(tf.keras.Model):
    """
    SASRec: Self-Attentive Sequential Recommendation.
    """
    def __init__(self, n_items, dim, max_len=50, num_heads=2, l2_weight=1e-7):
        super(SASRec, self).__init__()
        self.n_items = n_items  # n_items is also the padding index
        self.dim = dim
        self.max_len = max_len
        self.l2_weight = l2_weight

        # Add 1 to n_items for padding embedding (index = n_items)
        self.item_emb = tf.keras.layers.Embedding(
            n_items + 1, dim,
            embeddings_initializer=tf.keras.initializers.GlorotUniform(),
            name="item_embedding"
        )
        self.pos_emb = tf.keras.layers.Embedding(
            max_len, dim,
            embeddings_initializer=tf.keras.initializers.GlorotUniform(),
            name="position_embedding"
        )

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim)
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dim, activation="relu"),
            tf.keras.layers.Dense(dim)
        ])
        self.layernorm2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training=False):
        seq_ids = inputs["sequences"]  # (batch_size, max_len)
        target_ids = inputs["items"]   # (batch_size,)

        # Embed sequences and add positions
        seq_emb = self.item_emb(seq_ids)  # (batch_size, max_len, dim)
        positions = tf.range(self.max_len)
        pos_emb = self.pos_emb(positions)  # (max_len, dim)
        pos_emb = tf.expand_dims(pos_emb, axis=0)  # (1, max_len, dim)
        x = seq_emb + pos_emb

        # Self-attention with causal mask
        attn_out = self.mha(x, x, use_causal_mask=True, training=training)
        x = self.layernorm1(x + attn_out, training=training)

        # FFN
        ffn_out = self.ffn(x, training=training)
        x = self.layernorm2(x + ffn_out, training=training)

        # Last item representation as user embedding
        u_emb = x[:, -1, :]  # (batch_size, dim)

        # Target item embedding
        i_emb = self.item_emb(target_ids)  # (batch_size, dim)

        logits = tf.reduce_sum(u_emb * i_emb, axis=1)
        predictions = tf.sigmoid(logits)

        return predictions

    def compute_loss(self, labels, predictions):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        rec_loss = bce(labels, predictions)

        # L2 regularization
        l2_loss = 0.0
        for var in self.trainable_variables:
            l2_loss += tf.nn.l2_loss(var)
        l2_term = self.l2_weight * l2_loss

        return rec_loss + l2_term, rec_loss, l2_term


class BERT4Rec(tf.keras.Model):
    """
    BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformers.
    """
    def __init__(self, n_items, dim, max_len=50, num_heads=2, l2_weight=1e-7):
        super(BERT4Rec, self).__init__()
        self.n_items = n_items  # n_items is also the padding index
        self.mask_token = n_items + 1  # mask token is index n_items + 1
        self.dim = dim
        self.max_len = max_len
        self.l2_weight = l2_weight

        # Add 2 to n_items for padding (index = n_items) and mask (index = n_items + 1)
        self.item_emb = tf.keras.layers.Embedding(
            n_items + 2, dim,
            embeddings_initializer=tf.keras.initializers.GlorotUniform(),
            name="item_embedding"
        )
        self.pos_emb = tf.keras.layers.Embedding(
            max_len, dim,
            embeddings_initializer=tf.keras.initializers.GlorotUniform(),
            name="position_embedding"
        )

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim)
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dim, activation="relu"),
            tf.keras.layers.Dense(dim)
        ])
        self.layernorm2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training=False):
        seq_ids = inputs["sequences"]  # (batch_size, max_len)
        target_ids = inputs["items"]   # (batch_size,)

        # In BERT4Rec, for CTR target item prediction:
        # We append a [MASK] token at the end of the sequence to predict the target item
        # Shift sequence left by 1 and insert mask token at the end
        mask_tokens = tf.fill([tf.shape(seq_ids)[0], 1], self.mask_token)
        masked_seq_ids = tf.concat([seq_ids[:, 1:], mask_tokens], axis=1)

        # Embed sequences and add positions
        seq_emb = self.item_emb(masked_seq_ids)  # (batch_size, max_len, dim)
        positions = tf.range(self.max_len)
        pos_emb = self.pos_emb(positions)  # (max_len, dim)
        pos_emb = tf.expand_dims(pos_emb, axis=0)  # (1, max_len, dim)
        x = seq_emb + pos_emb

        # Bidirectional self-attention (no causal mask)
        attn_out = self.mha(x, x, use_causal_mask=False, training=training)
        x = self.layernorm1(x + attn_out, training=training)

        # FFN
        ffn_out = self.ffn(x, training=training)
        x = self.layernorm2(x + ffn_out, training=training)

        # Mask token representation as user embedding
        u_emb = x[:, -1, :]  # (batch_size, dim)

        # Target item embedding
        i_emb = self.item_emb(target_ids)  # (batch_size, dim)

        logits = tf.reduce_sum(u_emb * i_emb, axis=1)
        predictions = tf.sigmoid(logits)

        return predictions

    def compute_loss(self, labels, predictions):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        rec_loss = bce(labels, predictions)

        # L2 regularization
        l2_loss = 0.0
        for var in self.trainable_variables:
            l2_loss += tf.nn.l2_loss(var)
        l2_term = self.l2_weight * l2_loss

        return rec_loss + l2_term, rec_loss, l2_term
