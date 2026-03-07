"""
model.py — RKGCN (Ripple Knowledge Graph Convolutional Network) model in TensorFlow.

Two-module architecture:
  Module 1: User Preference Aggregation (Eq. 5-7)
  Module 2: Entity Enhancement via GCN (Eq. 8-10)
  Prediction: inner product + sigmoid (Eq. 11)
  Loss: cross-entropy + L2 regularization (Eq. 12)
"""

import tensorflow as tf
import numpy as np


class RKGCN(tf.keras.Model):
    """
    RKGCN: Ripple Knowledge Graph Convolutional Networks for Recommendation.

    Args:
      n_entities: number of entities in KG
      n_relations: number of relation types in KG
      dim: embedding dimension d
      n_hop: number of preference propagation hops H
      n_memory: preference set size per hop N_p
      n_neighbor: neighbor set size for GCN N_e
      kge_weight: weight for KGE loss
      l2_weight: L2 regularization weight lambda
      gcn_iter: number of GCN aggregation iterations
    """

    def __init__(
        self,
        n_entities,
        n_relations,
        dim,
        n_hop,
        n_memory,
        n_neighbor,
        kge_weight=0.01,
        l2_weight=1e-7,
        gcn_iter=1,
    ):
        super(RKGCN, self).__init__()

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.dim = dim
        self.n_hop = n_hop
        self.n_memory = n_memory
        self.n_neighbor = n_neighbor
        self.kge_weight = kge_weight
        self.l2_weight = l2_weight
        self.gcn_iter = gcn_iter

        # ---- Shared Embeddings ----
        self.entity_emb = tf.keras.layers.Embedding(
            n_entities, dim,
            embeddings_initializer=tf.keras.initializers.GlorotUniform(),
            name="entity_embedding",
        )
        self.relation_emb = tf.keras.layers.Embedding(
            n_relations, dim * dim,
            embeddings_initializer=tf.keras.initializers.GlorotUniform(),
            name="relation_embedding",
        )
        # Relation embedding for GCN (d-dimensional, not d*d)
        self.relation_emb_gcn = tf.keras.layers.Embedding(
            n_relations, dim,
            embeddings_initializer=tf.keras.initializers.GlorotUniform(),
            name="relation_embedding_gcn",
        )

        # ---- User Preference Aggregation (Eq. 7) ----
        self.transform_layers = []
        for hop in range(n_hop):
            self.transform_layers.append(
                tf.keras.layers.Dense(
                    dim,
                    activation="tanh",
                    name=f"transform_hop_{hop}",
                )
            )

        # ---- Entity Enhancement / GCN (Eq. 10) ----
        self.gcn_layers = []
        for i in range(gcn_iter):
            self.gcn_layers.append(
                tf.keras.layers.Dense(
                    dim,
                    activation="relu",
                    name=f"gcn_layer_{i}",
                )
            )

    def call(self, inputs, training=False):
        """
        Forward pass.

        Args:
          inputs: dict with keys:
            - items: (batch_size,) — candidate item entity IDs
            - memories_h: list of n_hop arrays, each (batch_size, n_memory)
            - memories_r: list of n_hop arrays, each (batch_size, n_memory)
            - memories_t: list of n_hop arrays, each (batch_size, n_memory)
            - neighbor_entities: (n_entities, n_neighbor) — precomputed
            - neighbor_relations: (n_entities, n_neighbor) — precomputed

        Returns:
          predictions: (batch_size,) — predicted click probabilities
          kge_loss: scalar — KG embedding loss
        """
        items = inputs["items"]                      # (batch_size,)
        memories_h = inputs["memories_h"]            # list of (batch_size, n_memory)
        memories_r = inputs["memories_r"]            # list of (batch_size, n_memory)
        memories_t = inputs["memories_t"]            # list of (batch_size, n_memory)
        neighbor_entities = inputs["neighbor_entities"]    # (n_entities, n_neighbor)
        neighbor_relations = inputs["neighbor_relations"]  # (n_entities, n_neighbor)

        # Get item embedding: (batch_size, dim)
        item_embeddings = self.entity_emb(items)

        # =============================================
        # Module 1: User Preference Aggregation
        # =============================================
        user_emb, kge_loss = self._user_preference_aggregation(
            item_embeddings, memories_h, memories_r, memories_t
        )

        # =============================================
        # Module 2: Entity Enhancement via GCN
        # =============================================
        item_enhanced = self._entity_enhancement(
            items, item_embeddings, user_emb,
            neighbor_entities, neighbor_relations
        )

        # =============================================
        # Prediction (Eq. 11): y_hat = sigma(i_h^T * o_n)
        # =============================================
        # user_emb: (batch_size, dim), item_enhanced: (batch_size, dim)
        logits = tf.reduce_sum(user_emb * item_enhanced, axis=1)  # (batch_size,)
        predictions = tf.sigmoid(logits)

        return predictions, kge_loss

    def _user_preference_aggregation(self, item_embeddings, memories_h, memories_r, memories_t):
        """
        User Preference Aggregation module (Eq. 5, 6, 7).

        Propagates user preferences through H hops of ripple sets,
        weighted by attention between item embedding and ripple set entities.
        """
        o_list = []
        kge_loss = 0.0
        v = item_embeddings  # (batch_size, dim) — updated each hop

        for hop in range(self.n_hop):
            # Get embeddings for this hop's ripple set
            h_emb = self.entity_emb(memories_h[hop])     # (batch_size, n_memory, dim)
            r_emb = self.relation_emb(memories_r[hop])   # (batch_size, n_memory, dim*dim)
            t_emb = self.entity_emb(memories_t[hop])     # (batch_size, n_memory, dim)

            # Reshape relation to matrix: (batch_size, n_memory, dim, dim)
            r_emb = tf.reshape(r_emb, [-1, self.n_memory, self.dim, self.dim])

            # --- Eq. 5: Attention score ---
            # a_i = Softmax(R * h * v^T)
            # Rh: (batch_size, n_memory, dim) — apply relation matrix to head
            # h_emb: (batch_size, n_memory, dim) -> expand -> (batch_size, n_memory, dim, 1)
            h_expanded = tf.expand_dims(h_emb, axis=3)  # (B, M, d, 1)
            Rh = tf.squeeze(tf.matmul(r_emb, h_expanded), axis=3)  # (B, M, d)

            # v: (batch_size, dim) -> (batch_size, 1, dim)
            v_expanded = tf.expand_dims(v, axis=1)  # (B, 1, d)

            # Dot product: (B, M)
            att_scores = tf.reduce_sum(Rh * v_expanded, axis=2)  # (B, M)
            att_weights = tf.nn.softmax(att_scores, axis=1)      # (B, M)

            # --- Eq. 6: Weighted aggregation ---
            # O_u = sum(a_i * t_i)
            att_weights_expanded = tf.expand_dims(att_weights, axis=2)  # (B, M, 1)
            o = tf.reduce_sum(att_weights_expanded * t_emb, axis=1)     # (B, d)
            o_list.append(o)

            # --- Eq. 7: Update item embedding ---
            # o_n = W(o_{n-1} + v)
            v = self.transform_layers[hop](o + v)  # (B, d)

            # --- KGE loss: ||h + r - t||^2 ---
            # Use the d-dimensional mean of r for KGE (simpler approximation)
            r_for_kge = tf.reduce_mean(
                tf.reshape(r_emb, [-1, self.n_memory, self.dim, self.dim]),
                axis=3
            )  # (B, M, d)
            kge_loss += tf.reduce_mean(
                tf.reduce_sum(tf.square(h_emb + r_for_kge - t_emb), axis=2)
            )

        # Final user embedding: use the last hop's output
        user_emb = v  # (batch_size, dim)

        return user_emb, kge_loss

    def _entity_enhancement(self, items, item_embeddings, user_emb,
                             neighbor_entities, neighbor_relations):
        """
        Entity Enhancement module via GCN (Eq. 8, 9, 10).

        Enriches item embeddings using KG neighborhood information,
        weighted by user-relation relevance scores.
        """
        # Get neighbor entity and relation IDs for batch items
        # items: (batch_size,)
        # neighbor_entities: (n_entities, n_neighbor)
        # neighbor_relations: (n_entities, n_neighbor)

        # Gather neighbors for batch items
        item_neighbor_entities = tf.gather(neighbor_entities, items)    # (B, N_e)
        item_neighbor_relations = tf.gather(neighbor_relations, items)  # (B, N_e)

        # Get neighbor entity embeddings: (B, N_e, d)
        neighbor_emb = self.entity_emb(item_neighbor_entities)

        # Get relation embeddings (d-dimensional for GCN): (B, N_e, d)
        neighbor_rel_emb = self.relation_emb_gcn(item_neighbor_relations)

        current_item_emb = item_embeddings  # (B, d)

        for gcn_layer in self.gcn_layers:
            # --- Eq. 8: Relation score ---
            # score_u^r = u * r^T
            # user_emb: (B, d) -> (B, 1, d)
            user_expanded = tf.expand_dims(user_emb, axis=1)  # (B, 1, d)

            # Dot product with relation embeddings: (B, N_e)
            relation_scores = tf.reduce_sum(
                user_expanded * neighbor_rel_emb, axis=2
            )  # (B, N_e)

            # --- Eq. 9: User-aware neighborhood aggregation ---
            # N_v^u = sum(softmax(score) * e)
            neighbor_weights = tf.nn.softmax(relation_scores, axis=1)  # (B, N_e)
            neighbor_weights = tf.expand_dims(neighbor_weights, axis=2)  # (B, N_e, 1)

            # Weighted sum of neighbor embeddings
            neighbor_agg = tf.reduce_sum(
                neighbor_weights * neighbor_emb, axis=1
            )  # (B, d)

            # --- Eq. 10: Combine with item embedding ---
            # i_h = W(v + N_v^u) + b
            current_item_emb = gcn_layer(current_item_emb + neighbor_agg)  # (B, d)

        return current_item_emb  # (B, d)

    def compute_loss(self, labels, predictions, kge_loss):
        """
        Total loss (Eq. 12):
          L = cross_entropy + kge_weight * kge_loss + l2_weight * ||params||^2
        """
        # Cross-entropy loss
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        rec_loss = bce(labels, predictions)

        # KGE loss
        kge_term = self.kge_weight * kge_loss

        # L2 regularization
        l2_loss = 0.0
        for var in self.trainable_variables:
            l2_loss += tf.nn.l2_loss(var)
        l2_term = self.l2_weight * l2_loss

        total_loss = rec_loss + kge_term + l2_term
        return total_loss, rec_loss, kge_term, l2_term
