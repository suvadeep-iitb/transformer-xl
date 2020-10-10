import tensorflow as tf


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, demb, **kwargs):
        super().__init__(**kwargs)

        self.inv_freq = 1 / (10000 ** (tf.range(0, demb, 2.0) / demb))

    def call(self, pos_seq, bsz=None):
        sinusoid_inp = tf.einsum("i,j->ij", pos_seq, self.inv_freq)
        pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)

        if bsz is not None:
            return tf.tile(pos_emb[:, None, :], [1, bsz, 1])
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(tf.keras.layers.Layer):
    def __init__(self, d_model, d_inner, dropout, kernel_initializer,
                 pre_lnorm=False, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.layer_1 = tf.keras.layers.Dense(
            d_inner, kernel_initializer=kernel_initializer, activation=tf.nn.relu, name='layer_1'
        )
        self.drop_1 = tf.keras.layers.Dropout(dropout, name='drop_1')
        self.layer_2 = tf.keras.layers.Dense(d_model, kernel_initializer=kernel_initializer, name='layer_2')
        self.drop_2 = tf.keras.layers.Dropout(dropout, name='drop_2')

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm')
        self.pre_lnorm = pre_lnorm

    def call(self, inp, training=False):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.layer_norm(inp)
            core_out = self.layer_1(core_out)
            core_out = self.drop_1(core_out, training=training)
            core_out = self.layer_2(core_out)
            core_out = self.drop_2(core_out, training=training)

            output = [core_out + inp]
        else:
            # positionwise feed-forward
            core_out = self.layer_1(inp)
            core_out = self.drop_1(core_out, training=training)
            core_out = self.layer_2(core_out)
            core_out = self.drop_2(core_out, training=training)

            output = [self.layer_norm(inp + core_out)]

        return output


class RelativeMultiHeadAttn(tf.keras.layers.Layer):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt,
        kernel_initializer,
        pre_lnorm=False,
        r_r_bias=None,
        r_w_bias=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.kernel_initializer=kernel_initializer

        self.qkv_net = tf.keras.layers.Dense(
            3 * n_head * d_head, kernel_initializer=kernel_initializer, use_bias=False, name="qkv"
        )
        self.r_net = tf.keras.layers.Dense(
            self.n_head * self.d_head, kernel_initializer=kernel_initializer, use_bias=False, name="r"
        )
        self.drop = tf.keras.layers.Dropout(dropout)
        self.dropatt = tf.keras.layers.Dropout(dropatt)
        self.o_net = tf.keras.layers.Dense(
            d_model, kernel_initializer=kernel_initializer, use_bias=False, name="o"
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.scale = 1 / (d_head ** 0.5)

        if r_r_bias is not None and r_w_bias is not None:  # Biases are shared
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias
        else:
            self.r_r_bias = self.add_weight(
                shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_r_bias"
            )
            self.r_w_bias = self.add_weight(
                shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_w_bias"
            )

        self.pre_lnorm = pre_lnorm

    def _rel_shift(self, x):
        x_size = shape_list(x)

        x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
        x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
        x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
        x = tf.reshape(x, x_size)

        return x

    def call(self, inputs, training=False):
        w, r, attn_mask, mems = inputs
        qlen, rlen, bsz = shape_list(w)[0], shape_list(r)[0], shape_list(w)[1]

        if mems is not None:
            cat = tf.concat([mems, w], 0)
        else:
            cat = w
        
        if self.pre_lnorm:
            cat = self.layer_norm(cat)

        w_heads = self.qkv_net(cat)
        r_head_k = self.r_net(r)

        w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, axis=-1)
        w_head_q = w_head_q[-qlen:]

        klen = shape_list(w_head_k)[0]

        w_head_q = tf.reshape(w_head_q, (qlen, bsz, self.n_head, self.d_head))
        w_head_k = tf.reshape(w_head_k, (klen, bsz, self.n_head, self.d_head))
        w_head_v = tf.reshape(w_head_v, (klen, bsz, self.n_head, self.d_head))

        r_head_k = tf.reshape(r_head_k, (rlen, self.n_head, self.d_head))

        rw_head_q = w_head_q + self.r_w_bias
        rr_head_q = w_head_q + self.r_r_bias

        AC = tf.einsum("ibnd,jbnd->ijbn", rw_head_q, w_head_k)
        BD = tf.einsum("ibnd,jnd->ijbn", rr_head_q, r_head_k)
        BD = self._rel_shift(BD)

        attn_score = AC + BD
        attn_score = attn_score * self.scale

        attn_mask_t = attn_mask[:, :, None, None]
        attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

        attn_prob = tf.nn.softmax(attn_score, axis=1)
        attn_prob = self.dropatt(attn_prob, training=training)

        attn_vec = tf.einsum("ijbn,jbnd->ibnd", attn_prob, w_head_v)
        size_t = shape_list(attn_vec)
        attn_vec = tf.reshape(attn_vec, (size_t[0], size_t[1], self.n_head * self.d_head))

        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out, training=training)

        if self.pre_lnorm:
            outputs = [w + attn_out]
        else:
            outputs = [self.layer_norm(w + attn_out)]

        return outputs


class TransformerXLLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        dropatt,
        initializer,
        pre_lnorm=False,
        r_w_bias=None,
        r_r_bias=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.d_inner = d_inner
        self.dropout = dropout
        self.dropatt = dropatt
        self.initializer = initializer
        self.pre_lnorm = pre_lnorm

        self.xltran_attn = RelativeMultiHeadAttn(
            n_head=self.n_head,
            d_model=self.d_model,
            d_head=self.d_head,
            dropout=self.dropout,
            dropatt=self.dropatt,
            kernel_initializer=self.initializer,
            pre_lnorm=self.pre_lnorm,
            r_w_bias=r_w_bias,
            r_r_bias=r_r_bias,
            name="xltran_attn",
        )
        self.pos_ff = PositionwiseFF(
            d_model=self.d_model,
            d_inner=self.d_inner,
            dropout=self.dropout,
            kernel_initializer=self.initializer,
            pre_lnorm=self.pre_lnorm,
            name="pos_ff",
        )

    def call(self, inputs, training=False):
        inp, r, attn_mask, mems = inputs
        attn_outputs = self.xltran_attn([inp, r, attn_mask, mems], training=training)
        ff_output = self.pos_ff(attn_outputs[0], training=training)

        outputs = [ff_output[0]]

        return outputs


class AdaptiveEmbedding(tf.keras.layers.Layer):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, initializer, \
                 proj_initializer=None, div_val=1, proj_same_dim=True, \
                 use_tpu=True, **kwargs):
        super().__init__(**kwargs)

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs

        self.initializer = initializer
        self.proj_initializer = proj_initializer if proj_initializer is not None else initializer

        self.div_val = div_val
        self.proj_same_dim = proj_same_dim

        self.use_tpu = use_tpu

        self.emb_scale = d_proj ** 0.5

        self.emb_weights = []
        self.emb_projs = []

        for i in range(len(self.cutoffs)):
            l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
            d_emb_i = self.d_embed // (self.div_val ** i)
            self.emb_weights.append(
                self.add_weight(
                    shape=(r_idx - l_idx, d_emb_i),
                    initializer=self.initializer,
                    name="emb_weights_._{}".format(i),
                )
            )
            if d_emb_i == d_proj and \
                    (not self.proj_same_dim or self.div_val == 1):
                self.emb_projs.append(None)
            else:
                self.emb_projs.append(
                    self.add_weight(
                        shape=(d_emb_i, self.d_proj),
                        initializer=self.proj_initializer,
                        trainable=True,
                        name="emb_projs_._{}".format(i),
                    )
                )

    def get_weights(self):
        weights = {"emb_layers": [], "emb_projs": []}
        for i in range(len(self.emb_layers)):
            weights["emb_layers"].append(self.emb_layers[i].get_weights())
            weights["emb_projs"].append(self.emb_projs[i])
        return weights

    @staticmethod
    def _embedding_lookup(lookup_table, x, use_tpu=False):
        if use_tpu:
            n_token = shape_list(lookup_table)[0]
            one_hot_idx = tf.one_hot(x, n_token)
            if one_hot_idx.shape.ndims == 2:
                return tf.einsum('nd,in->id', lookup_table, one_hot_idx)
            else:
                return tf.einsum('nd,ibn->ibd', lookup_table, one_hot_idx)
        else:
            return tf.nn.embedding_lookup(lookup_table, x)

    def call(self, inp):
        inp_flat = tf.reshape(inp, (-1,))
        emb_flat = tf.zeros([shape_list(inp_flat)[0], self.d_proj])
        for i in range(len(self.cutoffs)):
            l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

            mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
            inp_i = tf.minimum(inp_flat, r_idx-1)
            inp_i = tf.maximum(inp_i-l_idx, 0)
            emb_i = self._embedding_lookup(self.emb_weights[i], inp_i, self.use_tpu)
            if self.emb_projs[i] is not None:
                emb_i = tf.einsum("id,de->ie", emb_i, self.emb_projs[i])

            mask_i = tf.tile(tf.reshape(mask_i, [-1, 1]), [1, self.d_proj])
            emb_flat = tf.where(mask_i, emb_i, emb_flat)

        embed_shape = shape_list(inp) + [self.d_proj]
        embed = tf.reshape(emb_flat, embed_shape)

        embed *= self.emb_scale

        return embed


class AdaptiveSoftmax(tf.keras.layers.Layer):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, tie_projs, \
                 initializer=None, proj_initializer=None, div_val=1, \
                 proj_same_dim=True, tied_to=None, **kwargs):
        super().__init__(**kwargs)

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.n_clusters = len(self.cutoffs) - 1

        self.div_val = div_val
        self.proj_same_dim=True

        self.tied_to = tied_to
        assert tied_to is not None
        self.tie_projs = tie_projs

        self.out_weights = []
        self.out_biases = []
        self.out_projs = []

        if self.n_clusters > 0:
            self.cluster_weight = self.add_weight(
                shape=(self.n_clusters, self.d_embed), initializer="zeros", \
                    trainable=True, name="cluster_weight"
            )
            self.cluster_bias = self.add_weight(
                shape=(self.n_clusters,), initializer="zeros", trainable=True, \
                    name="cluster_bias"
            )

        for i, emb_weight in enumerate(self.tied_to.emb_weights):
            self.out_weights.append(emb_weight)
            vocab_size = shape_list(emb_weight)[0]
            self.out_biases.append(
                self.add_weight(
                    shape=(vocab_size,),
                    initializer="zeros",
                    trainable=True,
                    name="out_layers_._{}_.bias".
                        format(i)
                )
            )

        for i, emb_proj in enumerate(self.tied_to.emb_projs):
            out_proj = emb_proj
            if emb_proj is not None and not self.tie_projs[i]:
                out_proj = self.add_weight(
                    shape=shape_list(emb_proj),
                    initializer=proj_initializer,
                    trainable=True,
                    name="out_projs_._{}".format(i)
                )
            self.out_projs.append(out_proj)

    @staticmethod
    def _logit(x, W, b, proj=None):
        y = x
        if x.shape.ndims == 3:
            if proj is not None:
                y = tf.einsum("ibd,ed->ibe", y, proj)
            return tf.einsum("ibd,nd->ibn", y, W) + b
        else:
            if proj is not None:
                y = tf.einsum('id,ed->ie', y, proj)
            return tf.einsum('id,nd->in', y, W) + b

    @staticmethod
    def _gather_logprob(logprob, target):
        lp_size = shape_list(target)
        r = tf.range(lp_size[0])
        c = tf.range(lp_size[1])
        C, R = tf.meshgrid(c, r)
        idx = tf.stack([R, C, target], axis=2)
        return tf.gather_nd(logprob, idx)

    def call(self, inputs, return_mean=True):
        hidden, target = inputs
        head_logprob = 0
        if self.n_clusters == 0:
            output = self._logit(hidden, self.out_weights[0], self.out_biases[0], self.out_projs[0])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=output)
        else:
            hidden_sizes = shape_list(hidden)
            out = []
            loss = tf.zeros(hidden_sizes[:2], dtype=tf.float32)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask = (target >= l_idx) & (target < r_idx)
                cur_target = tf.minimum(target, r_idx-1)
                cur_target = tf.maximum(cur_target-l_idx, 0)

                cur_W = self.out_weights[i]
                cur_b = self.out_biases[i]
                cur_P = self.out_projs[i]

                if i == 0:
                    cur_W = tf.concat([cur_W, self.cluster_weight], 0)
                    cur_b = tf.concat([cur_b, self.cluster_bias], 0)

                    head_logit = self._logit(hidden, cur_W, cur_b, cur_P)
                    head_logprob = tf.nn.log_softmax(head_logit)
                        
                    cur_loss = self._gather_logprob(head_logprob, cur_target)
                    loss = tf.where(mask, cur_loss, loss)
                else:
                    tail_logit = self._logit(hidden, cur_W, cur_b, cur_P)
                    tail_logprob = tf.nn.log_softmax(tail_logit)

                    cluster_prob_idx = self.cutoffs[0] + i - 1
                    logprob_i = head_logprob[..., cluster_prob_idx, None] + tail_logprob

                    cur_loss = self._gather_logprob(logprob_i, cur_target)
                    loss = tf.where(mask, cur_loss, loss)
            loss = -loss
        if return_mean:
            loss = tf.reduce_mean(loss)

        return loss


class TransformerXL(tf.keras.Model):
    def __init__(self, n_token, n_layer, d_model, d_embed, n_head, d_head, d_inner, \
                 dropout, dropatt, initializer, proj_initializer=None, pre_lnorm=False, tgt_len=None, \
                 mem_len=0, cutoffs=[], div_val=1, tie_projs=[], same_length=False, \
                 clamp_len=-1, untie_r=False, proj_same_dim=True, use_tpu=True):

        super(TransformerXL, self).__init__()

        self.n_token = n_token
        self.n_layer = n_layer
        self.d_model = d_model
        self.d_embed = d_embed 
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner

        self.tgt_len = tgt_len
        self.mem_len = mem_len

        self.dropout = dropout 
        self.dropatt = dropatt 

        self.cutoffs = cutoffs 
        self.div_val = div_val
        self.tie_projs = tie_projs
        self.same_length = same_length
        self.clamp_len = clamp_len
        self.untie_r = untie_r
        self.proj_same_dim = proj_same_dim

        self.initializer = initializer
        self.proj_initializer = proj_initializer if proj_initializer is not None else initializer

        self.pre_lnorm = pre_lnorm
        self.use_tpu = use_tpu

        self.embedding_layer = AdaptiveEmbedding(
                n_token=self.n_token, 
                d_embed=self.d_embed, 
                d_proj=self.d_model, 
                cutoffs=self.cutoffs, 
                initializer=self.initializer, 
                proj_initializer=self.proj_initializer,
                div_val=self.div_val,
                proj_same_dim=self.proj_same_dim,
                use_tpu=self.use_tpu,
                name='emb_layer'
            )
        self.pos_emb = PositionalEmbedding(d_model)

        self.emb_dropout = tf.keras.layers.Dropout(dropout, name='emb_drop')
        self.pos_dropout = tf.keras.layers.Dropout(dropout, name='pos_drop')

        if not self.untie_r:
            self.r_w_bias = self.add_weight(
                shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_w_bias"
            )
            self.r_r_bias = self.add_weight(
                shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_r_bias"
            )

        self.tran_layers = []
        for i in range(self.n_layer):
            self.tran_layers.append(
                TransformerXLLayer(
                    n_head=self.n_head,
                    d_model=self.d_model,
                    d_head=self.d_head,
                    d_inner=self.d_inner,
                    dropout=self.dropout,
                    dropatt=self.dropatt,
                    initializer=self.initializer,
                    pre_lnorm=self.pre_lnorm,
                    r_w_bias=None if self.untie_r else self.r_w_bias,
                    r_r_bias=None if self.untie_r else self.r_r_bias,
                    name='layers_._{}'.format(i)
                )
            )

        self.out_dropout = tf.keras.layers.Dropout(dropout, name='out_drop')
        self.logsoftmax_layer = AdaptiveSoftmax(
                n_token=self.n_token,
                d_embed=self.d_embed,
                d_proj=self.d_model,
                cutoffs=self.cutoffs,
                tie_projs=self.tie_projs,
                initializer=self.initializer,
                proj_initializer=self.proj_initializer,
                div_val=self.div_val,
                proj_same_dim=self.proj_same_dim,
                tied_to=self.embedding_layer,
                name='softmax_layer'
            )

    def reset_length(self, tgt_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len

    def init_mems(self, bsz, mem_len):
        mems = []
        for i in range(self.n_layer):
            empty = tf.zeros([mem_len, bsz, self.d_model])
            mems.append(empty)
        return mems

    def _update_mems(self, hids, mems):
        if mems is None:
            return None
        assert len(hids) == len(mems), "len(hids) != len(mems)"
        new_mems = []
        for i in range(len(hids)):
            cat = tf.concat([mems[i], hids[i]], axis=0)
            cat = tf.stop_gradient(cat)
            mlen = shape_list(mems[i])[0]
            if mlen > 0:
                new_mems.append(cat[-mlen:])
            else:
                shape = [mlen]+shape_list(cat)[1:]
                new_mems.append(tf.zeros(shape))
        return new_mems

    def _create_mask(self, qlen, mlen, same_length=False):
        attn_mask = tf.ones([qlen, qlen])
        mask_u = tf.linalg.band_part(attn_mask, 0, -1)
        mask_dia = tf.linalg.band_part(attn_mask, 0, 0)
        attn_mask_pad = tf.zeros([qlen, mlen])
        ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
        if same_length:
            mask_l = tf.linalg.band_part(attn_mask, -1, 0)
            ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
        return ret

    def call(self, inp, tgt, mems=None, return_mean=False, training=False):
        # the original code for Transformer-XL used shapes [len, bsz] 
        # so we transpose here from shape [bsz, len] to shape [len, bsz]
        inp = tf.transpose(inp, perm=(1, 0))
        tgt = tf.transpose(tgt, perm=(1, 0))

        qlen, bsz = shape_list(inp)

        if mems is None:
            mlen = self.mem_len
        else:
            mlen = shape_list(mems)[1]
        klen = mlen + qlen

       
        if mems is None:
            mems = self.init_mems(bsz, mlen)
        else:
            mems = tf.unstack(mems, axis=0)
            assert(shape_list(mems[0])[1] == bsz)
            assert(len(mems) == self.n_layer)

        attn_mask = self._create_mask(qlen, mlen, self.same_length)

        word_emb = self.embedding_layer(inp)
        d_word_emb = self.emb_dropout(word_emb, training=training)

        pos_seq = tf.range(klen - 1, -1, -1.0)
        if self.clamp_len > 0:
            pos_seq = tf.minimum(pos_seq, self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)
        d_pos_emb = self.pos_dropout(pos_emb, training=training)

        core_out = d_word_emb
        hids = []
        for i, layer in enumerate(self.tran_layers):
            hids.append(core_out)
           
            mems_i = mems[i]
            all_out = layer([core_out, d_pos_emb, attn_mask, mems_i], training=training)
            core_out = all_out[0]
        core_out = self.out_dropout(core_out, training=training)

        new_mems = self._update_mems(hids, mems)
        new_mems = tf.stack(new_mems)

        loss = self.logsoftmax_layer([core_out, tgt], return_mean=return_mean, training=training)

        # transpose loss back to shape [bsz, len] if necessary
        if loss.shape.ndims == 2:
            loss = tf.transpose(loss, [1, 0])

        return loss, new_mems
        

