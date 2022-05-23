import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn

from torch.nn import functional as F
from dgl.base import DGLError
from dgl.utils import expand_as_pair, check_eq_shape, dgl_warning


class AttentionAGG(nn.Module):
    """
    Transform sequence of vectors into one vector by summing with attention scores.
    """
    def __init__(self, in_feats, out_feats, bias=True, batch_first=True):
        super().__init__()
        self.in_feats, self.out_feats, self.bias, self.batch_first = in_feats, out_feats, bias, batch_first
        self.fc1 = nn.Linear(in_feats, out_feats, bias)
        self.fc2 = nn.Linear(out_feats, 1, bias)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # (1 if batch_first else 0))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight,
                                gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.fc2.weight)

    def _get_attention_weights(self, inp):
        """
        Calculates attention scores.

        :param inp: torch tensor in shape (batch_size, seq_lenght, dim_size)
        :return: attention scores.
        """
        inp = self.tanh(self.fc1(inp))
        inp = self.fc2(inp)
        weights = self.softmax(inp)
        return weights

    def forward(self, inp):
        if not self.batch_first:
            inp = inp.transpose(0, 1)
        weights = self._get_attention_weights(inp)
        inp = (weights * inp).sum(dim=1)
        return inp


class GAttConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(GAttConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.att_agg = AttentionAGG(self._in_dst_feats + self._in_src_feats, self._in_src_feats, bias=bias,
                                    batch_first=True)
        self.fc_layer = nn.Linear(self._in_src_feats + self._in_dst_feats, self._out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.att_agg.reset_parameters()
        nn.init.xavier_uniform_(self.fc_layer.weight,
                                gain=nn.init.calculate_gain('relu'))

    def _att_reducer(self, nodes):
        num_messages = nodes.mailbox["m"].size(1)
        att_inp = torch.cat(
            [nodes.data["h"].unsqueeze(1).expand(-1, num_messages, -1),
             nodes.mailbox["m"]], dim=2
        )
        att_weights = self.att_agg._get_attention_weights(att_inp)
        return {'neigh': (att_weights * nodes.mailbox["m"]).sum(dim=1)}

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            msg_fn = dgl.function.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                msg_fn = dgl.function.u_mul_e('h', '_edge_weight', 'm')

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            # Message Passing
            graph.srcdata['h'] = feat_src
            graph.dstdata['h'] = feat_dst
            graph.update_all(msg_fn, self._att_reducer)
            rst = self.fc_layer(torch.cat([h_self, graph.dstdata['neigh']], dim=1))

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


class GCN(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            "ui": dglnn.SAGEConv(in_feat, out_feat, "mean", activation=nn.ReLU())
        })

    def forward(self, blocks, feats):
        feats = self.conv1(blocks[0], feats)
        return feats


class GAttN(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            "ui": GAttConv(in_feat, out_feat, activation=nn.ReLU())
        })

    def forward(self, blocks, feats):
        feats = self.conv1(blocks[0], feats)
        return feats


class DotPredictor(nn.Module):
    def forward(self, g, item_feats, user_feats):
        with g.local_scope():
            g.nodes["item"].data["feats"] = item_feats
            g.nodes["user"].data["feats"] = user_feats
            g.apply_edges(
                dgl.function.u_dot_v('feats', 'feats', 'score')
            )
            return g.edata['score'][:, 0]


class MLPPredictor(nn.Module):
    def __init__(self, feats_size, bias):
        super().__init__()
        self.fc1 = nn.Linear(feats_size * 2, feats_size, bias=bias)
        self.fc2 = nn.Linear(feats_size, 1, bias=bias)
        self.relu = nn.ReLU()

    def apply_edges(self, edges):
        h = torch.cat([edges.src['feats'], edges.dst['feats']], dim=1)
        return {'score': self.fc2(self.relu(self.fc1(h))).squeeze(1)}

    def forward(self, g, item_feats, user_feats):
        with g.local_scope():
            g.nodes["item"].data["feats"] = item_feats
            g.nodes["user"].data["feats"] = user_feats
            g.apply_edges(self.apply_edges)
            return g.edata['score']


class HMModel(nn.Module):
    def __init__(self, gnn, purchases_agg_func, rnn, rnn_out_agg_func, predictor,
                 gnn_in_size, gnn_out_size, rnn_in_size, rnn_hid_size,
                 rnn_num_layers, num_users, num_items, raw_feats_dim):
        super().__init__()
        (self.gnn_in_size, self.gnn_out_size, self.rnn_in_size,
         self.rnn_hid_size, self.rnn_num_layers, self.num_users,
         self.num_items, self.raw_feats_dim) = (gnn_in_size, gnn_out_size, rnn_in_size, rnn_hid_size,
                                                rnn_num_layers, num_users, num_items, raw_feats_dim)
        self.raw_feats = dglnn.HeteroEmbedding({"user": num_users, "item": num_items}, raw_feats_dim)
        self.gnn = gnn
        self.purchases_agg_func = purchases_agg_func
        self.rnn = rnn
        self.rnn_out_agg_func = rnn_out_agg_func
        self.predictor = predictor

    def get_graph_item_embs(self, dataloader, out=None):
        """Computes graph-aware representation of each item."""
        device = next(self.parameters()).device
        if out is None:
            out = torch.empty(self.num_items, self.gnn_out_size, device=device)
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(device) for b in blocks]
            input_nodes = {k: v.to(device) for k, v in input_nodes.items()}
            output_nodes = {k: v.to(device) for k, v in output_nodes.items()}
            input_feats = self.raw_feats(input_nodes)
            out[output_nodes["item"].long()] = self.gnn(blocks, input_feats)["item"]
        return out

    def purchases_agg(self, purchases_graph, items_embs):
        """Aggregates purchased items embeddings for each user in the graph."""
        with purchases_graph.local_scope():
            purchases_graph.nodes["item"].data["embs"] = items_embs
            purchases_graph.multi_update_all(
                {'iu': (dgl.function.copy_u("embs", 'm'), self.purchases_agg_func)},
                "sum"
            )
            return purchases_graph.nodes["user"].data["embs"]

    def get_user_embs(self, purchases_graphs, item_embs, users_inds):
        """Computes vector representation of each user in `users_inds`."""
        device = next(self.parameters()).device
        users_embs_seq = list()
        for purchases_g in purchases_graphs:
            items_inds = purchases_g.in_edges(users_inds)[0].unique()
            sub_g = purchases_g.subgraph({"item": items_inds, "user": users_inds},
                                         output_device=device)
            users_embs = self.purchases_agg(sub_g, item_embs[sub_g.nodes["item"].data[dgl.NID].long()])
            users_embs_seq.append(users_embs.unsqueeze(0))
        users_embs_seq = torch.cat(users_embs_seq)
        outp, _ = self.rnn(users_embs_seq)
        if self.rnn_out_agg_func is None:
            return outp[-1]
        return self.rnn_out_agg_func(outp)

    def predict_purchases(self, items_embs, users_embs, num_preds=12, item_enc=None):
        """Predicts `num_preds` purchases for each user in `users_embs.size(0)`."""
        device = next(self.parameters()).device
        users_pred = list()
        data_dict = {("user", "ui", "item"):
                         (torch.zeros(self.num_items, dtype=torch.int32, device=device),
                          torch.arange(self.num_items, dtype=torch.int32, device=device))
                     }
        num_dict = {"user": 1, "item": self.num_items}
        full_g = dgl.heterograph(data_dict, num_dict, device=device)
        for i in range(users_embs.shape[0]):
            with full_g.local_scope():
                scores = self.predictor(full_g, items_embs, users_embs[i].view(1, -1))
                full_g.edata["score"] = nn.Sigmoid()(scores).view(-1, 1)
                item_inds = full_g.edges()[1][dgl.topk_edges(full_g, "score", num_preds)[-1].squeeze()]
            if item_enc is not None:
                item_inds = item_enc.inverse_transform(item_inds.to("cpu"))
            users_pred.append(item_inds)
        users_pred = np.array(users_pred)
        return users_pred


# Predefined HMModel variations
class HMModel_GAttN_MEAN_GRU_Att_DOT(HMModel):
    def __init__(self, gnn_in_size, gnn_out_size, rnn_in_size, rnn_hid_size,
                 rnn_num_layers, num_users, num_items, raw_feats_dim):
        gnn = GAttN(gnn_in_size, gnn_out_size)
        purchases_agg_func = dgl.function.mean('m', 'embs')
        rnn = nn.GRU(rnn_in_size, rnn_hid_size, rnn_num_layers)
        rnn_out_agg_func = AttentionAGG(rnn_hid_size, rnn_hid_size, bias=True, batch_first=False)
        predictor = DotPredictor()
        super().__init__(gnn, purchases_agg_func, rnn, rnn_out_agg_func, predictor,
                         gnn_in_size, gnn_out_size, rnn_in_size, rnn_hid_size,
                         rnn_num_layers, num_users, num_items, raw_feats_dim)


class HMModel_GAttN_MAX_GRU_Att_DOT(HMModel):
    def __init__(self, gnn_in_size, gnn_out_size, rnn_in_size, rnn_hid_size,
                 rnn_num_layers, num_users, num_items, raw_feats_dim):
        gnn = GAttN(gnn_in_size, gnn_out_size)
        purchases_agg_func = dgl.function.max('m', 'embs')
        rnn = nn.GRU(rnn_in_size, rnn_hid_size, rnn_num_layers)
        rnn_out_agg_func = AttentionAGG(rnn_hid_size, rnn_hid_size, bias=True, batch_first=False)
        predictor = DotPredictor()
        super().__init__(gnn, purchases_agg_func, rnn, rnn_out_agg_func, predictor,
                         gnn_in_size, gnn_out_size, rnn_in_size, rnn_hid_size,
                         rnn_num_layers, num_users, num_items, raw_feats_dim)


class HMModel_GAttN_Att_GRU_Att_DOT(HMModel):
    def __init__(self, gnn_in_size, gnn_out_size, rnn_in_size, rnn_hid_size,
                 rnn_num_layers, num_users, num_items, raw_feats_dim):
        gnn = GAttN(gnn_in_size, gnn_out_size)
        purchases_agg_func = lambda nodes: {"embs": self.purchases_att_agg(nodes.mailbox["m"])}
        rnn = nn.GRU(rnn_in_size, rnn_hid_size, rnn_num_layers)
        rnn_out_agg_func = AttentionAGG(rnn_hid_size, rnn_hid_size, bias=True, batch_first=False)
        predictor = DotPredictor()
        super().__init__(gnn, purchases_agg_func, rnn, rnn_out_agg_func, predictor,
                         gnn_in_size, gnn_out_size, rnn_in_size, rnn_hid_size,
                         rnn_num_layers, num_users, num_items, raw_feats_dim)
        self.purchases_att_agg = AttentionAGG(gnn_out_size, gnn_out_size, bias=True, batch_first=True)
