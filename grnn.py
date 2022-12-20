import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class GRNNConv(nn.Layer):
    def __init__(self,
                 input_size,
                 hidden_size,
                 feat_drop=0.6,
                 attn_drop=0.6,
                 num_heads=1,
                 concat=True,
                 activation=None,
                 residual = False):
        super(GRNNConv, self).__init__()
        self.hidden_size = hidden_size
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.num_heads = num_heads
        self.concat = concat
        self.residual = residual

        self.linear = nn.Linear(input_size, num_heads * hidden_size)
        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        if isinstance(activation, str):
            activation = getattr(F, activation)
        self.activation = activation
        
    def _degree_norm(self, graph, beta=0.5, mode="indegree"):
        if mode == "indegree":
            degree = graph.indegree()
        elif mode == "outdegree":
            degree = graph.outdegree()
        norm = paddle.cast(degree, dtype=paddle.get_default_dtype())
        norm = paddle.clip(norm, min=0.5)
        norm = paddle.pow(norm, -1 * float(beta))
        norm = paddle.reshape(norm, [-1, 1])
        return norm

    def _send_attention(self, src_feat, dst_feat, edge_feat):
        return {'h':src_feat['h'], 'e':edge_feat['e']}

    def _reduce_attention(self, msg):
        alpha = msg["e"]
        alpha = paddle.reshape(alpha, [-1, 1])
        if self.attn_drop > 1e-15:
            alpha = self.attn_dropout(alpha)
        feature = msg["h"]
        feature = feature * alpha
        feature = msg.reduce(feature, pool_type="sum")
        return feature

    def forward(self, graph, feature, feat):
        X = feature.clone()
        if self.feat_drop > 1e-15:
            feature = self.feat_dropout(feature)
        feature = self.linear(feature) + nn.Linear(feat.shape[1], self.num_heads*self.hidden_size)(feat)
        norm = self._degree_norm(graph)
        feature = feature * norm
        msg = graph.send(self._send_attention, src_feat={"h": feature},
                        edge_feat={'e':paddle.ones([graph.num_edges, 1])})
        feature = graph.recv(reduce_func=self._reduce_attention, msg=msg)
        feature = feature * norm
        
        if self.concat == False:
            feature = sum(paddle.split(feature, self.num_heads, axis=1), 1)
            
        if self.activation is not None:
            feature = self.activation(feature)  
            
        if self.residual == True:
            if feature.shape[1] == X.shape[1]:
                feature = feature + X
            else:
                feature = feature + nn.Linear(X.shape[1], feature.shape[1])(X)
            
        return feature
    
    
class GRNN(nn.Layer):
    """Implement of GRNN
    """

    def __init__(
            self,
            input_size,
            num_class,
            feat_drop=0.6,
            attn_drop=0.6,
            input_drop = 0.6,
            hidden_size=8, 
            num_heads=[8, 1],
            activation = 'elu',
            residual = False):
        super(GRNN, self).__init__()
        self.input_size = input_size
        self.num_class = num_class
        self.num_layers = len(num_heads)
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.input_drop = input_drop
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.activation = activation
        self.residual = residual
        self.gats = nn.LayerList()
        if self.input_drop > 1e-15:
            self.gats.append(nn.Dropout(self.input_drop))
        for i in range(self.num_layers):
            
            if self.num_layers == 1:
                self.gats.append(
                    GRNNConv(
                        self.input_size,
                        self.num_class,
                        self.feat_drop,
                        self.attn_drop,
                        self.num_heads[i],
                        concat=False,
                        activation=None, 
                        residual = self.residual))
                break
            
            if i == 0:
                self.gats.append(
                    GRNNConv(
                        self.input_size,
                        self.hidden_size,
                        self.feat_drop,
                        self.attn_drop,
                        self.num_heads[i],
                        concat=True,
                        activation=self.activation,
                        residual = self.residual))
            elif i == (self.num_layers - 1):
                self.gats.append(
                    GRNNConv(
                        self.num_heads[i-1] * self.hidden_size,
                        self.num_class,
                        self.feat_drop,
                        self.attn_drop,
                        self.num_heads[i],
                        concat=False,
                        activation=None, 
                        residual = self.residual))
            else:
                self.gats.append(
                    GRNNConv(
                        self.num_heads[i-1] * self.hidden_size,
                        self.hidden_size,
                        self.feat_drop,
                        self.attn_drop,
                        self.num_heads[i],
                        concat=True,
                        activation=self.activation, 
                        residual = self.residual))

    def forward(self, graph, feature):
        feat = feature.clone()
        for m in self.gats:
            if isinstance(m, nn.Dropout):
                feature = m(feature)
            else:
                feature = m(graph, feature, feat)
        return feature