import numpy as np
import math
import torch
from src.utils import config
import torch.nn as nn


class TreePositionalEncodings(torch.nn.Module):
    # Novel positional encodings to enable tree-based transformers
    # https://papers.nips.cc/paper/2019/file/6e0917469214d8fbd8c517dcdc6b8dcf-Paper.pdf
    #def __init__(self, depth, degree, n_feat, d_model, max_d_tree_param):
    def __init__(self, d_model, max_d_tree_param):
        """
            depth: max tree depth
            degree: max num children
            n_feat: number of features
            d_model: size of model embeddings
        """
        super(TreePositionalEncodings, self).__init__()
        #self.depth = depth
        #self.width = degree
        #self.d_pos = n_feat * depth * degree
        self.d_model = d_model
        self.max_d_tree_param = max_d_tree_param
        #self.d_tree_param = self.d_pos // (self.depth * self.width)
        # NOTE: d_tree_param = n_feat, 用几位数表示one-hot的特征
        self.p = torch.nn.Parameter(torch.ones(self.max_d_tree_param, dtype=torch.float32), requires_grad=True)
        # NOTE: 用均匀分布[0.7, 0.999]中的抽样数值来初始化参数
        self.init_weights()

    def init_weights(self):
        self.p.data.uniform_(0.7, 0.999)

    def build_weights(self, width, depth, n_feat):
        # NOTE: 当前句对应的n_feat, 将参数self.p截断
        d_tree_param = n_feat
        tree_params = torch.tanh(self.p[:d_tree_param])
        # NOTE: [n_feat, ]-->[1, 1, n_feat]-->[depth, width, n_feat], 参数复制depth*width次
        tiled_tree_params = tree_params.reshape((1, 1, -1)).repeat(depth, width, 1)
        # NOTE: [depth, ]-->[depth, 1, 1]-->[depth, width, n_feat]
        #print("[self.p.device]:",self.p.device)
        tiled_depths = torch.arange(depth, dtype=torch.float32, device=self.p.device) \
            .reshape(-1, 1, 1).repeat(1, width, d_tree_param)
        # NOTE: 将encoding乘以一个因子(根号下1-p的平方)来近似地正则化它
        tree_norm = torch.sqrt((1 - torch.square(tree_params)) * self.d_model / 2)
        # NOTE: [depth * width, n_feat]
        #print(tiled_tree_params.shape,tiled_depths.shape,tree_norm.shape)
        if tiled_depths.shape[-1] != 300 and tiled_depths.shape[-1] != 150:
            tree_weights = (torch.pow(tiled_tree_params, tiled_depths) * tree_norm) \
                .reshape(depth * width, d_tree_param)
        else: tree_weights = torch.zeros(width*depth,300//(width*depth)).to(config.device)
        #print(tree_weights.shape)
        return tree_weights

    def treeify_positions(self, width, depth, n_feat, positions, tree_weights):
        # NOTE: positions=[bz, n, width * depth], tree_weights=[depth * width, n_feat]
        # NOTE: [bz, n, width * depth, 1] * [depth * width, n_feat] = [bz, n, depth * width, n_feat]
        treeified = positions.unsqueeze(-1) * tree_weights
        shape = treeified.shape
        # NOTE: [bz, n] + (n_feat * depth * degree, )
        shape = shape[:-2] + (width * depth * n_feat,)
        # NOTE: [bz, n, n_feat * depth * degree]
        #print(treeified.shape,shape)
        treeified = torch.reshape(treeified, shape)
        return treeified

    def _gen_tree_pos_signal(self, width, depth, n_feat, positions):
        """
            positions: Tensor [bs, n, width * depth]
            returns: Tensor [bs, n, width * depth * n_features]
        """

        # NOTE: [depth * width, n_feat]
        tree_weights = self.build_weights(width, depth, n_feat)
        # NOTE: [bz, n, width * depth]-->[bz, n, n_feat * depth * depth]
        positions = self.treeify_positions(width, depth, n_feat, positions, tree_weights)
        return positions


    def TreePosEnc_for_one_sent(self, dp_x, dp_y):
        dp_x=dp_x.tolist()
        dp_y=dp_y.tolist()
        # NOTE: 父节点集合
        fathers = set(dp_x)
        # NOTE: 父节点num的子节点集合
        rel_dict = {}
        max_width = 0
        for father in fathers:
            rel_dict[father] = [i for i,x in enumerate(dp_x) if x == father and i != 0]
            child_num = len(rel_dict[father])
            if child_num > max_width:
                max_width = child_num
        #print("[max_width]:",max_width)
        #print("[rel_dict]:",rel_dict)
        paths = {}
        for y in dp_y:
            # NOTE: save path from y to root
            paths[y] = []
            current_path = [y]
            current_node = dp_x[y]
            if current_node not in current_path:
                current_path.append(current_node)
                paths[y].append((y, current_node))
                while current_node != 0:
                    y_ = current_node
                    current_node = dp_x[y_]
                    if current_node not in current_path:
                        current_path.append(current_node)
                        paths[y].append((y_, current_node))
                    else:
                        if y_ not in rel_dict[0]:
                            rel_dict[0].append(y_)
                            if len(rel_dict[0]) > max_width: max_width = len(rel_dict[0])
                        paths[y].append((y_, 0))
                        break
            else:
                if y not in rel_dict[0] and y != 0:
                    rel_dict[0].append(y)
                    if len(rel_dict[0]) > max_width: max_width = len(rel_dict[0])
                paths[y].append((y, 0))
        paths_ids = {}
        max_deepth = 0
        for node, path in paths.items():
            # NOTE: PAD dp_map (0,0) 无须处理
            if node != 0:
                paths_ids[node] = []
                for path_node in path:
                    paths_ids[node].append(rel_dict[path_node[1]].index(path_node[0]))
                path_len = len(paths_ids[node])
                if path_len > max_deepth:
                    max_deepth = path_len
        #print("[max_deepth]:",max_deepth)
        #print("[paths_ids]:",paths_ids)
        tensor_list = []
        for node, path in paths_ids.items():
            # 当前节点node对应的路径path的长度就是当前层数, 返回列表中是该层所有节点对应位置的tensor
            # 根据path计算node在当前层所有节点中的位置
            path_tensor = [torch.zeros(max_width) for _ in range(max_deepth)]
            # level从最底层向上
            for level, path_id in enumerate(path):
                path_tensor[level][path_id] = 1

            tensor_list.append(torch.cat(path_tensor, dim=-1))
        # NOTE: [1, len, max_width * max_deepth]
        init_tree_pos_encoding = torch.stack(tensor_list).unsqueeze(0).to(config.device)
        #print("[init_tree_pos_encoding]:\n", init_tree_pos_encoding)

        n_feat = self.d_model // (max_width * max_deepth)

        tree_pos_embeddings_ = self._gen_tree_pos_signal(max_width, max_deepth, n_feat, init_tree_pos_encoding)
        tree_pos_embeddings = torch.zeros(1, tree_pos_embeddings_.shape[1], self.d_model - max_width * max_deepth * n_feat).to(config.device)
        tree_pos_embeddings = torch.cat([tree_pos_embeddings_, tree_pos_embeddings], dim=-1)
        return tree_pos_embeddings

    def forward(self, dp_maps, n_dim):
        # NOTE: dp_maps = [bz, 2, len]
        tps_list = []
        max_len = dp_maps.shape[-1]
        for dp_map in dp_maps:
            #print("[dp_map]:", dp_map)

            tree_pos_tensor = self.TreePosEnc_for_one_sent(dp_map[0], dp_map[1])
            sos_tensor = torch.zeros(1, 1, n_dim).to(config.device)
            pad_tensor = torch.zeros(1, max_len - tree_pos_tensor.shape[1] - 1, n_dim).to(config.device)
            tps_list.append(torch.cat([sos_tensor, tree_pos_tensor, pad_tensor], dim=1))
        tps_batch = torch.cat(tps_list, dim=0)
        return tps_batch















