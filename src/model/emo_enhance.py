import torch

# NOTE: Step1: 判断情感词是否在input_batch当中, 避免分词方式不同所造成的瑕疵, 并获得情感词在input_batch中的位置

def get_emotion_words_pos(context_batch, emotion_context_batch):
    # [bz, len]
    context_batch = context_batch.tolist()
    emotion_context_batch = emotion_context_batch.tolist()
    pos_for_each_emotion_word = {}
    for value in emotion_context_batch:
        # 剔除PAD以及context中未出现的情感词
        if value != 1 and value in context_batch:
            pos_for_each_emotion_word[value] = [pos for pos, word in enumerate(context_batch) if word == value]
    #print("[pos_for_each_emotion_word]:",pos_for_each_emotion_word)

    emotion_words_pos = []
    num_dict = {k:0 for k in set(emotion_context_batch)}
    for value in emotion_context_batch:
        # 剔除PAD以及context中未出现的情感词
        if value != 1 and value in context_batch:
            num_dict[value] += 1
            emotion_words_pos.append(pos_for_each_emotion_word[value][num_dict[value]-1])
    #print("[emotion_words_pos]:",emotion_words_pos)

    return emotion_words_pos


def get_emo_enhance_op_order(dp_x, emotion_words_pos):
    # NOTE: Step2: 根据情感词在input_batch中的位置寻找路径, 得到一棵将所有情感词连在一起的子树
    #emotion_words_pos = torch.tensor([3,6,9,12,31,36]).tolist()

    dp_x=dp_x.tolist()
    #dp_y=dp_y.tolist()
    # NOTE: 父节点集合
    fathers = set(dp_x)
    # NOTE: 父节点num的子节点集合
    rel_dict = {}
    max_width = 0
    for father in fathers:
        rel_dict[father] = [i for i,x in enumerate(dp_x) if x == father and i != 0]
    paths = {}
    emo_rel_path = []
    for y in emotion_words_pos:
        # NOTE: save path from y to root
        paths[y] = []
        current_path = [y]
        current_node = dp_x[y]
        if current_node not in current_path:
            current_path.append(current_node)
            paths[y].append((y, current_node))
            emo_rel_path.append((y, current_node))
            while current_node != 0:
                y_ = current_node
                current_node = dp_x[y_]
                if current_node not in current_path:
                    current_path.append(current_node)
                    paths[y].append((y_, current_node))
                    emo_rel_path.append((y_, current_node))
                else:
                    if y_ not in rel_dict[0]:
                        rel_dict[0].append(y_)
                        if len(rel_dict[0]) > max_width: max_width = len(rel_dict[0])
                    paths[y].append((y_, 0))
                    emo_rel_path.append((y_, 0))
                    break
        else:
            if y not in rel_dict[0] and y != 0:
                rel_dict[0].append(y)
                if len(rel_dict[0]) > max_width: max_width = len(rel_dict[0])
            paths[y].append((y, 0))
            emo_rel_path.append((y, 0))

    #print("[path]:",paths)
    #print("[emo_rel_path]:",emo_rel_path)
    # NOTE:
    emo_rel_words_pos = []
    for path in emo_rel_path:
        if path[0] not in emo_rel_words_pos:
            emo_rel_words_pos.append(path[0])
        if path[1] not in emo_rel_words_pos:
            emo_rel_words_pos.append(path[1])
    # NOTE: all_paths在paths情感词路径的基础上添加了情感相关词的路径
    all_paths = {}
    for y in emo_rel_words_pos:
        # NOTE: save path from y to root
        all_paths[y] = []
        current_path = [y]
        current_node = dp_x[y]
        if current_node not in current_path:
            current_path.append(current_node)
            all_paths[y].append((y, current_node))
            while current_node != 0:
                y_ = current_node
                current_node = dp_x[y_]
                if current_node not in current_path:
                    current_path.append(current_node)
                    all_paths[y].append((y_, current_node))
                else:
                    if y_ not in rel_dict[0]:
                        rel_dict[0].append(y_)
                        if len(rel_dict[0]) > max_width: max_width = len(rel_dict[0])
                    all_paths[y].append((y_, 0))
                    break
        else:
            if y not in rel_dict[0] and y != 0:
                rel_dict[0].append(y)
                if len(rel_dict[0]) > max_width: max_width = len(rel_dict[0])
            all_paths[y].append((y, 0))
    #print("[all_paths]:",all_paths)
    # NOTE: 分层, 得到自底向上的操作顺序
    # NOTE: 编号与层数对应
    emo_rel_word_level = {} #
    for id, path in all_paths.items():
        emo_rel_word_level[id] = len(path)
    emo_rel_word_level_sorted = sorted(emo_rel_word_level.items(), key=lambda x: x[1], reverse=True)
    #print("[emo_rel_word_level]:",emo_rel_word_level_sorted) # 返回一个元组列表，[(编号, 层数)]
    max_depth = emo_rel_word_level_sorted[0][1]
    # NOTE: i表示在第i层, 对应列表中是位于第i层的节点
    emo_rel_word_level = {i:[] for i in range(max_depth,0,-1)}
    for word_level in emo_rel_word_level_sorted:
        if word_level[0] not in emo_rel_word_level[word_level[1]]:
            emo_rel_word_level[word_level[1]].append(word_level[0])
    #print("[emo_rel_word_level]:",emo_rel_word_level)
    # NOTE: 每个情感相关词的孩子, 以及父亲
    emo_rel_word_father = {pos:[] for pos in emo_rel_words_pos}
    emo_rel_word_children = {pos: [] for pos in emo_rel_words_pos}
    for path in emo_rel_path:
        # path[0]: son, path[1]: father
        if path[1] not in emo_rel_word_father[path[0]]:
            emo_rel_word_father[path[0]].append(path[1])
        if path[0] not in emo_rel_word_children[path[1]]:
            emo_rel_word_children[path[1]].append(path[0])
    #print("[emo_rel_word_father]:",emo_rel_word_father)
    #print("[emo_rel_word_children]:", emo_rel_word_children)

    op_order = []  # 共分几步, 以及每一步的操作顺序
    for level, rel_words in emo_rel_word_level.items():
        op_order += [(w, emo_rel_word_father[w][0]) for w in rel_words if w != 0]
    #print("[op_order]:", op_order)
    op_orders = {op[1]: [op[1]] for op in op_order}
    for op in op_order:
        op_orders[op[1]].append(op[0])
    #print("[op_orders]:", op_orders)
    return op_orders


#emotion_words_pos = get_emotion_words_pos(context_batch, emotion_context_batch)
#get_emo_enhance_op_order(dp_x, get_emotion_words_pos(context_batch, emotion_context_batch))