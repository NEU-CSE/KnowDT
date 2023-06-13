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






def get_emotion_path(dp_x, dp_y, emotion_words_pos):
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

    # NOTE: Step3: 得到一张mask, 将与情感词无关的内容遮挡
    emo_rel_words = []
    for x,y in emo_rel_path:
        if x not in emo_rel_words:
            emo_rel_words.append(x)
        if y not in emo_rel_words:
            emo_rel_words.append(y)
    #print("[emo_rel_words]:",emo_rel_words)
    #print(dp_y)
    mask = torch.zeros_like(dp_y)
    for index in range(len(dp_y)):
        if dp_y[index] not in emo_rel_words:
            mask[index] = 1
    #print("[mask]:",mask.data.eq(1))
    return mask




def get_emotion_path_mask_for_one_batch(context_batch, emotion_context_batch, rel_map_batch):
    # context_batch=[bz, len], emotion_context_batch=[bz, len], rel_map=[bz, 2, len]
    mask_list = []
    for i in range(len(context_batch)):
        context = context_batch[i]
        emotion_context = emotion_context_batch[i]
        rel_map = rel_map_batch[i]

        dp_x = rel_map[0]
        dp_y = rel_map[1]
        emotion_words_pos = get_emotion_words_pos(context, emotion_context)
        mask = get_emotion_path(dp_x, dp_y, emotion_words_pos)
        mask_list.append(mask)

    emo_mask_batch = torch.stack(mask_list).unsqueeze(1)
    return emo_mask_batch
