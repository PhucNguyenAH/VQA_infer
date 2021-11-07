from __init__ import PYTHON_PATH
from cfgs.base_cfgs import Cfgs
import yaml

from core.data.load_data import DataSet
from core.model.net import Net
from core.data.data_utils import tokenize,tokenize_question, ans_stat, proc_ques, img_feat_path_load, proc_img_feat


import os, json, torch, datetime, pickle, copy, shutil, time
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data as Data

cfg_file = os.path.join(PYTHON_PATH, "ailibs_data/modular_coattention/large_model.yml")
with open(cfg_file, 'r') as f:
    yaml_dict = yaml.load(f)
__C = Cfgs()
__C.add_args(yaml_dict)
__C.proc()

path = os.path.join(PYTHON_PATH, "ailibs_data/modular_coattention/model.pkl")
print('Loading ckpt {}'.format(path))
state_dict = torch.load(path)['state_dict']
print('Finish!')


dataset = DataSet(__C)
data_size = dataset.data_size
token_size = dataset.token_size
ans_size = dataset.ans_size
pretrained_emb = dataset.pretrained_emb

net = Net(
    __C,
    pretrained_emb,
    token_size,
    ans_size
)
net.cuda()
net.eval()

net.load_state_dict(state_dict)

question = "What are there?"


token_to_ix, pretrained_emb = tokenize_question(question, __C.USE_GLOVE)
token_size = token_to_ix.__len__()
print(token_to_ix)
print(pretrained_emb)
print(token_size)

ques_ix_iter = torch.from_numpy(proc_ques(question, token_to_ix, __C.MAX_TOKEN))[np.newaxis, :]
print(ques_ix_iter.size())
print(ques_ix_iter)



# image_path = os.path.join(PYTHON_PATH, "ailibs_data/modular_coattention/large_model.yml")



img_feat_path = os.path.join(PYTHON_PATH,"datasets/coco_extract/test2015/COCO_test2015_000000010868.jpg.npz")
# iid_to_img_feat_path = img_feat_path_load(img_feat_path_list)
# print(iid_to_img_feat_path)
img_feat = np.load(img_feat_path)
img_feat_x = img_feat['x'].transpose((1, 0))

img_feat_iter = torch.from_numpy(proc_img_feat(img_feat_x, __C.IMG_FEAT_PAD_SIZE))[np.newaxis, :]


img_feat_iter = img_feat_iter.cuda()
ques_ix_iter = ques_ix_iter.cuda()

print("img_feat_iter",img_feat_iter.size())

pred = net(
    img_feat_iter,
    ques_ix_iter
)
pred_np = pred.cpu().data.numpy()
pred_argmax = np.argmax(pred_np, axis=1)

print(pred_argmax)
print(dataset.ix_to_ans[str(pred_argmax[0])])
