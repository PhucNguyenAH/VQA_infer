from __init__ import PYTHON_PATH

import yaml
from cfgs.base_cfgs import Cfgs
from core.model.net import Net
from core.data.load_data import DataSet
from core.data.data_utils import tokenize,tokenize_question, ans_stat, proc_ques, img_feat_path_load, proc_img_feat


import os, json, torch, datetime, pickle, copy, shutil, time
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data as Data

class Inference:
    def __init__(self, __C, pretrained_emb, token_size, ans_sizes, state_dict,dataset):
        self.__C = __C
        self.net = Net(__C, pretrained_emb, token_size, ans_size)
        self.net.cuda()
        self.net.eval()
        self.net.load_state_dict(state_dict)
        self.dataset = dataset
    
    def infer(self,question,img_feat):
        token_to_ix, emb = tokenize_question(question, self.__C.USE_GLOVE)
        ques_ix_iter = torch.from_numpy(proc_ques(question, token_to_ix, self.__C.MAX_TOKEN))[np.newaxis, :]
        img_feat_x = img_feat['x'].transpose((1, 0))

        img_feat_iter = torch.from_numpy(proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE))[np.newaxis, :]


        img_feat_iter = img_feat_iter.cuda()
        ques_ix_iter = ques_ix_iter.cuda()

        pred = self.net(img_feat_iter, ques_ix_iter)
        pred_np = pred.cpu().data.numpy()
        pred_argmax = np.argmax(pred_np, axis=1)

        print(pred_argmax)
        print(self.dataset.ix_to_ans[str(pred_argmax[0])])

if __name__ == '__main__':
    cfg_file = os.path.join(PYTHON_PATH, "ailibs_data/modular_coattention/large_model.yml")
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.load(f)
    __C = Cfgs()
    __C.add_args(yaml_dict)
    __C.proc()

    dataset = DataSet(__C)
    data_size = dataset.data_size
    token_size = dataset.token_size
    ans_size = dataset.ans_size
    pretrained_emb = dataset.pretrained_emb
    path = os.path.join(PYTHON_PATH, "ailibs_data/modular_coattention/model.pkl")
    print('Loading ckpt {}'.format(path))
    state_dict = torch.load(path)['state_dict']
    print('Finish!')

    inference = Inference(__C, pretrained_emb, token_size, ans_size,state_dict,dataset)

    question = "Does the door open or close?"
    img_feat_path = os.path.join(PYTHON_PATH,"datasets/coco_extract/test2015/COCO_test2015_000000010868.jpg.npz")
    img_feat = np.load(img_feat_path)

    inference.infer(question, img_feat)