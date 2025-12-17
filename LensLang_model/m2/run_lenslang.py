import collections
import json
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader_lenslang import *
from utils.Trainer import *
from model.FakingRecipe_lenslang import *

class Run():
    def __init__(self,config):
        self.dataset = config['dataset']
        self.mode = config['mode']
        self.epoches = config['epoches']
        self.batch_size = config['batch_size']
        self.early_stop = config['early_stop']
        self.device = config['device']
        self.lr = config['lr']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.path_ckp=config['path_ckp']
        self.path_tb=config['path_tb']
        self.inference_ckp=config['inference_ckp']

        self.use_lenslang = config.get('use_lenslang', False)

        # 如果 config 显式给了 lenslang_root，就用用户的；
        # 否则根据 dataset 自动选一个默认根目录
        cfg_root = config.get('lenslang_root', None)
        if cfg_root is not None:
            self.lenslang_root = cfg_root
        else:
            if self.dataset == 'fakesv':
                self.lenslang_root = './fea/fakesv/preprocess_lenslang'
            elif self.dataset == 'fakett':
                self.lenslang_root = './fea/fakett/preprocess_lenslang'
            else:
                # 理论上不会走到这里，防御性写法
                self.lenslang_root = None

    def get_dataloader(self,data_path):
        dataset=FakingRecipe_Dataset(data_path,
                                     self.dataset,
                                     use_lenslang=self.use_lenslang,
                                     lenslang_root=self.lenslang_root)
        collate_fn=collate_fn_FakeingRecipe
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=0,
                                collate_fn=collate_fn)
        return dataloader

    def main(self):
        self.model = FakingRecipe_Model(self.dataset)
        
        if self.mode=='train':
            if self.dataset=='fakesv':
                data_split_dir='./data/FakeSV/data-split/'
                save_predict_result_path='./predict_result/FakeSV/'
            elif self.dataset=='fakett':
                data_split_dir='./data/FakeTT/data-split/'
                save_predict_result_path='./predict_result/FakeTT/'
            
            train_data_path=data_split_dir+'vid_time3_train.txt'
            test_data_path=data_split_dir+'vid_time3_test.txt'
            val_data_path=data_split_dir+'vid_time3_val.txt'

            data_load_time_start = time.time()
            train_dataloader=self.get_dataloader(train_data_path)
            test_dataloader=self.get_dataloader(test_data_path)
            val_dataloader=self.get_dataloader(val_data_path)
            dataloaders=dict(zip(['train','test','val'],[train_dataloader,test_dataloader,val_dataloader]))
            print ('data load time: %.2f' % (time.time() - data_load_time_start))
            trainer=Trainer(model=self.model,device=self.device,lr=self.lr,dataloaders=dataloaders,epoches=self.epoches,model_name='FakingRecipe',save_predict_result_path=save_predict_result_path,beta_c=self.alpha,beta_n=self.beta,early_stop=self.early_stop,save_param_path=self.path_ckp+self.dataset+"/",writer=SummaryWriter(self.path_tb+self.dataset+"/"))
            ckp_path=trainer.train()
            result=trainer.test(ckp_path)
        elif self.mode=='inference_test':
            if self.dataset=='fakesv':
                test_file='./data/FakeSV/data-split/vid_time3_test.txt'
                save_predict_result_path='./predict_result/FakeSV/'
            elif self.dataset=='fakett':
                test_file='./data/FakeTT/data-split/vid_time3_test.txt'
                save_predict_result_path='./predict_result/FakeTT/'
            dataloader=self.get_dataloader(test_file)
            inferncer=Inferencer(model=self.model,device=self.device,model_name='FakingRecipe',dataset=self.dataset,dataloader=dataloader,save_predict_result_path=save_predict_result_path)
            result=inferncer.inference(self.inference_ckp)
