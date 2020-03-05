import os
import sys
from tqdm import tqdm
from tqdm import tnrange, tqdm_notebook
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from .dataMgr.ScanReferDataset import ScanRefer, ReferredScanNetDataset
from .configs.config import CONF
from .models.WordEmbedding import Embedding
from .models.ReferModel import ReferModel
from .metrics.Metrics import Metrics
from .losses.PointLoss import FocalLoss

class Solver(object):
    
    def __init__(self, args):
        ### Parameters
        self.epochs = args.epoch
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.cfg = CONF
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        
        ### Word Embedder
        self.embedder = Embedding(self.cfg)
        
        ### Datasets
        self._get_datasets()
        
        ### Model
        self._get_model()
        
        ### Criterions & Optimizers
        self._get_loss_and_optim()
        self.bb_loss = 0
        self.bs_loss = 0
        self.i_loss = 0
        self.t_loss = 0
        
        ### Metrics
        self.metrics = Metrics()
        self.metrics_history = {}
        
        ### SummaryWriter for tensorboard
        log_dir = os.path.join("outputs", self.cfg.PATH.CHECKPOINT_PREFIX)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        
        ### Use multiple GPUs
        if torch.cuda.device_count() > 1 and self.cfg.TRAINING.MULTIGPU:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)

    def train(self):
        for epoch in range(self.epochs):
            self._run_epoch(epoch, 'train')
            
            if (epoch+1)%2 == 0:
                self._run_epoch(epoch, 'valid')
    
    def test(self):
        pass
    
    def _run_iter(self, sample, phrase):
        x_pc = sample['point_cloud'].to(self.device)
        x_expr = sample['description'].type(torch.LongTensor).to(self.device)
        y_bbox = sample['corners'].to(self.device)
        y_inst_seg = sample['instance_seg'].to(self.device)
        
        o_labels, b_labels, i_labels = self.model(x_pc, x_expr)
        bbox_loss = self.bbox_criterion(b_labels, y_bbox)
        inst_loss = self.inst_seg_criterion(i_labels, y_inst_seg)
        
        bbox_acc = self.metrics.compute_bbox_metrics(b_labels.detach().cpu().numpy(), y_bbox.detach().cpu().numpy())
        inst_seg_prec, inst_seg_rec, inst_seg_iou = self.metrics.compute_inst_seg_metrics(i_labels.detach().cpu().numpy(), y_inst_seg.detach().cpu().numpy())
        
        self.metrics_history['bbox_acc'] = bbox_acc
        self.metrics_history['inst_seg_prec'] = np.max(inst_seg_prec)
        self.metrics_history['inst_seg_rec'] = np.max(inst_seg_rec)
        self.metrics_history['inst_seg_iou'] = np.max(inst_seg_iou)
        
        total_loss = bbox_loss + 0.1 * inst_loss

        self.bb_loss += bbox_loss.item()
        self.i_loss += inst_loss.item()
        self.t_loss += total_loss.item()

        if phrase == 'train':
            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()
    
    def _run_epoch(self, epoch, phrase):
        
        if phrase == 'train':
            train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
            tnrange = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Train')
        elif phrase == 'valid':
            val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
            tnrange = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='Valid')
            
        self.bb_loss = 0
        self.bs_loss = 0
        self.i_loss = 0
        self.t_loss = 0
        self.metrics.reset()
        
        for i, sample in tnrange:
            self._run_iter(sample, phrase)
            
            m_acc, m_prec, m_rec, m_IoU = self.metrics.compute_final_metrics()
            
            tnrange.set_postfix(bb_loss=self.bb_loss/(i+1), 
                                i_loss=self.i_loss/(i+1), 
                                t_loss=self.t_loss/(i+1), 
                                x_acc=self.metrics_history['bbox_acc'], 
                                x_iou=self.metrics_history['inst_seg_iou'])
            
            self.writer.add_scalar('Average bbox loss', self.bb_loss/(i+1), global_step=epoch)
            self.writer.add_scalar('Average inst loss', self.i_loss/(i+1), global_step=epoch)
            self.writer.add_scalar('Average total loss', self.t_loss/(i+1), global_step=epoch)
            self.writer.add_scalar('Bbox mean Accuracy', m_acc, global_step=epoch)
            self.writer.add_scalar('IS mean Precision', m_prec, global_step=epoch)
            self.writer.add_scalar('IS mean Recall', m_rec, global_step=epoch)
            self.writer.add_scalar('IS mean IoU', m_IoU, global_step=epoch)
            
        print ('Bbox mean accuracy:', m_acc)
        print ('Inst. Seg. mean precision:', m_prec)
        print ('Inst. Seg. mean recall:', m_rec)
        print ('Inst. Seg. mean IoU:', m_IoU)
        print ('==============================================================')
        
        if phrase == 'train':
            self._save_model(epoch)
            
    def _compute_metrics(self, o_labels, b_labels, i_labels, y_bbox, y_inst_seg):
        
        # self.metrics.compute_bbox_metrics()
        
        pass
    
    def _get_datasets(self):
        scanrefer_train = ScanRefer(self.cfg, embedder=self.embedder, num_scenes=-1, split='train')
        scanrefer_val = ScanRefer(self.cfg, embedder=self.embedder, num_scenes=-1, split='val')
        
        self.train_dataset = ReferredScanNetDataset(self.cfg, scanrefer=scanrefer_train, embedder=self.embedder, split=scanrefer_train.split, augment=False, debug=False)
        
        self.val_dataset = ReferredScanNetDataset(self.cfg, scanrefer=scanrefer_val, embedder=self.embedder, split=scanrefer_val.split, augment=False, debug=False) 
    
    def _get_model(self):
        self.model = ReferModel(self.cfg, self.embedder, p_feature_dim=3, l_hidden_dim=128).to(self.device)
    
    def _get_loss_and_optim(self):
        self.bbox_criterion = nn.MSELoss()
        # self.bbscore_criterion = BBscoreLoss()
        # self.inst_seg_criterion = nn.BCELoss()
        self.inst_seg_criterion = FocalLoss(alpha=self.cfg.TRAINING.FOCAL_LOSS.ALPHA, gamma=self.cfg.TRAINING.FOCAL_LOSS.GAMMA)
        
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
    def _save_model(self, epoch):
        model_dirs = os.path.join("models", self.cfg.PATH.SAVE_PATH)
        if not os.path.exists(model_dirs):
            os.makedirs(model_dirs)
            
        model_path = os.path.join(model_dirs, 'model.pkl.' + str(epoch))
        torch.save(self.model.state_dict(), model_path)

if __name__ == "__main__":
    
    solver = Solver()