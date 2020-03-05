import torch
import torch.nn as nn
import etw_pytorch_utils as pt_utils
from .LanguageModule import LanguageModule
from .PointNet2Backbone import PointNet2Backbone

class ReferModel(nn.Module):
    def __init__(self, cfg, embedder, p_feature_dim, l_hidden_dim):
        super(ReferModel, self).__init__()
        
        self.p_feature_dim = p_feature_dim
        self.l_hidden_dim = l_hidden_dim
        self.language_module = LanguageModule(cfg, embedder)
        self.backbone_net = PointNet2Backbone(cfg, input_feature_dim=self.p_feature_dim, lang_hidden_dim=self.l_hidden_dim)
                      
        self.box_proposals_layers = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=6, kernel_size=1, stride=1),
        )
        
        self.bbox_pred_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(6*16, 7),
        )
        
        self.interpolation_layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=4096, kernel_size=1, stride=1),
        )
                      
        self.inst_seg_pred_layers = (
            pt_utils.Seq(263)
            .conv1d(263, bn=True)
            .dropout()
            .conv1d(1, activation=None) # 0/1
        )
        
        
        
    def forward(self, pc, expr):
        
        l_features = self.language_module(expr) # BS x length x hidden size
        
        # global: BS x point feature size(1024) x points(16)
        # local: BS x point feature size(256) x points(1024)
        global_features, local_features = self.backbone_net(pc, l_features)
        
        ### BBox prediction
        box_proposals = self.box_proposals_layers(global_features) # BS x 6 x points (16)
        m_box_proposals = box_proposals.view((box_proposals.size(0), -1)) # BS x (6 * 16)
        
        pred_box = self.bbox_pred_layers(m_box_proposals) # BS x 7
                       
        ### Instance segmentation prediction
        # BS x points(1024) x point feature size(256)
        inv_local_features = local_features.transpose(1, 2)
        
        # BS x points(1024) x point feature size(7)
        expanded_pred_box = pred_box.unsqueeze(1).expand(pred_box.size(0), inv_local_features.size(1), pred_box.size(1))
        
        # BS x points(1024) x point feature size(263)
        combined_features = torch.cat((inv_local_features, expanded_pred_box), dim=2)
        
        interpolated_features = self.interpolation_layers(combined_features) # BS x point feature size(4096) x points(263)
        inv_interpolated_features = interpolated_features.transpose(1, 2) # BS x point feature size(263) x points(4096)
        instance_points = self.inst_seg_pred_layers(inv_interpolated_features) # BS x point feature size(1) x points(4096)
        
        ### Output
        pred_confidence = torch.sigmoid(pred_box[:,-1:])
        pred_box = pred_box[:,:-1]
        instance_points = instance_points.view((-1, instance_points.size(2))) # BS x points(4096)
        instance_points = torch.sigmoid(instance_points)
        
        return pred_confidence, pred_box, instance_points
    
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.getcwd())
    from src.models.LanguageModule import LanguageModule
    from src.models.PointNet2Backbone import PointNet2Backbone
    from src.models.WordEmbedding import Embedding
    from src.configs.config import CONF
    
    embedder = Embedding(CONF)
    model = ReferModel(CONF, embedder, p_feature_dim=3, l_hidden_dim=128).cuda()
    
    out1, out2, out3 = model(torch.rand(4,4096,6).cuda(), torch.randint(low=0, high=embedder.get_vocabulary_size(), size=(4,20)).cuda())
    print ('Output:')
    print (out1, out1.shape)
    print (out2, out2.shape)
    print (out3, out3.shape)