from pointnet2.utils.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
import torch
import torch.nn as nn
import etw_pytorch_utils as pt_utils

class PointNet2Backbone(nn.Module):
    def __init__(self, cfg, input_feature_dim, lang_hidden_dim, use_xyz=True):
        super(PointNet2Backbone, self).__init__()
        
        self.input_feature_dim = input_feature_dim
        self.expr_feat_dim = lang_hidden_dim

        c_in = input_feature_dim
        # --------- 4 SET ABSTRACTION LAYERS ---------
        self.sa1 = PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=use_xyz,
            )
        
        c_out_0 = 32 + 64

        c_in = c_out_0

        self.sa2 = PointnetSAModuleMSG(
                npoint=256,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=use_xyz,
            )
        
        c_out_1 = 128 + 128 + self.expr_feat_dim

        ##--- Here insert expr ---##

        c_in = c_out_1

        self.sa3 = PointnetSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
                use_xyz=use_xyz,
            )
        c_out_2 = 256 + 256

        c_in = c_out_2

        self.sa4 = PointnetSAModuleMSG(
                npoint=16,
                radii=[0.4, 0.8],
                nsamples=[16, 32],
                mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
                use_xyz=use_xyz,
            )
        c_out_3 = 512 + 512
        
        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        self.fp1 = PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512])
        self.fp2 = PointnetFPModule(mlp=[512 + c_out_1, 512, 512])
        self.fp3 = PointnetFPModule(mlp=[512 + c_out_0, 256, 256])
        self.fp4 = PointnetFPModule(mlp=[256 + self.input_feature_dim, 128, 128])
           
    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pc, l_features):
        """
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            data_dict: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        
        xyz, features = self._break_up_pc(pc)

        ### --------- 4 SET ABSTRACTION LAYERS ---------
        sa1_xyz, sa1_features = self.sa1(xyz, features)
        sa2_xyz, sa2_features = self.sa2(sa1_xyz, sa1_features) # this fps_inds is just 0,1,...,1023
        
        ### Fuse point features & language features
        enc = l_features.unsqueeze(2)
        sa2_features = torch.cat((sa2_features, enc.expand(enc.size(0), enc.size(1), sa2_features.size(2))), dim=1)
        
        sa3_xyz, sa3_features = self.sa3(sa2_xyz, sa2_features) # this fps_inds is just 0,1,...,511
        sa4_xyz, sa4_features = self.sa4(sa3_xyz, sa3_features) # this fps_inds is just 0,1,...,255
        
        ### --------- 2 FEATURE UPSAMPLING LAYERS --------
        local_features = self.fp1(sa3_xyz, sa4_xyz, sa3_features, sa4_features)
        local_features = self.fp2(sa2_xyz, sa3_xyz, sa2_features, local_features)
        local_features = self.fp3(sa1_xyz, sa2_xyz, sa1_features, local_features)
        local_features = self.fp4(xyz, sa1_xyz, features, local_features)
        
        # output_xyz = sa2_xyz
        # num_seed = output_xyz.shape[1]
        # output_indices = sa1_fps_inds[:,0:num_seed] # indices among the entire input point clouds
        # output_features = features
        
        global_features = sa4_features
        
        return global_features, local_features

if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.getcwd())
    from src.configs.config import CONF as cfg
    
    pointnet_backbone = PointNet2Backbone(cfg, input_feature_dim=3, lang_hidden_dim=128).cuda()
    print (pointnet_backbone)
    out1, out2 = pointnet_backbone(torch.rand(16,4096,6).cuda(), torch.rand(16,128).cuda())
    print (out1.shape)
    print (out2.shape)