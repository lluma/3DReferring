"""
#### Data format for bounding box
- **center_x** = (**x_min** + **x_max**) / 2
- **center_y** = (**y_min** + **y_max**) / 2
- **center_z** = (**z_min** + **z_max**) / 2
- **length_x** = (**x_max** - **x_min**)
- **length_y** = (**y_max** - **y_min**)
- **length_z** = (**z_max** - **z_min**)
"""
import numpy as np

class Metrics(object):
    
    def __init__(self):
        
        self.n_hit = 0
        self.n_bbox = 0
        self.accs = np.array([])
        
        self.n_precision = np.array([])
        self.n_recall = np.array([])
        self.n_correct = np.array([])
        
        self.precisions = None
        self.recalls = None
        self.IoUs = None
    
    def _convert_vertices2center_length(self, vertices):
        min_vertex = vertices[:,:3]
        max_vertex = vertices[:,3:6]
        
        output = np.zeros_like(vertices)
        output[:,0] = min_vertex[:,0] + max_vertex[:,0] / 2.0
        output[:,1] = min_vertex[:,1] + max_vertex[:,1] / 2.0
        output[:,2] = min_vertex[:,2] + max_vertex[:,2] / 2.0
        output[:,3] = max_vertex[:,0] - min_vertex[:,0]
        output[:,4] = max_vertex[:,1] - min_vertex[:,1]
        output[:,5] = max_vertex[:,2] - min_vertex[:,2]
        
        return output
    
    def _compute_3d_iou(self, x, y):
        """ 
        Helper functions for calculating 2D and 3D bounding box IoU.
        From: https://github.com/facebookresearch/votenet/blob/master/utils/box_util.py

        Collected and written by Charles R. Qi
        Last modified: Jul 2019
        """
        x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1 = x[:,0], x[:,3], x[:,1], x[:,4], x[:,2], x[:,5]
        x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2 = y[:,0], y[:,3], y[:,1], y[:,4], y[:,2], y[:,5]
        
        xA = np.maximum(x_min_1, x_min_2)
        yA = np.maximum(y_min_1, y_min_2)
        zA = np.maximum(z_min_1, z_min_2)
        xB = np.minimum(x_max_1, x_max_2)
        yB = np.minimum(y_max_1, y_max_2)
        zB = np.minimum(z_max_1, z_max_2)
        inter_vol = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0) * np.maximum((zB - zA + 1), 0)
        box_vol_1 = (x_max_1 - x_min_1 + 1) * (y_max_1 - y_min_1 + 1) * (z_max_1 - z_min_1 + 1)
        box_vol_2 = (x_max_2 - x_min_2 + 1) * (y_max_2 - y_min_2 + 1) * (z_max_2 - z_min_2 + 1)
        iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-20)

        return iou
    
    def compute_bbox_metrics(self, pred_bbox, gt_bbox):
        
        iou = self._compute_3d_iou(pred_bbox, gt_bbox)
        hit = iou > 0.25
        self.n_hit += np.sum(hit.astype(np.int32))
        self.n_bbox += hit.size
        acc = self.n_hit / (self.n_bbox + 1e-20)
        self.accs = np.append(self.accs, acc)
        
        return acc
    
    def compute_inst_seg_metrics(self, pred_inst_seg, gt_inst_seg):
        '''
            Input:
                pred_inst_seg (B x N): The predicted output result.
                gt_inst_seg (B x N): The groundTruth per-point mask.
            Output:
                precision (B): Accumulated precision for current training.
                recall (B): Accumulated recall for current training.
                precision (B): Accumulated precision for current training.
        '''
        pred_inst_seg = pred_inst_seg > 0.25 # B x N
        
        self.n_precision = np.append(self.n_precision, np.sum(pred_inst_seg.astype(np.int32), axis=1)) # B
        self.n_recall = np.append(self.n_recall, np.sum(gt_inst_seg.astype(np.int32), axis=1)) # B
        self.n_correct = np.append(self.n_correct, np.sum(pred_inst_seg.astype(np.int32) * gt_inst_seg.astype(np.int32), axis=1)) # B
           
        precision = self.n_correct / (self.n_precision + 1e-20) # B * K
        recall = self.n_correct / (self.n_recall + 1e-20) # B * K
        IoU = self.n_correct / ((self.n_precision + self.n_recall - self.n_correct) + 1e-20) # B * K
            
        self.precisions = precision if self.precisions is None else np.concatenate((self.precisions, precision), axis=0) # K x B
        self.recalls = recall if self.recalls is None else np.concatenate((self.recalls, recall), axis=0) # K x B
        self.IoUs = IoU if self.IoUs is None else np.concatenate((self.IoUs, IoU), axis=0) # K x B
        
        return precision, recall, IoU
    
    def compute_final_metrics(self):
        
        # m_acc = np.sum(self.accs) / (np.asarray(self.accs).size + 1e-20)
        m_acc = self.n_hit / (self.n_bbox + 1e-20)
        
        # m_prec = np.sum(self.precisions) / (np.asarray(self.precisions).size + 1e-20)
        m_prec = np.mean(self.n_correct / (self.n_precision + 1e-20))
        # m_prec = np.median(self.precisions)
        
        # m_rec = np.sum(self.recalls) / (np.asarray(self.recalls).size + 1e-20)
        m_rec = np.mean(self.n_correct / (self.n_recall + 1e-20))
        # m_rec = np.median(self.recalls)
        
        # m_IoU = np.sum(self.IoUs) / (np.asarray(self.IoUs).size + 1e-20)
        m_IoU = np.mean(self.n_correct / ((self.n_precision + self.n_recall - self.n_correct) + 1e-20))
        # m_IoU = np.mean(self.IoUs)
        
        return m_acc, m_prec, m_rec, m_IoU
    
    def reset(self):
        
        self.n_hit = 0
        self.n_bbox = 0
        self.accs = np.array([])
        
        self.n_precision = np.array([])
        self.n_recall = np.array([])
        self.n_correct = np.array([])
        
        self.precisions = None
        self.recalls = None
        self.IoUs = None
    
if __name__ == "__main__":
    
    metrics = Metrics()
    
    b_labels = np.random.randn(5, 6)
    y_bbox = np.random.randn(5, 6)
    print ('bbox acc:', metrics.compute_bbox_metrics(b_labels, y_bbox))
    print ('===========================================================')
    import torch
    i_labels = np.array([[0,0,1,1,1],
                         [0,1,1,0,0],
                         [0,1,0,0,1],
                         [0,1,0,0,1],
                         [1,0,1,0,0]])
    
    y_inst_seg = np.array([[0,0,1,0,0],
                           [0,1,1,1,0],
                           [0,0,1,0,1],
                           [0,0,0,0,1],
                           [1,1,1,1,0]])
   
    
    print ('inst. seg.:', metrics.compute_inst_seg_metrics(i_labels, y_inst_seg))
    print ('===========================================================')
    
    i_labels = np.array([[0,0,1,1,1],
                         [0,1,1,0,0],
                         [0,1,0,0,1],
                         [0,1,0,0,1],
                         [1,0,1,0,0],
                         [1,0,0,1,1],
                         [0,1,1,1,0]])
    
    y_inst_seg = np.array([[0,0,1,0,0],
                           [0,1,1,1,0],
                           [0,0,1,0,1],
                           [0,0,0,0,1],
                           [0,1,1,0,0],
                           [1,0,0,1,0],
                           [1,0,1,1,0]])
    
    
    print ('inst. seg.:', metrics.compute_inst_seg_metrics(i_labels, y_inst_seg))
    print ('===========================================================')
    
    print ('all:', metrics.compute_final_metrics())
    