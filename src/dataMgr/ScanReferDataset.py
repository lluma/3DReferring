from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import sys
import json
import numpy as np
from tqdm import tqdm
from tqdm import tnrange, tqdm_notebook
from multiprocessing import Pool
sys.path.insert(0, os.getcwd())
from data.scannet.model_util_scannet import rotate_aligned_boxes, ScannetDatasetConfig, rotate_aligned_boxes_along_axis
from utils.pc_utils import random_sampling, rotx, roty, rotz
 
'''
[ Data format for ScanRefer ]
Current ScanRefer only has the referring descriptions of the scenes indexed with scenexxxx_00, and the total number of scene it used is 703 over 1513 (# ScanNet all scenes)

- **scene_id** (str) scene index as "scene{scene_id}_{split_id}". (ex. "scene0000_00") 
- **object_id** (str) object index in current scene corresponding to the labels in ScanNet. (ex. "4")
- **object_name** (str) object's class name. (ex. cabinet)
- **ann_id** (str) description index for the same object in current scene.
- **description** (str) a referring expresssion to this object in current scene.
- **token** (array) the tokens of the description.
'''

class ScanRefer(object):
    
    def __init__(self, cfg, embedder, num_scenes=-1, split='train'):
        self.cfg = cfg
        self.embedder = embedder
        self.num_scenes = num_scenes
        self.split = split
        
        # Load
        if self.split == 'train':
            self.train = json.load(open(self.cfg.PATH.SCANREFER_TRAIN))
            self.scene_list = sorted(list(set([data["scene_id"] for data in self.train])))
            self.data = self._get_filtered_train_data()
        else:
            self.data = json.load(open(self.cfg.PATH.SCANREFER_VAL))
            self.scene_list = sorted(list(set([data["scene_id"] for data in self.data])))

        self.token_to_indices()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def token_to_indices(self):
        tnrange = tqdm(enumerate(self.data), total=len(self.data), desc='ScanRefer word indexing')
        for idx, data in tnrange:
            data['indexed_token'] = [self.embedder.to_index(word) for word in data['token']]
        
    def _get_filtered_train_data(self):
        
        if self.num_scenes == -1: 
            self.num_scenes = len(self.scene_list)
        else:
            assert len(self.scene_list) >= self.num_scenes
            
        # Slice train_scene_list
        self.filtered_train_scene_list = self.scene_list[:self.num_scenes]
        
        # Filter data in chosen scenes
        new_scanrefer_train = []
        for data in self.train:
            if data["scene_id"] in self.filtered_train_scene_list:
                new_scanrefer_train.append(data)
                
        return new_scanrefer_train

'''
[ Data format for ScanNet ]
**Each Scene in scene_data:**
- **mesh_vertices** Point ( position, color, normal ): [ x, y, z, r, g, b, n_x, n_y, n_z ] * 50000 points .
- **instance_labels** Point ( instance index ): [ labeled instance index for this point ] * 50000 points .
- **semantic_labels** Point ( semantic category index ): [ labeled category index for this point ] * 50000 points .
- **instance_bboxes** Box ( center_position, x-axis length, y-axis length, z-axis length, semantic label, instance label ): [ c_x, c_y, c_z, dx, dy, dz, semantic_id, instance_id] * #instances in this scene .

**Labeled classes map:**
- id 
- raw_category
- category 
- count
- nyu40id
- eigen13id
- nyuClass
- nyu40class
- eigen13class
- ModelNet40
- ModelNet10
- ShapeNetCore55
- synsetoffset
- wnsynsetid
- wnsynsetkey
- mpcat40
- mpcat40index
''' 

class ReferredScanNetDataset(Dataset):
    def __init__(self, cfg, scanrefer, embedder, split="train", augment=False, debug=False):
        
        super(ReferredScanNetDataset, self).__init__()
        self.cfg = cfg
        self.scanrefer = scanrefer
        self.embedder = embedder
        self.split = split
        self.augment = augment
        self.debug = debug
        self.max_len = self.cfg.TRAINING.MAX_LEN
        self.sos_token = float(embedder.to_index('<sos>'))
        self.eos_token = float(embedder.to_index('<eos>'))
        self.pad_token = float(embedder.to_index('<pad>'))
        self.scene_list = self.scanrefer.scene_list if self.scanrefer else self._load_scannet_scene_list()
        
        with open(self.cfg.PATH.SCANNET_V2_TSV) as f:
            lines = [line.rstrip() for line in open(self.cfg.PATH.SCANNET_V2_TSV)]
            # Remove the first row about the data titles.
            self.classes_map = lines[1:]
           
        self._load_scannet_data()
        
        self.num_points = self.cfg.TRAINING.NUM_POINTS
        self.max_num_obj = self.cfg.TRAINING.MAX_NUM_OBJ
        self.use_color = self.cfg.TRAINING.USE_COLOR
        
        self._prepare_scannet_data()
        
    def __len__(self):
        return len(self.sample_data)
    
    def __getitem__(self, idx):
        return self.sample_data[idx]
  
    
    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]
        
        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox
    
    def _load_scannet_scene_list(self):
        with open(self.cfg.PATH.SCANNET_SCENE_LIST, "r") as f:
            scene_list = [row.strip() for row in f.readlines()]
            f.close()
        return scene_list
    
    def _get_raw2nyuid(self):
        
        raw2nyuid = {}
        for line in self.classes_map:
            ele = line.split('\t')
            raw_name = ele[1]
            nyu40_id = int(ele[4])
            raw2nyuid[raw_name] = nyu40_id
            
        return raw2nyuid
            
    def _get_raw2label(self):
        '''
            ScanNet label classes in DC.type2class (18 classes): 
            0  'cabinet',
            1  'bed',
            2  'chair',
            3  'sofa',
            4  'table',
            5  'door',
            6  'window',
            7  'bookshelf', 
            8  'picture',
            9  'counter',
            10 'desk',
            11 'curtain',
            12 'refrigerator',
            13 'shower curtain', 
            14 'toilet',
            15 'sink',
            16 'bathtub', 
            17 'others'
        '''
        ### Mapping NYU40 labeled classes to ScanNet labeled classes.
        DC = ScannetDatasetConfig()
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}
        
        raw2label = {}
        for line in self.classes_map:
            label_classes_set = set(scannet_labels)
            ele = line.split('\t')
            raw_name = ele[1]
            nyu40_name = ele[7]
            
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label
    
    def _load_scannet_data(self):
        
        print ('Loading ScanNet data...')
        self.scene_data = {}
        
        ### Loading mesh data (mesh_vertices) and the annotation data
        #  (instance_labels, semantic_labels, instance_bboxes) we need.
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(self.cfg.PATH.SCANNET_DATA, scene_id)+"_vert.npy")
            self.scene_data[scene_id]["instance_labels"] = np.load(os.path.join(self.cfg.PATH.SCANNET_DATA, scene_id)+"_ins_label.npy")
            self.scene_data[scene_id]["semantic_labels"] = np.load(os.path.join(self.cfg.PATH.SCANNET_DATA, scene_id)+"_sem_label.npy")
            self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(self.cfg.PATH.SCANNET_DATA, scene_id)+"_bbox.npy")
        
        ### Loading the class mapping for different subset dataset settings
        
        ## ScanNet classes indices to NYU40 classes indices.
        self.raw2nyuid = self._get_raw2nyuid()
        self.raw2label = self._get_raw2label()
    
    def _preprocess_sample(self, data):
        
        ### Get the data information in annotation item.
        scene_id = data['scene_id']
        object_id = int(data['object_id'])
        object_name = " ".join(data["object_name"].split("_"))
        ann_id = int(data["ann_id"])
        
        ### Get the referring expression
        description = data["indexed_token"]
        if len(description) > self.max_len:
            description = description[:self.max_len]
        else:
            description = description + [self.pad_token]*(self.max_len - len(description))
        description = [self.sos_token] + description + [self.eos_token]
        
        original_description = data['description']
        
        ### Get the original annotation data.
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]
        instance_labels = self.scene_data[scene_id]["instance_labels"]
        semantic_labels = self.scene_data[scene_id]["semantic_labels"]
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
        
        ### Get point cloud data
        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            
            # Point cloud centering
            point_cloud[:,:3] = point_cloud[:,:3] - point_cloud[:,:3].mean(axis=0, keepdims=True)
            
            # Point cloud RGB scaling
            point_cloud[:,3:] = (point_cloud[:,3:] - self.cfg.TRAINING.MEAN_COLOR_RGB) / 255.0
            # point_cloud[:,3:] = point_cloud[:,3:] * 2.7 / 255.0
            
            pcl_color = point_cloud[:,3:]
        
        ### Sampling points
        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)        
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]
        
        ### Specify the number of box we need to predict and create a mask for it.
        # num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < self.max_num_obj else self.max_num_obj
        # target_bboxes_mask = np.zeros((self.max_num_obj))
        # target_bboxes_mask[0:num_bbox] = 1
        # target_bboxes = instance_bboxes[:num_bbox, 0:6]
        target_bboxes = instance_bboxes
        
        ### Data augmentation (*Warning: after augmenting, target_boxes will be left only 6 element dimension)
        if self.augment and not self.debug:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                target_bboxes[:,0] = -1 * target_bboxes[:,0]                
                
            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:,1] = -1 * point_cloud[:,1]
                target_bboxes[:,1] = -1 * target_bboxes[:,1]                                

            # Rotation along X-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = rotx(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "x")

            # Rotation along Y-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = roty(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "y")

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = rotz(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "z")

            # Translation
            point_cloud, target_bboxes = self._translate(point_cloud, target_bboxes)
            
            target_bboxes = np.concatenate([target_bboxes, instance_bboxes], axis=1)
        
        ### Build up referred targets' labels 
        sample = {}
        for idx, bbox in enumerate(target_bboxes):
            if int(bbox[-1]) == object_id:
                gt_instance_id = bbox[-1]
                gt_semantic_id = bbox[-2]

                x_min = (2*bbox[0] - bbox[3])/2
                x_max = (2*bbox[0] + bbox[3])/2
                y_min = (2*bbox[1] - bbox[4])/2
                y_max = (2*bbox[1] + bbox[4])/2
                z_min = (2*bbox[2] - bbox[5])/2
                z_max = (2*bbox[2] + bbox[5])/2
                
                sample['point_cloud'] = point_cloud
                sample['object_name'] = object_name
                sample['corners'] =  np.array([x_min, y_min, z_min, x_max, y_max, z_max]).astype(np.float32)
                sample['class_id'] = float(bbox[-2])
                
                instance_seg = np.zeros_like(instance_labels)
                instance_seg[instance_labels == gt_instance_id] = 1
                sample['instance_seg'] = instance_seg.astype(np.float32)
                sample['description'] = np.array(description).astype(np.float32)
                sample['original_description'] = original_description
                
        return sample
    
    def _preprocess_samples(self, batch_start, batch_end):
        processed = []
        datas = self.scanrefer[batch_start: batch_end]
        tnrange = tqdm(range(len(datas)), total=len(datas))
        
        for idx in tnrange:
            processed_sample = self._preprocess_sample(datas[idx])
            processed.append(processed_sample)
            
        return processed
    
    def _prepare_scannet_data(self, n_workers=8):
        
        print ('Preparing ScanRefer %s data...' % (self.split))
        results = [None] * n_workers
        
        with Pool(processes=n_workers) as pool:
            for i in range(n_workers):
                batch_start = (len(self.scanrefer) // n_workers) * i
                if i == n_workers - 1:
                    batch_end = len(self.scanrefer)
                else:
                    batch_end = (len(self.scanrefer) // n_workers) * (i + 1)

                # batch = self.scanrefer[batch_start: batch_end]
                results[i] = pool.apply_async(self._preprocess_samples, args=(batch_start, batch_end))

            pool.close()
            pool.join()
            
        self.sample_data = []
        for result in results:
            self.sample_data += result.get()
            
        return self.sample_data  
            
if __name__ == '__main__':
    
    import torch
    from src.models.WordEmbedding import Embedding
    from src.configs.config import CONF
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print ('Cuda device cannot use, please tackle down this error!')
        sys.exit(0)
    
    embedder = Embedding(CONF)
    scanrefer_train = ScanRefer(CONF, embedder=embedder, num_scenes=-1, split='train')
    # scanrefer_val = ScanRefer(CONF, embedder=embedder, num_scenes=-1, split='val')
    
    train_dataset = ReferredScanNetDataset(CONF, scanrefer=scanrefer_train, embedder=embedder, split=scanrefer_train.split, augment=False, debug=False)
    # scannet_valData = ReferredScanNetDataset(CONF, scanrefer=scanrefer_val, embedder=embedder, split=scanrefer_val.split, augment=False, debug=False)  

    ######
    sys.exit(0)
    ######
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    tnrange = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Train')
        
    ### Training
    for i, sample in tnrange:
        x_pc = sample['point_cloud'].to(device)
        x_expr = sample['description'].to(device)
    