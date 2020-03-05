import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

def get_scene_plot(pc):
    
     return go.Scatter3d(x=pc[:,0], y=pc[:,1], z=pc[:,2],
                               marker=dict(
                                    size=2,
                                    color=pc[:,3:6],                # set color to an array/list of desired values
                                    opacity=1.0
                                ),
                               mode='markers')
    
def instance_seg_visualize(scene, class_id):
    
    pc = copy.deepcopy(scene["mesh_vertices"])
    instances = copy.deepcopy(scene["instance_labels"])
    semantics = copy.deepcopy(scene["semantic_labels"])
    
    target_semantics = np.zeros_like(semantics)
    target_semantics[semantics == class_id] = 1
    
    semantic_indices = np.argwhere(target_semantics)
    semantic_indices = semantic_indices.reshape(-1)
    
    target_instances = instances[semantic_indices]
    
    pc[semantic_indices, 3:6] = np.array([255,0,0]).reshape(1,3)
    
    scene_plot = get_scene_plot(pc)
    plots = [scene_plot]
    
    fig = go.Figure(data=plots)
    fig.show()