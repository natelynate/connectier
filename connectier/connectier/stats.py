import numpy as np
import pandas as pd

def add_intercentroid_edges(cntm, data:pd.DataFrame, subnetwork_id:str, node_id:str):
        grouped_by_subnetwork = pd.Series(data.groupby(subnetwork_id)[node_id].apply(list))
        intercluster_edge_array = np.zeros((len(cntm.centroids), len(cntm.centroids)))
        for idx, subnetwork in zip(grouped_by_subnetwork.index, grouped_by_subnetwork.values):
            cntm.subnetworks.append(subnetwork)
            for node in subnetwork:
                if node in cntm.nodes:
                    cntm.node_metadata[node]['weight'] += 1
            
            edges = [(subnetwork[idx], subnetwork[idx+1]) for idx in range(len(subnetwork)-1)]
            for edge in edges:
                src_node_label = cntm.node_metadata[edge[0]]['cluster_label']
                dst_node_label = cntm.node_metadata[edge[1]]['cluster_label']
                intercluster_edge_array[src_node_label][dst_node_label] += 1 # intercentroid travel frequency counts
        return intercluster_edge_array


def get_curve_trajectories(src, dst, direction:int, n_components:int=50):
    # Caculate curve line
    straight_edge = np.linspace(src[1], dst[1], n_components)
    a = (dst[1] - src[1]) / (np.cosh(dst[0]) - np.cosh(src[0]))
    b = src[1] - a * np.cosh(src[0])
    x = np.linspace(src[0], dst[0], n_components)
    y = a * np.cosh(x) + b

    # 이상치 때문에 임시조치함
    for i, v in enumerate(y):
        if y[i] > 7 or y[i]< -7:
            y[i] = y[i - 1]
             
    if direction == 1:
        return x, y
    else:
        y = straight_edge + (straight_edge - y)[::-1]
        for i, v in enumerate(y):
            if y[i] > 7 or y[i] < -7:
                y[i] = y[i - 1]
        
        return x, y