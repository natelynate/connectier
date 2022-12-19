from stats import get_curve_trajectories, add_intercentroid_edges
from seaborn import axes_style
import seaborn as sns
import seaborn.objects as so
import pandas as pd 
import numpy as np


def graph2d(cntm, subnetwork_id=None, topic_names:dict=None, figsize_=(20, 16)):
    """
    Visualize the current network stored in the FullNetwork object.
    """
    # Create Color Dictionary
    color_palette = {label:color for label, color in zip(topic_names.keys(), sns.color_palette().as_hex())} 

    # Initialize seaborn's Plot Object
    plot = so.Plot().layout(size=figsize_).label(title=f"Full Network Visualized with {subnetwork_id}th User Subnetwork Highlighted ").theme({**axes_style("ticks")})

    # Plot cluster centroids
    centroids = cntm.centroids
    centroid_data = pd.DataFrame(data=cntm.centroids.squeeze(), columns=['x', 'y'])
    centroid_data['topic'] = pd.Series(topic_names.values())
    plot = plot.add(so.Dots(alpha=1), data=centroid_data, x='x', y='y', text='topic')
    plot = plot.add(so.Text({"fontweight": "bold"}), data=centroid_data, x='x', y='y', text='topic')

    # Add cluster Label text objects

    # Plot bidirectional intercentroid edges
    intercluster_edge_array = cntm.intercluster_edge_array
    for i in range(intercluster_edge_array.shape[0]):
        for j in range(intercluster_edge_array.shape[0]):
            if i == j:
                continue
            src_node = centroids[i]
            dst_node = centroids[j] 
            weight = intercluster_edge_array[i][j]
            direction = 1 if i < j else -1
            x, y = get_curve_trajectories(src=src_node, dst=dst_node, direction=direction)
            color = 'grey' #'red' if j > 1 else 'blue'
            edge_data = pd.DataFrame(data={'x':x, 'y':y})
            plot = plot.add(so.Path(alpha=0.01 * (weight), linewidth=weight, color=color), data=edge_data, x='x', y='y')
    
    
    # Plot cluster node vectors
    cluster_nodes = pd.DataFrame(index=cntm.nodes.keys(), data=cntm.nodes.values(), columns=['x', 'y'])
    cluster_nodes['label'] = [cntm.node_metadata[node]['cluster_label'] for node in cntm.nodes] 
    cluster_nodes['color'] = [color_palette[i] for i in cluster_nodes['label']]
    cluster_nodes['weights'] = [cntm.node_metadata[node]['weight'] for node in cntm.nodes]
    cluster_nodes['x'] = cluster_nodes['x'].astype(float) # convert the dtype from ojbect to float it to float for faster computation
    cluster_nodes['y'] = cluster_nodes['y'].astype(float)
    plot = plot.add(so.Dots(alpha=0.9, pointsize=8, fillalpha=0.8), data=cluster_nodes, x='x', y='y', color='color', legend=False)
   
    idx = 0

    """
    # Plot innercluster edges
    for edge in cntm.innercluster_edges:
        idx += 1
        inner_edge = cntm.innercluster_edges[edge]
        src_node = inner_edge['src']
        dst_node = inner_edge['dst']
        x = np.linspace(float(src_node[0]), float(dst_node[0]), num=2)
        y = np.linspace(float(src_node[1]), float(dst_node[1]), num=2)
        edge_data = pd.DataFrame(data={'x':x, 'y':y})
        # color = color_palette[cntm.innercluster_edges[edge]['dst_id']]
        color = 'grey'
        plot = plot.add(so.Path(alpha=0.6, linewidth=0.7), data=edge_data, x='x', y='y') 
        if idx==100:
            break
    """
    # Plot self centroids
    if isinstance(subnetwork_id, int) and subnetwork_id in range(len(cntm.subnetworks)):
        self_centroids = centroids + (np.random.rand(*cntm.centroids.shape) * 0.1) # Initialize self centroids nearby the cluster centroids 
        self_centroid_data = pd.DataFrame(data=self_centroids, columns=['x', 'y'])
        plot = plot.add(so.Dots(pointsize=1), data=self_centroid_data, x='x', y='y')
    
        # Plot self intercentroid edges
        specific_subnetwork_df = reviews[reviews['user_id'] == subnetwork_id]
        self_intercluster_edge_array = add_intercentroid_edges(cntm, specific_subnetwork_df, subnetwork_id='user_id', node_id='node_id')
        
        for i in range(self_intercluster_edge_array.shape[0]):
            for j in range(self_intercluster_edge_array.shape[0]):
                if i == j:
                    continue
                src_node = self_centroids[i]
                dst_node = self_centroids[j] 
                weight = self_intercluster_edge_array[i][j]
                direction = 1 if i < j else -1
                x, y = get_curve_trajectories(src=src_node, dst=dst_node, direction=direction)
                color = 'red' # 'yellow' if direction == 1 else 'green'
                edge_data = pd.DataFrame(data={'x':x, 'y':y})
                plot = plot.add(so.Path(alpha=1, linewidth=weight * 2, color=color, linestyle='--'), data=edge_data, x='x', y='y')
    plot.show()