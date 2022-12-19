import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_distances
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


class Fullnetwork():
    """
    """
    def __init__(self, nodes=dict(), edges=dict(), metadata=dict()):
        self.nodes = nodes
        self.edges = edges
        self.node_metadata = dict()
        self.edge_metadata = dict()   
        self.subnetworks = [1]
    
    def add_nodes_from_text(self, 
                            data:pd.DataFrame, 
                            token_col:str, 
                            text_col:str=None, 
                            node_ids:str='', 
                            metadata_cols:list=None, 
                            vectorizer=None,
                            lda=None,
                            embedder=None, 
                            dim:int=2,
                            node_default_weight:int=1):
        if not node_ids:
            node_ids = data.index 

        if not text_col:
            text_col = token_col

        if metadata_cols:
            if not isinstance(metadata_cols, list):
                metadata_cols = list(metadata_cols)
        if not vectorizer:
            vectorizer = TfidfVectorizer(use_idf=True,
                             max_features=500
                             )
            
        if not embedder:
            embedder = TSNE(n_components=dim,
                            learning_rate='auto',
                            perplexity=200,
                            verbose=1,
                            init='random') 
        if not lda:
            n = 7
            lda = LatentDirichletAllocation(n_components=n)
                         
        result = vectorizer.fit_transform(data[token_col])
        self.dtm = pd.DataFrame.sparse.from_spmatrix(result, columns=vectorizer.get_feature_names_out(), index=data.index) # backup DTM 
        lda_result = lda.fit_transform(result) # topic 개수만큼의 column을 가진 벡터로 인코딩
        embedded = embedder.fit_transform(lda_result)
        self.embedded = embedded
        
        # cluster 개수는 LDA에서 상정한 Topic 개수와 일치
        kmeans = KMeans(n_clusters=n).fit(embedded)
        self.centroids = kmeans.cluster_centers_
        dist_to_centroid = paired_distances(embedded, np.array([self.centroids[label] for label in kmeans.labels_]))

        self.vectorizer = vectorizer
        self.lda = lda

        # Save unique node information 
        for idx, id in enumerate(data[node_ids]):
            # Save coordinate value for each node
            coord = [str(embedded[idx][i]) for i in range(dim)] 
            self.nodes[id] = coord

            # Assign node metadata 
            self.node_metadata[id] = dict()
            self.node_metadata[id]['weight'] = node_default_weight
            self.node_metadata[id]['cluster_label'] = kmeans.labels_[idx]
            self.node_metadata[id]['cluster_distance'] = dist_to_centroid[idx]

            # Assign extra metadata
            for col in metadata_cols:
                self.node_metadata[id][col] = str(data.loc[idx, col])

        # Create intercluster_edge_count array
        self.intercluster_edge_array = np.zeros((len(self.centroids), len(self.centroids)))
    

    def add_innercluster_edges(self):
    # Innercluster_edges = Cluster에 속한 Node가 해당 Cluster Centroid와 연결된 간선(무방향)
        self.innercluster_edges = dict()
        
        for idx, node, metadata in zip([i for i in range(len(self.nodes))], self.nodes, self.node_metadata.values()):
            src_node = node
            dst_node = self.centroids[metadata['cluster_label']][0], self.centroids[metadata['cluster_label']][1]
             

            if self.node_metadata[node]['cluster_distance'] > 0.5: 
                self.innercluster_edges[node] ={'src':self.nodes[src_node], 'dst':dst_node, 'src_id': src_node, 'dst_id': metadata['cluster_label']}
            
    
    def add_intercentroid_edges(self, data:pd.DataFrame, subnetwork_id:str, node_id:str, timestamp_col:str=None):
        if not timestamp_col:
            timestamp_col = 'timestamp'
        grouped_by_subnetwork = pd.Series(data.groupby(subnetwork_id)[node_id].apply(list))
        self.subnetwork_timestamps = pd.Series(data.groupby(subnetwork_id)[timestamp_col].apply(list))

        for idx, subnetwork in zip(grouped_by_subnetwork.index, grouped_by_subnetwork.values):
            self.subnetworks.append(subnetwork)
            for node in subnetwork:
                if node in self.nodes:
                    self.node_metadata[node]['weight'] += 1
            
            edges = [(subnetwork[idx], subnetwork[idx+1]) for idx in range(len(subnetwork)-1)]
            for edge in edges:
                src_node_label = self.node_metadata[edge[0]]['cluster_label']
                dst_node_label = self.node_metadata[edge[1]]['cluster_label']
                self.intercluster_edge_array[src_node_label][dst_node_label] += 1 # intercentroid travel frequency counts


    def get_topics(self, top_n:int=5):
        components = self.lda.components_
        feature_names =  self.vectorizer.get_feature_names()
        centroid_topic_pairings = {}
        for idx, topic in enumerate(components):
            result = [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-top_n - 1:-1]]
            print("Topic %d:" % (idx), result)
            centroid_topic_pairings[idx] = [idx, result]
        return centroid_topic_pairings