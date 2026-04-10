import numpy as np
import cv2 as cv
import pandas as pd
from itertools import chain
from math import hypot
from urllib.parse import urlparse
import tarfile
import os
import boto3
import glob

def find_zarr_store(channel_dir):
    matches = glob.glob(os.path.join(channel_dir, "*.zarr"))
    if not matches:
        raise FileNotFoundError(f"No .zarr store found in {channel_dir}")
    return matches[0]

def download_and_extract_state_dict(s3_uri: str, extract_dir: str = "./extracted_model") -> str | None:
    """
    Downloads model.tar.gz from S3 and extracts it.
    Returns path to model_best.pth, or None if not found.
    """
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key    = parsed.path.lstrip("/")

    os.makedirs(extract_dir, exist_ok=True)
    local_tar = os.path.join(extract_dir, "model.tar.gz")

    print(f"Downloading state dict from s3://{bucket}/{key} ...")
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_tar)

    print(f"Extracting to {extract_dir} ...")
    with tarfile.open(local_tar, "r:gz") as t:
        t.extractall(path=extract_dir)

    pth_path = os.path.join(extract_dir, "model_best.pth")
    if os.path.exists(pth_path):
        print(f"Found model_best.pth at {pth_path}")
        return pth_path
    else:
        print(f"Warning: model_best.pth not found. Extracted files: {os.listdir(extract_dir)}")
        return None 

def get_vocab():
    room_label = [
        (0, 'LivingRoom', 1, "PublicArea"),
        (1, 'MasterRoom', 0, "Bedroom"),
        (2, 'Kitchen', 1, "FunctionArea"),
        (3, 'Bathroom', 0, "FunctionArea"),
        (4, 'DiningRoom', 1, "FunctionArea"),
        (5, 'ChildRoom', 0, "Bedroom"),
        (6, 'StudyRoom', 0, "Bedroom"),
        (7, 'SecondRoom', 0, "Bedroom"),
        (8, 'GuestRoom', 0, "Bedroom"),
        (9, 'Balcony', 1, "PublicArea"),
        (10, 'Entrance', 1, "PublicArea"),
        (11, 'Storage', 0, "PublicArea"),
        (12, 'Wall-in', 0, "PublicArea"),
        (13, 'External', 0, "External"),
        (14, 'ExteriorWall', 0, "ExteriorWall")
    ]
    
    vocab = {
        'object_name_to_idx':{},
        'object_to_idx':{},
        'object_idx_to_name':[],
    }
    
    vocab['object_name_to_idx'] = { label:index for index,label,_,_ in room_label[:] }
    vocab['object_to_idx'] = {str(index):index for index,label,_,_ in room_label}
    vocab['object_idx_to_name'] = [label for index,label,_,_ in room_label]

    return vocab

def stack_normalize(boundary_mask, room_mask, door_mask):
    stacked_layers = np.sum([boundary_mask, room_mask*3, door_mask*4], axis=0)
    stacked_layers[boundary_mask.nonzero()] = boundary_mask[boundary_mask.nonzero()]
    return stacked_layers

def stack_binarize(stacked_layers):
    stacked_layers_bin = stacked_layers.copy()
    stacked_layers_bin[stacked_layers_bin.nonzero()] = 1
    return stacked_layers_bin

def negative_space(stacked_layers_bin, inside_mask):
    negative_space = np.ones(stacked_layers_bin.shape)
    negative_space[stacked_layers_bin.nonzero()] = 0
    negative_space[inside_mask==0] = 0
    return negative_space

def rooms_with_bounds(stacked_layers_norm, labels):
    labels_norm = labels.copy()
    labels_norm += 4
    labels_norm[labels_norm==4] = 0
    return np.sum([labels_norm, stacked_layers_norm], axis=0)

def conn_components(inside_mask, boundary_mask, room_mask, door_mask):
    stacked_layers_norm = stack_normalize(boundary_mask, room_mask, door_mask)
    stacked_layers_bin = stack_binarize(stacked_layers_norm.copy())
    negative_rooms = negative_space(stacked_layers_bin, inside_mask)
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(negative_rooms.astype(np.uint8), 
                                                                       connectivity=4, 
                                                                       ltype=cv.CV_32S)
    return num_labels, labels, stats, centroids
    
def cond_arr(mask_row):
	cond_arr = [int(mask_row[0])]

	for e in mask_row:
		if e != cond_arr[-1]:
			cond_arr.append(int(e))
	
	return np.array(cond_arr)

def extract_loop(cond_arr, locs, adj_type:int):
        output = []
        for loc in locs:
            try:
                down_layer=cond_arr[loc-1]
            except Exception:
                down_layer=0
            try:
                up_layer=cond_arr[loc+1]
            except Exception:
                up_layer=0
            if (up_layer > 4) & (down_layer > 4) & (up_layer != down_layer):
                edge = np.array([up_layer, down_layer])
                output.append({"n1": int(edge.min()),
                               "n2": int(edge.max()), 
								"adj_type": adj_type})
        if len(output) > 0:
        	return output

def extract_adjacencies(cond_arr):
    idx = np.arange(len(cond_arr))
    door_locs = idx[cond_arr==4]
    wall_locs = idx[cond_arr==3]
    wall_adj = extract_loop(cond_arr, wall_locs, 0)
    door_adj = extract_loop(cond_arr, door_locs, 1)
    
    if wall_adj and door_adj:
        return wall_adj.extend(door_adj)
    
    if wall_adj:
        return wall_adj
    
    if door_adj:
        return door_adj


def dedupe_edges(edges_list):
	edges_list = [e for e in edges_list if e is not None]
	edge_df = pd.DataFrame(chain(*edges_list)).dropna()
	edge_df = edge_df.groupby(["n1", "n2", "adj_type"]).agg({"adj_type": "count"})
	edge_df.columns = ["edge_strength"]
	edge_df = edge_df[edge_df["edge_strength"] > 1].reset_index()
 
	edge_dups = edge_df.groupby(["n1", "n2"]).agg({"edge_strength": "count"}).reset_index()
	edge_dups = edge_dups[edge_dups["edge_strength"] > 1]
	for n1, n2 in zip(edge_dups["n1"], edge_dups["n2"]):
		edge_df = edge_df[~((edge_df["n1"]==n1)&(edge_df["n2"]==n2)&(edge_df["adj_type"]==0))]
 
	return edge_df
	

def extract_all_adjacencies(img):
    vertical_pass = []
    horizontal_pass = []

    for i in range(img.shape[0]):
        arr = img[i, :]
        cond = cond_arr(arr)
        vertical_pass.append(extract_adjacencies(cond))
    
    for j in range(img.shape[1]):
        arr = img[:, j]
        cond = cond_arr(arr)
        horizontal_pass.append(extract_adjacencies(cond))
                
    return dedupe_edges(vertical_pass + horizontal_pass)

def join_meta_category(centroids, metadata_row):
    categories = []
    
    for centroid in centroids:
        dists = []    
        for meta_centroid in metadata_row:
            if meta_centroid:
                xdiff = centroid[0] - meta_centroid["centroid"][1]
                ydiff = centroid[1] - meta_centroid["centroid"][0]
                dists.append(hypot(xdiff, ydiff))
        
        categories.append(int(metadata_row[np.argmin(dists)]["category"]))
    return categories

def node_array(centroids, stats):
    output = []
    for c, s in zip(centroids, stats):
        output.append([c[0], c[1], s[4]] )
    
    return output   

def edge_arrays(adj_graph):
    src, dst, attrs = [], [], []
    for e in adj_graph.iterrows():
        row = e[1]
        src.append(row["n1"])
        dst.append(row["n2"])
        attrs.append([row["adj_type"], float(row["edge_strength"])])
    return src, dst, attrs