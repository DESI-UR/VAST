




#
# The code below was used in _voidfinder._hole_finder_multi_process, but wasn't found
# to provide any significant value.  The guess was that providing some kind of
# sort order to the input galaxy coordinates matrix would improve the data locality
# with respect to which galaxies are being used for the neighbor-finding process, and
# would help speed up the code.  My guess is that there are just too many other calculations
# going on per iteration for the data locality to be useful.  It did provide a minor speedup
# on DR7 runs, but I haven't fully quantified its impact.  
#
# I didn't want to throw this code away, so it remains in this file, commented out. Future
# testing on this idea may be warranted, but will require a more thorough investigation
# than what I put into it.
#
#
#




'''
#TSNE bin sort

mesh_indices = ((w_coord - coord_min)/search_grid_edge_length).astype(np.int64)

unique_bins = np.unique(mesh_indices, axis=0)

print("Starting TSNE, num_gal, unique bins: ", w_coord.shape[0], len(unique_bins))

tsne_time = time.time()

embedding = TSNE(n_components=1, verbose=1).fit_transform(unique_bins)

print(embedding.shape)

print(embedding[0:10])

print("Finished TSNE", time.time() - tsne_time)

bin_sort_order = np.argsort(embedding[:,0])


bin_sort_map = {}
for idx, row in enumerate(unique_bins):
    bin_sort_map[tuple(row)] = bin_sort_order[idx]
    
    
master_sort_order = []
for _ in range(bin_sort_order.shape[0]):
    master_sort_order.append([])


for idx in range(mesh_indices.shape[0]):
    bin_ID = tuple(mesh_indices[idx])
    bin_order = bin_sort_map[bin_ID]
    master_sort_order[bin_order].append(idx)
    
    
w_coord_sort_order = np.concatenate(master_sort_order)


sorted_w_coord = w_coord[w_coord_sort_order]

del w_coord

w_coord = sorted_w_coord

del mesh_indices



mesh_indices = ((w_coord - coord_min)/search_grid_edge_length).astype(np.int64)
'''













'''
#HDBSCAN bin sort
unique_bins = np.unique(mesh_indices, axis=0)

print("Starting HDBSCAN, num_gal, unique bins: ", w_coord.shape[0], len(unique_bins))

hdbscan_time = time.time()

clusterer = hdbscan.HDBSCAN()
clusterer.fit(unique_bins)

labels = clusterer.labels_

probs = clusterer.probabilities_

print("PROBS SHAPE: ", probs.shape)
#print(probs[0:10])

#labels = np.argmax(probs, axis=1)

#unique_labels = np.unique(labels)

#or curr_label in unique_labels:
    
    




print(labels.shape)

print(labels[0:10])

print("Finished HDBSCAN", time.time() - hdbscan_time)

bin_sort_order = np.argsort(labels)






bin_sort_map = {}
for idx, row in enumerate(unique_bins):
    bin_sort_map[tuple(row)] = bin_sort_order[idx]
    
    
master_sort_order = []
for _ in range(bin_sort_order.shape[0]):
    master_sort_order.append([])


for idx in range(mesh_indices.shape[0]):
    bin_ID = tuple(mesh_indices[idx])
    bin_order = bin_sort_map[bin_ID]
    master_sort_order[bin_order].append(idx)
    
    
w_coord_sort_order = np.concatenate(master_sort_order)


sorted_w_coord = w_coord[w_coord_sort_order]

del w_coord

w_coord = sorted_w_coord

del mesh_indices



mesh_indices = ((w_coord - coord_min)/search_grid_edge_length).astype(np.int64)
'''






























'''
unique_bins, bin_counts = np.unique(mesh_indices, return_counts=True, axis=0)

print("Max bin: ", bin_counts.max())
print("Min bin: ", bin_counts.min())
#bin_pre_sort_index = ngrid[1]*ngrid[2]*unique_bins[:,0] + \
#                        ngrid[2]*unique_bins[:,1] + \
#                        unique_bins[:,2]




#bin_sort_order = np.argsort(bin_pre_sort_index)
bin_sort_order = np.argsort(bin_counts)[::-1]

bin_sort_map = {}
for idx, row in enumerate(unique_bins):
    bin_sort_map[tuple(row)] = bin_sort_order[idx]
    
    
master_sort_order = []
for _ in range(bin_sort_order.shape[0]):
    master_sort_order.append([])


for idx in range(mesh_indices.shape[0]):
    bin_ID = tuple(mesh_indices[idx])
    bin_order = bin_sort_map[bin_ID]
    master_sort_order[bin_order].append(idx)
    
    
w_coord_sort_order = np.concatenate(master_sort_order)


sorted_w_coord = w_coord[w_coord_sort_order]

del w_coord

w_coord = sorted_w_coord

del mesh_indices



mesh_indices = ((w_coord - coord_min)/search_grid_edge_length).astype(np.int64)
'''





















'''
#bin sort?
galaxy_pre_sort_index = ngrid[1]*ngrid[2]*mesh_indices[:,0] + \
                        ngrid[2]*mesh_indices[:,1] + \
                        mesh_indices[:,2]
                        
del mesh_indices
                        
w_coord_sort_order = np.argsort(galaxy_pre_sort_index)

sorted_w_coord = w_coord[w_coord_sort_order]

del w_coord

w_coord = sorted_w_coord

mesh_indices = ((w_coord - coord_min)/search_grid_edge_length).astype(np.int64)
'''





'''
#Scramble

w_coord_scramble_order = np.random.permutation(w_coord.shape[0])

scrambled_w_coord = w_coord[w_coord_scramble_order]

del w_coord

w_coord = scrambled_w_coord

mesh_indices = ((w_coord - coord_min)/search_grid_edge_length).astype(np.int64)
'''

