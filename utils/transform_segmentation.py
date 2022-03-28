#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 12:09:55 2021

@author: quentin
"""

import SimpleITK as sitk
import numpy as np
from skimage import measure
from skimage import morphology


################################# Utils function : 


def transform_binary_image_connected_regions(binary_image):
    """ This function have been created to find the connected component which include the brain in a binary segmentation.
    In fact all MRI treated have systematically centered in the images. That's why the most centered component corresponds
    to what we are looking for.  More precisely and in order to avoid that little artifact appears and to speed up the processing
    only the 20 biggest connected component will be used.
    Finally, the notion of most centered connected component is obtained by comparing the centre of gravity of each connected component
    and the center of the volume image.

    Args:
        binary_image (3darray): Binnary image to post-process

    Returns:
        [3darray]: Binnary image on which voxel which was not in the most centered connected component have been removed (put to 0 value)
    """    
    
    image_copy = binary_image.copy()
    
    label_connected_region = measure.label(image_copy, connectivity=2)
    
    unique_elements, counts_elements = np.unique(label_connected_region[label_connected_region != 0], return_counts=True)
    
    argsort = counts_elements.argsort()
    
    increasing_label_NCC = unique_elements[argsort[::-1]]
    
    center_of_image = (np.asarray(binary_image.shape)/2).reshape((3,1))
    
    increasing_label_NCC = increasing_label_NCC[:20]
    
    min_distance_from_center_list = []
    
    labels_list=[]

    for label in increasing_label_NCC : 
    
        where = np.asarray(np.where(label_connected_region==label)) # shape = (3, 285)
    
        min_distance = np.min(np.linalg.norm(where - center_of_image,axis=0))
    
        min_distance_from_center_list.append(min_distance)
        labels_list.append(label)
        
    min_label = labels_list[np.argmin(min_distance_from_center_list)]
    
    image_one_NCC = np.zeros_like(image_copy).astype(float)
    
    image_one_NCC[label_connected_region==min_label] = binary_image[label_connected_region==min_label]  
    
    return image_one_NCC



def calculate_percent_of_label_arround_point(coord,binary_seg,neighbordhood_size):
    """Compute number of ones within a specified neighbordhood of the requested voxel in a binary segmentation.

    Args:
        coord (tuple): Coordinates of voxel to process
        binary_seg (3darray): Binary segmentation
        neighbordhood_size (int): 

    Returns:
        [float]: Percent of ones in the neighbordhood
    """
    x,y,z = coord
    
    binary_seg_around_point = binary_seg[x-neighbordhood_size:(x+neighbordhood_size+1),
                                         y-neighbordhood_size:(y+neighbordhood_size+1),
                                         z-neighbordhood_size:(z+neighbordhood_size+1)]
    
    return np.mean(binary_seg_around_point)



################################# Main function : 
    

def remove_label_keep_connected_and_add_label(image,labels_to_remove_for_connected):
    """This function allow to compute the first part of the post-processing pipeline which consist of removing some label
    of a segmentation image, binarize it , keep only the most centered component and add label which have been remove at the begining.
    More explanation about this step can be found on the readme.
    
    Args:
        image (3darray): Segmentation image to post-process
        labels_to_remove_for_connected (list): label of the segmentation that will be removed add the begining and add at the end. The voxel
        that will take one of the value in this list will be the same in the original and post-processed image.

    Returns:
        [3darray]: First pass post-processed segmentation image
    """
    
    res_image = np.zeros_like(image)
    
    binary_image = np.ones_like(image)
    
    binary_image[np.isin(image,[0]+labels_to_remove_for_connected)] = 0
    
    binary_one_ncc = transform_binary_image_connected_regions(binary_image)
    
    res_image[binary_one_ncc==1] = image[binary_one_ncc==1]
    
    res_image[np.isin(image,labels_to_remove_for_connected)] = image[np.isin(image,labels_to_remove_for_connected)]
    
    return res_image



def compute_kept_labels_by_neighbordhood_and_add_others(np_seg,neighbordhood_size,threshold,labels):
    """ Reduce the number of voxel segmented as labels by keeping only the ones which have 
    proportion of other labels grater than threshold in a squared neighborhood.

    Args:
        np_seg (3darray): Numpy array of segmentation
        neighbordhood_size (int): Size of neighborhood to compute proportion of other label
        threshold (float): Minimal proportion to keep voxel segmented as labels 
        labels (list): list of label to process

    Returns:
        [3darray]: Second pass post-processed labels
    """
    
        
    #we will pad the segmentation in order to avoid border problem :
    
    np_seg = np.pad(np_seg, 
                    ((neighbordhood_size, neighbordhood_size),
                     (neighbordhood_size, neighbordhood_size),
                     (neighbordhood_size, neighbordhood_size)), 
                    'constant', constant_values=0)
    
    labels_coord = np.where(np.isin(np_seg,labels))
    
    list_of_labels_coord = [(labels_coord[0][i],labels_coord[1][i],labels_coord[2][i]) for i in range(labels_coord[0].shape[0])]
    
    binary_seg_without_labels=np.zeros_like(np_seg)
    
    binary_seg_without_labels[(np_seg!=0)&(np.isin(np_seg,labels)==False)] = 1
    
    percent_of_neither_labels_nor_background_in_seg = [calculate_percent_of_label_arround_point
                                                    (coord,binary_seg=binary_seg_without_labels,
                                                     neighbordhood_size=neighbordhood_size)
                                                    for coord in list_of_labels_coord]
    
    
    
    # plt.hist(percent_of_neither_CSF_nor_background_in_seg,bins=500)
    # plt.show()
    
    percent_of_neither_labels_nor_background_in_seg= np.array(percent_of_neither_labels_nor_background_in_seg)
    
    list_of_labels_coord = np.array(list_of_labels_coord)
    
    
    Number_mean_number_of_other_label = np.zeros_like(np_seg)
    
    
    Number_mean_number_of_other_label[labels_coord] = percent_of_neither_labels_nor_background_in_seg
        
    
    labels_coord_to_keep = list_of_labels_coord[percent_of_neither_labels_nor_background_in_seg>threshold]
    
    
    labels_coord_to_keep = (labels_coord_to_keep[:,0],
                            labels_coord_to_keep[:,1],
                            labels_coord_to_keep[:,2])
    
    
    kept_labels = np.zeros_like(np_seg)
    
    kept_labels[labels_coord_to_keep] = np_seg[labels_coord_to_keep]
    
    kept_labels[binary_seg_without_labels==1] =  np_seg[binary_seg_without_labels==1]
    
    kept_labels = kept_labels[neighbordhood_size:-neighbordhood_size,
                              neighbordhood_size:-neighbordhood_size,
                              neighbordhood_size:-neighbordhood_size]
    
    return kept_labels
    
    
    


