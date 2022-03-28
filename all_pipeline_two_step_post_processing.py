#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:11:16 2022

@author: quentin
"""



from utils.transform_segmentation import remove_label_keep_connected_and_add_label, compute_kept_labels_by_neighbordhood_and_add_others
from utils.IO_nifti import read_image , write_image




def all_pipeline_with_neighbordhood_computation(path_seg,neighbordhood_size,threshold,labels):
    """Read segmentation, post-process segmentation (centered connected component) exluding 
    some labels and process the excluded labels in a second step (neighborhood analysis)

    Args:
        path_seg (str): path of segmentation to post process
        neighbordhood_size (int): Number of voxel before and after voxel to post-process for creating neighborhood
        threshold (float): Minimal proportions to kept voxel segmented as labels
        labels (list): list of label to exclude to post-process only during the second step

    Returns:
        [3darray]: Post-processed segmentation after the two step
    """

    image_itk , np_seg = read_image(path_seg)

    np_seg_first_pass = remove_label_keep_connected_and_add_label(image = np_seg,
                                                                labels_to_remove_for_connected = labels)


    np_seg_second_pass = compute_kept_labels_by_neighbordhood_and_add_others(np_seg = np_seg_first_pass,
                                                                            neighbordhood_size = neighbordhood_size,
                                                                            threshold = threshold,
                                                                            labels = labels)

    return image_itk,np_seg_second_pass





