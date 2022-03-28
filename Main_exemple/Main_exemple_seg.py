#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:47:33 2022

@author: quentin
"""

from all_pipeline_two_step_post_processing import all_pipeline_with_neighbordhood_computation
from utils.IO_nifti import write_image



seg_path = "image_test/Label_patch_128_step_20.nii.gz"
neighbordhood_size = 10
threshold = 0.5
labels = [7]


ref_itk , post_processed_seg = all_pipeline_with_neighbordhood_computation(path_seg = seg_path,
                                            neighbordhood_size = neighbordhood_size,
                                            threshold = threshold,
                                            labels = labels)

write_image(ref_itk,post_processed_seg,"image_test/Label_with_two_step_post_processing.nii.gz")

