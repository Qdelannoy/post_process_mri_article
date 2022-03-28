#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 15:12:11 2021

@author: quentin
"""

import SimpleITK as sitk

def read_image(image_path):
    
    
    image_itk = sitk.ReadImage(image_path)
    np_image = sitk.GetArrayFromImage(image_itk)

    
    return image_itk , np_image

def write_image(ref_itk,image,output_path):
    
    Output = sitk.GetImageFromArray(image)
    
    Output.SetSpacing(ref_itk.GetSpacing())
    Output.SetOrigin(ref_itk.GetOrigin())
    Output.SetDirection(ref_itk.GetDirection())
    
    sitk.WriteImage(Output,output_path)
    