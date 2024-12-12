#!/usr/bin/env python

import os
import PyPDF2
from pathlib import Path


def merge_list(
    files_to_merge, 
    output_path
):
    r"""
    """
    #-------------------------
    merger = PyPDF2.PdfMerger()
    for pdf in files_to_merge:
        try:
            merger.append(open(pdf, 'rb'))
        except PyPDF2.errors.PdfReadError:
            print(f"invalid PDF file: {pdf}")
        else:
            pass

    if not os.path.exists(Path(output_path).parent):
        os.makedirs(Path(output_path).parent)
    with open(output_path, 'wb') as fout:
        merger.write(fout)  



def merge_list_and_rotate_if_needed(
    files_to_merge, 
    output_path, 
    desired_orientation=0
):
    r"""
    This should work for merging multi-page pdfs as well...
    """
    #-------------------------
    writer = PyPDF2.PdfFileWriter()
    for pdf in files_to_merge:
        try:
            reader = PyPDF2.PdfFileReader(pdf)
            n_pages = reader.numPages
            for pagenum in range(n_pages):
                page = reader.getPage(pagenum)
                mb = page.mediaBox
                orientation = page.get('/Rotate')
                if orientation != desired_orientation:
                    page.rotateClockwise(desired_orientation)
                writer.addPage(page)
        except PyPDF2.utils.PdfReadError:
            print("invalid PDF file")
        else:
            pass

    if not os.path.exists(Path(output_path).parent):
        os.makedirs(Path(output_path).parent)
    with open(output_path, 'wb') as fout:
        writer.write(fout)


def merge_all_pdfs_in_dir(
    dir_to_merge, 
    output_path, 
    desired_orientation = 0
):
    r"""
    """
    #-------------------------
    # Find all PDF files in dir_to_merge
    files_in_dir = os.listdir(dir_to_merge)
    files_in_dir = [os.path.join(dir_to_merge, x) for x in files_in_dir if os.path.splitext(x)[1].lower()=='.pdf']
    #-------------------------
    if desired_orientation==0:
        merge_list(
            files_to_merge = files_in_dir, 
            output_path    = output_path
        )
    else:
        merge_list_and_rotate_if_needed(
            files_to_merge      = files_in_dir, 
            output_path         = output_path, 
            desired_orientation = desired_orientation
        )        