#!/bin/bash
# run in the base directory when you need to clear a previous runs stored data

rm -r Sequence_Info/
find . -name "clean_asu.pdb" -delete
find . -name "*.log" -delete
find . -name "clean_asu_for_refine_refine_design_*" -delete
