# Payne Lab: Swin Transformer and Automated Data Labeling

We are currently developing AI for semantic segmentation of carbonate thin section, supporting paleobiology and mass-extinction research. We are deploying four models across labeled and unlabeled high-resolution petrographic datasets from the Cretaceous and the Permian–Triassic, with two manuscripts in preparation. My primary contributions include building the end-to-end modeling pipeline, designing automated methods to improve training labels, and co-drafting the manuscripts.

This repository contains a representative sample of my work in the Payne Paleobiology Lab, organized around two core components:

1) **Model training pipeline.**
The training pipeline for one of the four models, featuring a novel use of a Swin Transformer backbone as a framework for hierarchical feature learning.
   
2) **Automated labeling and mask refinement.**
Due to the dense, information-rich nature of the images, as well as scarcity of labels, a large part of my work focuses on automated training-data labeling. This section includes scripts for an automated labeling scheme I designed to improve the quality of our limited ground-truth annotations, along with relabeling utilities and side-by-side visualizations of the original versus refined masks.

Both components draw from the same data directory, titled **carbonate_imgs_and_masks**.

Note on reproducibility (Colab): All work in this repository was developed and run in Google Colab, so I’ve linked the Colab notebooks directly below (rather than converting everything into local scripts and risking breaking the workflow). I’ve also included the corresponding .py versions in the repo for convenience. **To whom it may concern—if you download the sample data in this repo and upload it to your own Google Drive, the notebook paths are already set up so you can run everything end-to-end in Colab.**

**Google Drive notebooks and data directory: https://drive.google.com/drive/folders/1d1LSprCOvKxIvT1lskmYvRLjiXjL2ho-?usp=sharing**

