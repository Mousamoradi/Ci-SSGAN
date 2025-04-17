# SGANLLM
This repository focuses on detecting glaucoma subtypes from unstructured clinical notes using a Semi-Supervised Generative Adversarial Network and a Large Language Model (SGAN-LLM). The current version of SGAN_LLM can detect 6 glaucoma classes as below:

'Non-GL': 'Non-glaucoma',

'POAG': 'Primary open angle glaucoma',

'PACG': 'Primary angle closure glaucoma',

'PXG': 'Pseudoexfoliation  glaucoma',

'PDG':'Pigmentary dispersion glaucoma',

'SGL': 'Secondary glaucoma'

# How to execute the model:
Users should ensure they have all the necessary libraries and dependencies installed, as specified in the "Requirements". Once intalled, the code can be executed in two easy steps:

1) To make predictions using the model, ensure that your dataset includes at least two required columns: "MRN" and "note_txt". If your dataset uses different column names, please rename them accordingly before proceeding to the next steps.

2) Download or copy the code from the "Python Code" into your environment. When executed, the code will automatically download the trained model weights and generate predictions for the notes.
