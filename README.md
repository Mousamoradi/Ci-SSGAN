# SGAN-LLM
This repository focuses on detecting glaucoma subtypes from unstructured clinical notes using a Semi-Supervised Generative Adversarial Network and a Large Language Model (SGAN-LLM). The current version of SGAN_LLM can detect 7 glaucoma classes as below:

7 classes = {

    0: 'Non-GL', --> Non-glaucoma
    1: 'POAG', --> Primary open angle glaucoma
    2: 'POAS', --> Primary open angle suspect
    3: 'PACG', --> Primary angle closure glaucoma
    4: 'XFG', --> Pseudoexfoliation glaucoma
    5: 'PDG', --> Pigmentary dispersion glaucoma
    6: 'SGL' --> Secondary glaucoma
}

# Workflow

1. **Preprocessing**: Clean and normalize clinical text.
2. **Tokenization**: HuggingFace tokenizer with stopword removal.
3. **Model Weights**: The pre-trained generator and discriminator weights are pulled from HuggingFace Hub via hf_hub_download
4. **Prediction**: Trained SGAN-LLM inference with softmax class probabilities.
5. **Output**: Predicted class + probabilities for each MRN.

# How to execute the model:
Users should ensure they have all the necessary libraries and dependencies installed, as specified in the "Requirements". Once intalled, the code can be executed in two easy steps:

1) To make predictions using the model, ensure that your dataset includes at least two required columns: "MRN" and "note_txt". If your dataset uses different column names, please rename them accordingly before proceeding to the next steps.

2) Download or copy the code from the "Python Code" into your environment. When executed, the code will automatically download the trained model weights and generate predictions for the notes.
