# Ci-SSGAN
This repository focuses on detecting glaucoma subtypes from unstructured clinical notes using a clinically-informed semi-Supervised Generative Adversarial Network (Ci-SSGAN) and a Large Language Model. The current version of Ci-SSGAN can detect 6 glaucoma classes as below:

6 classes = {

    0: 'Non-GL', --> non-glaucoma
    1: 'OAG/S', --> open angle glaucoma or suspect
    2: 'ACG/S', --> angle closure glaucoma or suspect
    3: 'XFG/S', --> exfoliation glaucoma or syndrome
    4: 'PDG', --> pigmentary dispersion glaucoma or suspect
    5: 'SGL' --> secondary glaucoma
}

We provide two pre-trained Ci-SSGAN models:

| `ci_ssgan_25p` | 25% labeled | Different clinical domains, robustness | F1: 0.88 (note-level) |

| `ci_ssgan_100p` | 100% labeled | Maximum performance, similar domains | F1: 0.91 (note-level) |

# Workflow

1. **Preprocessing**: Clean and normalize clinical text.
2. **Tokenization**: HuggingFace tokenizer with stopword removal.
3. **Model Weights**: The pre-trained generator and discriminator weights are pulled from HuggingFace Hub via hf_hub_download
4. **Prediction**: Trained Ci-SSGAN inference with softmax class probabilities.
5. **Output**: Predicted class + probabilities for each MRN.

# How to execute the model:
Users should ensure they have all the necessary libraries and dependencies installed, as specified in the "Requirements". Once intalled, the code can be executed in two easy steps:

1) Model execution utilizes six input columns: ['MRN', 'note_id', 'note_txt', 'race', 'gender', 'age']. The minimum requirement consists of two mandatory columns: "MRN" and "note_txt". Column renaming is necessary if your dataset employs alternative naming conventions.

2) Download or copy the code from the "Python Code" into your environment. When executed, the code will automatically download the trained model weights and generate predictions for the notes.
