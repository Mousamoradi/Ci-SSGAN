# Ci-SSGAN
This repository focuses on detecting glaucoma subtypes from unstructured clinical notes using a clinically-informed semi-Supervised Generative Adversarial Network (Ci-SSGAN) and a Large Language Model. 

The model identifies 6 glaucoma subtypes from clinical notes:

| Class | Label | Description | 
|-------|-------|-------------|
| 0 | Non-GL | Non-glaucoma | 
| 1 | OAG/S | Open-angle glaucoma/suspect | 
| 2 | ACG/S | Angle-closure glaucoma/suspect |
| 3 | XFG/S | Exfoliation glaucoma/syndrome |
| 4 | PDG/S | Pigmentary dispersion glaucoma/syndrome | 
| 5 | SGL | Secondary glaucoma | 

Two pre-trained Ci-SSGAN models are available to accommodate different clinical settings and data characteristics:

| Model | Training Data | Best For | Validated performance (Internal dataset) |
|-------|--------------|----------|-------------|
| `ci_ssgan_25p` | 25% labeled | Different clinical domains, better generalization | AUROC: 0.97, F1: 0.88 |
| `ci_ssgan_100p` | 100% labeled | Maximum performance, similar domains | AUROC: 0.99, F1: 0.91 |

### Model selection guidance:
- **Not sure?** Start with `100p` model
- **Different hospital/EHR?** Try `25p` model  
- **Small inference dataset?** Either model works (both handle any size)

  
## Workflow
1. **Data Loading**: Load CSV/DataFrame with clinical notes (minimum: MRN, input_txt)
2. **Model Download**: Pre-trained Ci-SSGAN weights (text encoder, generator, discriminator) automatically downloaded from HuggingFace Hub
3. **Tokenization**: Bio-ClinicalBERT tokenizer processes text (truncation, padding to 512 tokens)
4. **Encoding**: Text encoder generates 768-dimensional embeddings from clinical notes
5. **Prediction**: Discriminator classifies embeddings into 6 glaucoma subtypes using softmax
6. **Output**: Returns DataFrame with predicted subtype and confidence scores for each patient

### Input Format
Your data should be a CSV/DataFrame with the following columns:

| Column | Required | Type | Description | Example |
|--------|----------|------|-------------|---------|
| MRN | ✅ Yes | String | Patient identifier | "PAT001" |
| input_txt | ✅ Yes | String | Clinical note text | "Patient presents with..." |
| note_id | Optional | String | Note identifier | "NOTE123" |
| race | Optional | Int | Race category (0-2) | 1 |
| gender | Optional | Int | Gender (0=F, 1=M) | 0 |
| age | Optional | Float | Age in years | 65.5 |
**Note:** Demographics (age, race, gender) are optional but may improve prediction accuracy if provided.

# How to execute the model:
Users should ensure they have all the necessary libraries and dependencies installed, as specified in the "Requirements". Once intalled, the code can be executed in two easy steps:

1) Model execution utilizes six input columns: ['MRN', 'note_id', 'input_txt', 'race', 'gender', 'age']. The minimum requirement consists of two mandatory columns: "MRN" and "input_txt". Column renaming is necessary if your dataset employs alternative naming conventions.

2) Download or copy the code from the "Python Code" into your environment. When executed, the code will automatically download the trained model weights and generate predictions for the notes.

### ** Installation:**
```markdown
### Requirements
- Python 3.8+
- PyTorch 2.5.1+cu124
- CUDA 11.7+ (for GPU support, optional)

### Install via pip
```bash
git clone https://github.com/yourusername/Ci-SSGAN.git
cd Ci-SSGAN
pip install -r requirements.txt

## Quick Start
import pandas as pd
from ci_ssgan_inference import run_inference

# Load your data
test_data = pd.read_csv('your_clinical_notes.csv')
# Minimum required columns: 'MRN', 'input_txt'

# Run inference with 100p model (best performance)
results = run_inference(test_data, model_variant='100p')

# Or use 25p model for different clinical domains
results = run_inference(test_data, model_variant='25p')

# Save results to CSV
results = run_inference(test_data, model_variant='100p', save_to_csv='predictions.csv')

# View results
print(results[['MRN', 'predicted_subtype', 'Probability', 'prob_{class}']])
..............................................................................................................................................................
** License
This project is licensed under the MIT License - see LICENSE file for details.

** Contact
Issues: Please open a GitHub issue for bugs or questions
Email: mmoradi2@meei.harvard.edu
