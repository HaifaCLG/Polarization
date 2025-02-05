# Unveiling Polarization Trends in Parliamentary Proceedings

This repository contains the code for our research on polarization trends in parliamentary proceedings.

---

## Project Resources

Below are the links to the resources we created for this project:

### 1. Hebrew VAD Lexicon

The Hebrew VAD lexicons are available here:  
[Hebrew VAD Lexicon](https://huggingface.co/datasets/GiliGold/Hebrew_VAD_lexicon)

Two versions are provided:
- **Version A:** Contains the original English words along with their translations and our additional annotations in Hebrew.
- **Version B:** Contains only the Hebrew words (formatted like the original English lexicon), where the V, A, D value for each word is the average of its values across all occurrences.

---

### 2. Knesset Corpus Labeled for VAD

The Knesset Corpus labeled for VAD is available here:  
[VAD Knesset Corpus](https://huggingface.co/datasets/GiliGold/VAD_KnessetCorpus)

This dataset includes:
- **Automatic Annotations:** Sentences from all Knesset committees labeled with V, A, and D using the automatic models we developed.
- **Manual Annotations:** A file containing 120 sentences manually annotated with V, A, and D values.  

---

### 3. Knesset-Multi Model

Our knesset-multi model is available here:  
[Knesset-multi-e5-large](https://huggingface.co/GiliGold/Knesset-multi-e5-large)

---

### 4. VAD Regression Models

Our three VAD regression models are available here:  
[VAD Binomial Regression Models](https://huggingface.co/datasets/GiliGold/VAD_binomial_regression_models)

---

## Code Overview

All the code for data processing, model training, and analysis is included in this repository.

