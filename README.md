# 🚀 NER Fine-tuning, Prediction & XML to CSV Conversion

## 📖 Overview
This repository contains scripts for:
- **Fine-tuning Named Entity Recognition (NER) models** 🏷️
- **Performing NER predictions on text files** 🔍
- **Converting XML raw data to CSV format** (specifically for Mizo & Khasi languages) 📂
- **Dataset preparation and training logs** 📊

## 📂 Project Structure
```
📦 YourRepoName
├── 📄 NER_Finetune_Pipeline.py             # NER fine-tuning script
├── 📄 NER_output.py                        # NER prediction script
├── 📄 Data_conversion_and_preprocess.ipynb # XML to CSV & conll conversion notebook for Mizo & Khasi data
├── 📂 logs/                                # Training logs and reports
├── 📄 requirements.txt                     # Dependencies
└── 📄 README.md                            # Project Documentation
```

## 🏷️ Fine-tuning NER Models
### 🔧 **How to Run**
1️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

2️⃣ Run the fine-tuning script:
```bash
python finetune_ner.py
```

### 📌 **Key Features**
✅ Supports multiple transformer-based models
✅ Implements class weighting for imbalanced datasets
✅ Generates classification reports automatically 📊
✅ Efficient GPU memory management 🖥️

## 🔍 NER Prediction
The `predict_ner.py` script loads a fine-tuned NER model and predicts entity labels for sentences in a given text file.

### 🔧 **How to Run**
1️⃣ Ensure that the fine-tuned model is available.

2️⃣ Run the prediction script:
```bash
python predict_ner.py
```

3️⃣ The output will be saved in a structured CoNLL format.

### 📌 **Key Features**
✅ Uses a fine-tuned transformer model for inference
✅ Splits long sentences automatically to fit model constraints
✅ Outputs results in a CoNLL-style format for easy analysis

## 📂 XML to CSV Conversion
This Jupyter Notebook extracts data from **raw XML files** and converts them into **CSV format** for Mizo and Khasi languages.

### 🔧 **How to Run**
1️⃣ Open the Jupyter Notebook:
```bash
jupyter notebook xml_to_csv.ipynb
```

2️⃣ Run the cells step by step to process the XML data.

### 📌 **Key Features**
✅ Parses XML files efficiently 📄
✅ Supports language-specific preprocessing 🌐
✅ Outputs structured CSV data 📊

## 📊 Dataset
The dataset used for training the NER model is available here: [Dataset Link](#) (Replace with actual link)

## 🛠️ Requirements
Ensure you have **Python 3.6+** and **CUDA 12.1** (if using GPU). Install dependencies:
```bash
pip install -r requirements.txt
```

## 🤝 Contributing
Feel free to submit issues or pull requests to improve this project! 🚀

## 📜 License
This project is licensed under [Your License Here].

