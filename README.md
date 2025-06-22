# Amharic E-commerce Data Extractor

A Named Entity Recognition (NER) system for extracting product information from Amharic Telegram e-commerce channels.

## Project Overview

This project aims to develop a fine-tuned NER model for Amharic language that can extract key business entities such as product names, prices, and locations from Telegram e-commerce channels. The extracted data will be used to populate EthioMart's centralized database, making it a comprehensive e-commerce hub.

## Key Objectives

1. Develop a repeatable workflow for data ingestion from Telegram channels
2. Fine-tune transformer-based models for Amharic NER
3. Compare multiple model approaches and select the best performing one
4. Apply model interpretability techniques using SHAP/LIME
5. Create a vendor scoring system for micro-lending decisions

## Project Structure

```
amharic-ecommerce-extractor/
├── data/                    # Data directory
│   ├── raw/                 # Raw scraped data from Telegram
│   ├── processed/           # Preprocessed data
│   ├── labeled/             # CoNLL formatted labeled data
│   └── models/              # Saved model checkpoints
├── notebooks/               # Jupyter notebooks for each step
│   ├── 01_data_collection.ipynb    # Telegram data scraping
│   ├── 02_data_preprocessing.ipynb # Data cleaning and normalization
│   ├── 03_data_labeling.ipynb      # Entity labeling and validation
│   ├── 04_model_finetuning.ipynb   # Model training and comparison
│   ├── 05_model_comparison.ipynb   # Detailed model comparison
│   ├── 06_model_interpretability.ipynb # SHAP/LIME analysis
│   └── 07_vendor_scoring.ipynb     # Vendor analytics for lending
├── src/                     # Source code
│   ├── data/                # Data collection and processing
│   ├── models/              # Model training and evaluation
│   └── scoring/             # Vendor scoring system
├── reports/                 # Project reports
│   ├── interim/             # Interim submission
│   └── final/               # Final submission
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Current Progress

We have set up the project structure and implemented the following components:

1. **Data Collection**: Created a Telegram scraper to collect messages from e-commerce channels
2. **Data Preprocessing**: Implemented text normalization, tokenization, and entity extraction
3. **Data Labeling**: Set up a workflow for labeling entities in CoNLL format
4. **Model Fine-tuning**: Prepared the framework for training and comparing NER models

## Next Steps

1. Complete the implementation of the model interpreter module
2. Develop the vendor scoring system
3. Finalize the model comparison and interpretability notebooks
4. Prepare the interim and final reports

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/amharic-ecommerce-extractor.git
cd amharic-ecommerce-extractor

# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Follow the numbered notebooks in sequence:

1. Data Collection: Scrape data from Telegram channels
2. Data Preprocessing: Clean and structure the raw data
3. Data Labeling: Label entities in CoNLL format
4. Model Fine-tuning: Train NER models
5. Model Comparison: Compare different models
6. Model Interpretability: Explain model predictions
7. Vendor Scoring: Create lending scores for vendors

## Entity Types

- **Product**: Product names or types
- **Price**: Monetary values or prices
- **Location**: Geographic locations
- **Delivery Fee** (optional): Transaction costs
- **Contact Info** (optional): Phone numbers, Telegram usernames

## Models

- XLM-Roberta
- bert-tiny-amharic
- afroxmlr
- mBERT (Multilingual BERT)
- DistilBERT

## License

This project is licensed under the MIT License - see the LICENSE file for details. 