# IMDB Movie Reviews Sentiment Analysis

A machine learning project that performs sentiment analysis on IMDB movie reviews using natural language processing techniques and logistic regression.

## Project Structure

```
sentiment_analysis/
│
├── data/
│   ├── IMDB_Dataset.csv        
│
├── src/
│   ├── __init__.py
│   └── sentiment_analyzer.py  
│
├── train.py                    
├── requirements.txt            
└── README.md                   
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/amMistic/Sentiment-Analysis-On-Movies-Reviews.git
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Unix or MacOS
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the IMDB Dataset of 50K Movie Reviews. Place the dataset file `IMDB_Dataset.csv` in the `data/` directory.

The dataset should have the following columns:
- review: The text of the movie review
- sentiment: The sentiment label ('positive' or 'negative')

## Requirements

Create a `requirements.txt` file with the following dependencies:
```
pandas==2.0.0
numpy==1.24.3
scikit-learn==1.2.2
nltk==3.8.1
```

## Usage

1. Ensure the IMDB dataset is placed in the `data/` directory

2. Nevigate into directory where train.py file is present.
   ```bash
   cd <directory_name>
   ```
3. Run the training script:
    ```bash
    python train.py
    ```

The script will:
- Load and preprocess the dataset
- Train a sentiment analysis model
- Evaluate the model's performance
- Show a sample prediction

## Model Details

The sentiment analyzer uses:
- TF-IDF vectorization (max 5000 features)
- Logistic Regression classifier
- Text preprocessing including:
  - Lowercase conversion
  - HTML tag removal
  - Special character removal
  - Stop word removal
  - Lemmatization

## Output

The training script will display:
- Data preprocessing progress
- Training progress
- Model evaluation metrics including:
  - Classification report (precision, recall, F1-score)
  - Confusion matrix
- Sample prediction with confidence score

## Example Output
```
Loading dataset...
Load analyzer..
Preprocess the input dataframe
Preprocessing data...
Splits the dataset into train and testing
Model starts training...
Vectorizing text...
Training model...
Evaluating model...

Classification Report:
              precision    recall  f1-score   support
           0       0.89      0.88      0.89      5000
           1       0.88      0.89      0.89      5000

Sample Review: This movie was amazing! The acting was superb and the plot was fantastic.
Predicted Sentiment: positive
Confidence: 0.95
```

## Customization

You can modify the following parameters in `sentiment_analyzer.py`:
- `max_features` in TF-IDF vectorizer
- Model parameters in LogisticRegression
- Text preprocessing steps in `preprocess_text` method

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
