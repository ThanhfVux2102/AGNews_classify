
# AG News - Fine-tuning DistilBERT for News Classification

This project focuses on fine-tuning the **DistilBERT** model on the **AG News** dataset for **text classification** into 4 categories: **World**, **Sports**, **Business**, and **Sci/Tech**. The aim is to preprocess the dataset, fine-tune the model, evaluate its performance, and export the model for future use or deployment.

## Project Overview

- **Model**: Fine-tuning **DistilBERT** (a distilled version of BERT) using the **Hugging Face** library.
- **Dataset**: **AG News** dataset with 4 categories: **World**, **Sports**, **Business**, and **Sci/Tech**.
- **Evaluation**: Performance is evaluated using **accuracy** and **F1-score** metrics.
- **Deployment**: The fine-tuned model is exported using **TensorFlow** for further use in **real-time predictions** via a **REST API**.

## Technologies Used

- **Hugging Face Transformers**: Used for pre-trained models and tokenization.
- **TensorFlow**: Used for training and fine-tuning the model.
- **scikit-learn**: Used for model evaluation.
- **Python**: Programming language for building the pipeline.
- **Pandas**: Used for data manipulation.
- **Jupyter Notebook**: For development and experimentation.

## Project Structure

```plaintext
AGNews_FineTuning/
├── app/                           # Application logic for serving the model
│   ├── inference.py               # Script for inference logic and API endpoint
│   ├── main.py                    # Main application script
│   └── models.py                  # Model loading and configuration
├── ml/                            # Machine learning scripts and utilities
│   ├── train_distilbert.py         # Script to train and fine-tune DistilBERT
│   ├── train_textcnn.py           # Script to train TextCNN model
│   └── utils.py                   # Utility functions for preprocessing and model operations
├── notebooks/                     # Jupyter notebooks for experimentation and analysis
│   ├── 01_eda_agnews.ipynb        # EDA on AG News dataset
│   └── agnews_ver1.ipynb          # First version of the notebook for AG News model training
├── requirements.txt               # List of dependencies required for the project
├── .gitignore                     # Git ignore file
└── README.md                      # This file


## Setup Instructions

Follow these steps to set up the environment and run the project:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Kaso45/AGNews-FineTuning
   cd AGNews-FineTuning
   ```

2. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For macOS/Linux
   venv\Scriptsctivate     # For Windows
   ```

3. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Fine-tune DistilBERT on the AG News dataset**:

   Run the training script to fine-tune **DistilBERT** on the AG News dataset:

   ```bash
   python src/train_distilbert.py      --output_dir ./checkpoints/distilbert-agnews      --num_epochs 3      --batch_size 16      --eval_batch_size 64      --max_length 128
   ```

   This will train the model and save the fine-tuned model to the specified directory (**`checkpoints/distilbert-agnews/`**).

5. **Evaluate the fine-tuned model**:

   Once training is complete, the model will be evaluated on the test set:

   ```bash
   python src/train_distilbert.py --output_dir ./checkpoints/distilbert-agnews
   ```

   This will output the model's performance metrics (accuracy and F1-score) on the test set.

6. **Make predictions using the fine-tuned model**:

   You can test the fine-tuned model using the **predict.py** script. This script loads the saved model and uses it for predictions:

   ```bash
   python src/predict.py
   ```

   Example output:
   ```bash
   Text: "Stocks rallied today as tech companies reported strong earnings."
   Predicted label: Business (confidence=0.89)
   ```

## Data Preprocessing

1. **Tokenization**: The text data is tokenized using **DistilBERT tokenizer** from **Hugging Face**.
2. **Padding and Truncation**: Sequences are padded and truncated to ensure they all have the same length.
3. **TensorFlow Dataset**: After preprocessing, the data is converted to a **TensorFlow Dataset** for model training.

### Preprocessing Steps:
- **Tokenization**: Convert text into tokens using the **DistilBERT tokenizer**.
- **Padding and Truncation**: Ensure input sequences are of uniform length.
- **Conversion to TensorFlow Dataset**: Use **TensorFlow**'s `from_tensor_slices` to convert the data into datasets for training.

## Model Training and Evaluation

1. **Training**: The model is fine-tuned on the AG News dataset using the **Hugging Face Trainer**.
2. **Evaluation**: The model is evaluated using **accuracy** and **F1-score**.

## Performance Metrics

- **Accuracy**: Proportion of correct predictions made by the model.
- **F1-score**: Harmonic mean of precision and recall, useful for imbalanced classes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, please contact **Vũ Minh Thành** at **thanhfvux2102@gmail.com**.
