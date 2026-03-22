# Spam Detector

This is a machine learning-based spam detection application built with Python. It uses a trained model to classify text messages (e.g., emails or SMS) as either "Spam" or "Ham" (not spam). The application provides a user-friendly web interface using Gradio.

## Program Structure

The project consists of the following files and directories:

- `app.py`: The main application file that runs the Gradio web interface for spam classification.
- `train_spam.py`: Script to train the spam detection model without stemming.
- `train_spam_stem.py`: Script to train the spam detection model with stemming.
- `spam_bot.py`: Additional script (possibly for bot integration or testing).
- `requirements.txt`: List of Python dependencies required to run the project.
- `artifacts/`: Directory containing the trained model files (e.g., `spam_nb_tfidf.joblib`).
- `data/`: Directory containing the dataset used for training (e.g., `spam.csv`).

## Libraries Used

The project uses the following Python libraries:

- **Gradio**: For creating the web-based user interface.
- **Scikit-learn**: For machine learning algorithms and text vectorization (TF-IDF).
- **NLTK**: For natural language processing tasks, such as tokenization and stemming.
- **Joblib**: For loading and saving the trained model.
- **Pandas**: For data manipulation and reading the CSV dataset.
- **Gradio Client**: For additional Gradio functionalities.

## What the Program Does

1. **Model Training** (via `train_spam.py` or `train_spam_stem.py`):
   - Loads a dataset of labeled text messages from `data/spam.csv`.
   - Preprocesses the text using NLTK (tokenization, optional stemming).
   - Converts text to numerical features using TF-IDF vectorization.
   - Trains a Naive Bayes classifier on the processed data.
   - Saves the trained model to the `artifacts/` directory.

2. **Spam Classification** (via `app.py`):
   - Loads the pre-trained model from `artifacts/`.
   - Provides a web interface where users can input text messages.
   - Classifies the input as "Spam" or "Ham" and provides a probability score.
   - Displays the result in JSON format.

## Process to Run the Program

1. **Set up the Environment**:
   - Ensure you have Python installed (version 3.7 or higher recommended).
   - Create a virtual environment (optional but recommended):
     ```
     python -m venv .venv
     ```
   - Activate the virtual environment:
     - On Windows: `.venv\Scripts\activate`
     - On macOS/Linux: `source .venv/bin/activate`

2. **Install Dependencies**:
   - Install the required packages:
     ```
     pip install -r requirements.txt
     ```

3. **Train the Model** (if not already trained):
   - Run the training script:
     ```
     python train_spam.py
     ```
     or for stemmed version:
     ```
     python train_spam_stem.py
     ```
   - This will generate the model file in `artifacts/`.

4. **Run the Application**:
   - Start the Gradio web interface:
     ```
     python app.py
     ```
   - Open your web browser and go to `http://127.0.0.1:7860` (or the URL provided in the terminal).
   - Enter a text message in the input box and click submit to get the spam classification result.

## Notes

- The model uses Naive Bayes algorithm with TF-IDF features for text classification.
- The web app runs locally and can also provide a public link if `share=True` is set.
- Ensure the `artifacts/` directory contains a trained model file before running `app.py`.
