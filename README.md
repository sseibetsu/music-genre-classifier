# Music Genre Classifier

## Overview

The Music Genre Classifier is a machine learning project designed to classify music genres based on audio files. The project utilizes different neural network architectures, including Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), and Multilayer Perceptrons (MLP), to accurately predict the genre of a given audio file. This README provides an overview of the project, instructions on how to use it, and details on the steps involved.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)

## Features
- Supports multiple neural network architectures: CNN, RNN, and MLP
- Predicts music genres based on audio features extracted from files
- Easy-to-use web interface for uploading files and displaying predictions

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/sseibetsu/music-genre-classifier.git
   cd music-genre-classifier
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.7 or above installed. Then, install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Models:**
   Download the pre-trained models (CNN, RNN, MLP) and place them in the `models` directory.

4. **Prepare Data:**
   Ensure your data is in the correct format. The project uses JSON files for training data containing MFCC features and labels.

## Usage

1. **Start the Web Application:**
   ```bash
   python interface.py
   ```
   This will start a web server. Open your browser and navigate to `localhost:8000` to access the web interface.

2. **Upload an Audio File:**
   Use the "Select Audio File" button to choose an audio file from your computer.

3. **Choose Neural Network Type:**
   After selecting the file, you will be prompted to choose the type of neural network to use for prediction: CNN, RNN, or MLP.

4. **Predict Genre:**
   Click the "Predict Genre" button to see the predicted genre and the corresponding probabilities for each genre.

## How It Works

1. **Data Loading and Preparation:**
   The audio files are processed to extract Mel-frequency cepstral coefficients (MFCCs), which are used as features for the models.

2. **Model Training:**
   Three different neural network architectures (CNN, RNN, MLP) are trained on the extracted features. Each model is saved after training for later use in predictions.

3. **Genre Prediction:**
   The chosen model is loaded and used to predict the genre of the uploaded audio file. The result includes the predicted genre and the probabilities for each genre.

## Models

### Convolutional Neural Network (CNN)
- **Pros:** Effective for spatial data; automatically extracts features.
- **Cons:** Requires substantial computational resources.

### Recurrent Neural Network (RNN)
- **Pros:** Handles temporal data well; maintains information across sequences.
- **Cons:** Can struggle with long-term dependencies due to vanishing gradients.

### Multilayer Perceptron (MLP)
- **Pros:** Simple and easy to implement.
- **Cons:** Less effective with complex data structures compared to CNN and RNN.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure your code follows the existing coding style and is well-documented.

## License

There are no LICENSE(yet), so feel free to use for your own aims.

---

Feel free to customize this README as needed for your specific project details and instructions.
