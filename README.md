# Fake News Detector

A machine learning project that classifies news articles as **Real** or **Fake** using text data and natural language processing.

## Overview

Fake news can spread quickly and make it difficult to tell which information can be trusted.

This project explores how machine learning can be used to analyze news articles and identify patterns commonly found in fake and real news.

The model is trained using labelled news datasets and then used to classify new news content.

## How It Works

```text
News Article
     ↓
Text Preprocessing
     ↓
Feature Extraction
     ↓
Machine Learning Model
     ↓
Real / Fake
```

The input text is cleaned and prepared before being converted into features that can be understood by the machine learning model. The trained model then predicts whether the news is likely to be real or fake.

## Features

* Real and fake news classification
* Text preprocessing and cleaning
* Machine learning based prediction
* Separate datasets for real and fake news
* NLP-based text analysis
* Simple structure for further development

## Dataset

The project uses two datasets:

| File       | Description                 |
| ---------- | --------------------------- |
| `Fake.csv` | Contains fake news articles |
| `True.csv` | Contains real news articles |

These datasets are used for training and testing the classification model.

## Tech Stack

**Language**

* Python

**Machine Learning**

* Scikit-learn

**Data Processing**

* Pandas
* NumPy

**Natural Language Processing**

* NLP techniques for processing and preparing text data

## Project Structure

```text
fake-news-detector/
│
├── fake-news-project/
│   └── ...
│
├── Fake.csv
├── True.csv
└── README.md
```

## Getting Started

### Prerequisites

Make sure Python and pip are installed on your system.

### Clone the Repository

```bash
git clone https://github.com/harshchaudhary8649-abc/fake-news-detector.git
cd fake-news-detector
```

### Install Dependencies

Install the required Python libraries used by the project.

```bash
pip install -r requirements.txt
```

### Run the Project

Open the project folder and run the main Python file used by the application.

```bash
python app.py
```

> The exact run command may vary depending on the current project setup.

## Learning From This Project

I built this project to get hands-on experience with machine learning and text-based data.

While working on it, I focused on:

* Preparing and cleaning text data
* Working with real-world datasets
* Understanding NLP for text classification
* Training a machine learning model
* Making predictions from news content

## Future Improvements

* Improve model performance
* Add a cleaner web interface
* Try different classification algorithms
* Add a prediction confidence score
* Use a larger and more diverse dataset
* Deploy the project as a web application

## Author

**Harsh Chaudhary**

GitHub: [@harshchaudhary8649-abc](https://github.com/harshchaudhary8649-abc)

## License

This project is licensed under the MIT License.
