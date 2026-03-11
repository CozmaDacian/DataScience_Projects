# Data Science & Machine Learning Projects

This repository contains two distinct data science projects: a sophisticated Graph-Based Movie Recommendation System and a Customer Churn Prediction analysis. 


### 1. IMDB Movie Analysis & Recommendation System (`ImDb_Movie_Review.ipynb`)
**Description:**
This project explores the IMDB Top 1000 movies dataset. It goes beyond basic EDA by utilizing Natural Language Processing (NLP) for data imputation, advanced feature engineering, and building a hybrid Markov Chain recommendation system.

**Key Features & Techniques:**
* **Machine Learning Imputation:** Uses a `TfidfVectorizer` and a `MultinomialNB` (Naive Bayes) classifier to accurately predict and fill in missing movie certificates based on the text of the movie's overview.
* **Feature Engineering:** * Adjusts historical box office Gross revenues for inflation.
  * Applies multi-label encoding for movie genres.
  * Calculates a custom **"Duo Synergy"** score, mathematically weighting the historical rating success of specific Actor-Director collaborations.
* **Hybrid Recommendation Engine:** Implements a Graph-Based / Markov Chain recommendation system (Random Walk). By calculating the stationary distribution (similar to PageRank) via eigenvectors, the system blends immediate movie-to-movie similarity (local context) with global network popularity to prevent the "filter bubble" in recommendations.

**Technologies Used:** `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`

---

### 2. Customer Churn Prediction (`ChurnPrediction.ipynb`)
**Description:**
This project focuses on analyzing user behavior and subscription details to lay the groundwork for predicting customer retention. It automatically fetches the latest data via `kagglehub` and processes it for machine learning classification.

**Key Features & Techniques:**
* **Data Ingestion & Cleaning:** Automatically downloads the dataset and maps categorical text data into machine-readable numeric ranks (e.g., Subscription Types to basic/standard/premium tiers, and Contract Lengths to months).
* **Exploratory Data Analysis (EDA):** Generates detailed visualizations (Histograms with KDE overlays) to uncover the distributions of key churn indicators:
  * Payment Delays
  * Usage Frequency
  * Customer Tenure
  * Total Spend
  * Days Since Last Interaction
* **Goal:** To structure and clean behavioral data so it can be effectively fed into predictive classification models to identify at-risk customers (the `Churn` target variable).

**Technologies Used:** `pandas`, `kagglehub`, `matplotlib`, `seaborn`


### 3. Visual Clothes Recommendation System (`ClothesRecommandationSystem.ipynb`)
**Description:**
This project implements a reverse-image search and visual recommendation engine for fashion and apparel. It allows users to upload an image of a clothing item and automatically retrieves the most visually similar items from a catalog.



**Key Features & Techniques:**
* **Image Feature Extraction:** Utilizes Deep Learning computer vision models to process images and extract high-dimensional mathematical embeddings that represent the style, color, and pattern of the clothing.
* **Similarity Matching:** Computes nearest neighbors using distance metrics (like Cosine Similarity) to find the closest matches between the user's uploaded image and the inventory database.
* **Interactive File Uploads:** Features custom JavaScript and Colab widget integration, allowing users to seamlessly upload their own local images directly into the notebook for real-time recommendations.

**Technologies Used:** `Python`, Deep Learning Frameworks (e.g., `TensorFlow`/`Keras` or `PyTorch`), `numpy`, `matplotlib` (for displaying images), `scikit-learn`
