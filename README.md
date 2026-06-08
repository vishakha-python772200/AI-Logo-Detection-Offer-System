# AI Logo Detection & Offer Recommendation System

## Project Overview

This project is an AI-powered Logo Detection System built using YOLO Classification and Streamlit.

The system detects brand logos from uploaded images and displays brand-specific promotional offers automatically.

Supported brands include:

* KFC
* McDonald's
* Volkswagen
* Bata
* BMW
* Domino's
* Fastrack
* Haldiram
* Starbucks
* Zudio

---

## Features

* Logo Classification using YOLO
* Automatic Offer Recommendation
* Streamlit Web Application
* Multi-brand Support
* Real-time Prediction
* Brand-wise Accuracy Report
* Professional User Interface

---

## Tech Stack

* Python
* YOLOv8 Classification
* OpenCV
* Streamlit
* NumPy
* Pillow

---

## Dataset Structure

logos/

├── kfc/

├── mcd/

├── vw/

├── bata/

├── bmw/

├── dominos/

├── fastrack/

├── haldiram/

├── starbuks/

└── zudio/

Each folder contains logo images for a specific brand.

---

## Model Training

YOLO Classification model was trained on custom logo datasets.

Training Parameters:

* Epochs: 100
* Image Size: 224
* Batch Size: 16
* Learning Rate: 0.001
* Patience: 20

---

## Project Workflow

1. Collect logo images
2. Create train/validation dataset
3. Train YOLO classification model
4. Save best model weights
5. Deploy using Streamlit
6. Detect logo and show offer

---

## Sample Offers

| Brand      | Offer                       |
| ---------- | --------------------------- |
| KFC        | 50% OFF ON ALL COMBOS TODAY |
| McDonald's | BUY 1 BURGER GET 1 FREE     |
| Volkswagen | 1 DAY FREE TEST DRIVE       |
| Bata       | 50% OFF ON ALL FOOTWEAR     |
| BMW        | NEW MODEL LAUNCH - BOOK NOW |
| Domino's   | BUY 1 PIZZA GET 1 FREE      |
| Fastrack   | 20% OFF ON ALL WATCHES      |
| Haldiram   | BUY 1 GET 1 FREE ON SWEETS  |
| Starbucks  | BUY 2 COFFEE GET 1 FREE     |
| Zudio      | 10% OFF ON ALL CLOTHES      |

---

## Installation

```bash
git clone https://github.com/vishakha-python772200/AI-Logo-Detection-Offer-System.git

cd AI-Logo-Detection-Offer-System

pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

---

## Project Accuracy

Overall Model Accuracy: 97%

Brand-wise evaluation performed on all available logo images.

---

## Author

Vishakha Badgujar

Machine Learning & AI Developer

GitHub:
https://github.com/vishakha-python772200
