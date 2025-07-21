# Find Salem - Black Cat Identification AI

A computer vision project using FastAI to identify a specific black cat (Salem) among other black cats.

## Project Overview

This project uses transfer learning with FastAI and ResNet34 to distinguish Salem (a specific black cat) from other black cats. It achieves 86.7% validation accuracy on a perfectly balanced dataset of 456 cat photos.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
find-salem/
├── src/
│   ├── app.py                # Main Streamlit web application
│   ├── collectors/           # Data collection modules
│   ├── trainers/             # Model training scripts
│   └── utils/                # Utility scripts
├── data/
│   ├── train/
│   └── test/
├── models/
│   └── salem_fastai_model.pkl
├── demo_samples/
└── requirements.txt
```

## Usage

### Running the Web App

```bash
streamlit run src/app.py
```

### Running Scripts

To run any of the helper scripts (e.g., for training or data collection), use the `python -m` command. This ensures the imports work correctly from the root directory.

#### Example: Retraining the FastAI model

```bash
python -m src.trainers.fastai_trainer
```

#### Example: Collecting more cat photos

```bash
python -m src.collectors.fastai_collector
```

## Performance

- **FastAI model**: 86.7% validation accuracy
- Dataset: 456 photos (228 Salem, 228 other black cats)
