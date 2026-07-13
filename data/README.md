# Dataset Information

The raw dataset files are not included in this repository to prevent large file uploads and to comply with standard GitHub practices.

## How to get the data:

This project uses the **PhysioNet MIT-BIH Arrhythmia Database**, preprocessed and provided by Shayan Fazeli on Kaggle.

1. Go to Kaggle: [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
2. Download the dataset archive.
3. Extract the contents and locate the following two files:
   - `mitbih_train.csv`
   - `mitbih_test.csv`
4. Place both CSV files directly into this `data/` directory.

The `src/data.py` scripts and the Jupyter Notebooks are configured to load the data from this location automatically.

> **Note**: Do not rename the CSV files. They must remain exactly as `mitbih_train.csv` and `mitbih_test.csv`.
