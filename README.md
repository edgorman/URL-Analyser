# URL-Analyser

A comprehensive investigation for the development of a malicious URL detection system using Machine Learning. This contributed to my BSc Computer Science degree at University of Southampton, which received a First-Class with honours classification.

## Installation
Use the following command to clone the respository:
```
cd your/repo/directory
git clone https://github.com/edgorman/URL-Analyser
```

Install [Anaconda](https://www.anaconda.com/) and create a python environment using this command:
```
conda env create --file environment.yml
```

And then activate it using conda
```
conda activate URLAnalayser
```

## Usage
Make sure you have conda environment installed before running the URLAnalayser:
```
python -m URLAnalyser [-h] [-u URL] [-m MODEL] [-d DATA] [-f FEATS] [-save] [-refine] [-verbose] [-version]
```
Without any optional arguments, the program will run the best model and return the classification metrics. The following are some example commands and what they perform:

Check if a URL is classified as malicious or benign:
```
python -m URLAnalyser -u https://example.com
```

Run the svm model on the content feature set:
```
python -m URLAnalyser -m svm -d content -f 0
```

Run the testing scripts in the base directory:
```
python -m pytest
```

Add the flag `--disable-pytest-warnings` to remove Keras warnings.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
