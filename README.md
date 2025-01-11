# Categorizing Trends in Science - Project README

## Author

**Guglielmo Luraschi Sicca**
Matriculation Number: 92125339
IU International University of Applied Sciences
Exam: Unsupervised Learning

## Project Overview

This project analyzes the arXiv dataset to identify and categorize trends in scientific research, focusing specifically on publications from the year 2024. The analysis involves natural language processing (NLP) techniques, including TF-IDF vectorization and K-means clustering, to discover trending topics. The results are visualized using a word cloud and a bar chart to highlight the most prominent terms and their significance.

## Prerequisites

Before running the project, ensure you have the following prerequisites installed:

-   **Python 3.x**
-   **Required Python Packages:**
    -   `pandas`
    -   `nltk`
    -   `scikit-learn`
    -   `matplotlib`
    -   `seaborn`
    -   `wordcloud`
    -   `json`

You can install these packages using pip:

```bash
pip install pandas nltk scikit-learn matplotlib seaborn wordcloud


-   **NLTK Resources:**
    -   The script automatically downloads necessary NLTK resources (`stopwords`, `wordnet`, `omw-1.4`). Ensure you have an internet connection when running the script for the first time.

## Dataset

The project uses the arXiv dataset, a repository of electronic preprints approved for publication after moderation.

**Dataset Download:**

1.  The script will first prompt you whether you have already downloaded the arXiv dataset.
2.  If you answer "n" (no), you will be directed to download the dataset manually from the following link: [arXiv Dataset on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv).
3.  Download the dataset and extract it. The expected data file is named `arxiv-metadata-oai-snapshot.json`.
4.  Place the extracted `arxiv-metadata-oai-snapshot.json` file in the same directory as the Python script.

## Usage

1.  **Clone the Repository (if applicable):**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Run the Script:**

    Execute the Python script from the command line:

    ```bash
    python your_script_name.py
    ```

    (Replace `your_script_name.py` with the actual name of the Python file).

3.  **Follow the Prompts:**

    -   The script will prompt you to confirm if you have downloaded the arXiv dataset. Enter "y" if you have, and "n" if you haven't.
    -   If you answer "n", follow the instructions provided to download and place the dataset correctly.

4.  **Wait for Execution:**

    -   The script will process the dataset, filter entries for the year 2024, perform NLP analysis, and generate visualizations.
    -   Depending on the size of the dataset and your system's resources, this process might take from a few minutes to an hour.

5.  **View Results:**

    -   Once the processing is complete, a word cloud will be displayed, showing the trending words in science based on the analysis.
    -   After you close the word cloud window, a bar chart will be displayed, showing the top trending words and their corresponding TF-IDF scores.
    -   Close the bar chart window to terminate the program.

## Script Details

### Preliminary Checks

-   Checks if the arXiv dataset has been downloaded and is located in the same directory as the script.
-   Verifies the presence of the `arxiv-metadata-oai-snapshot.json` file.

### Data Preprocessing

-   **`create_filtered_json(input_file, output_file, target_year=2024)`:**
    -   Filters the input JSON file to create a new JSON file (`filtered_arxiv_data_2024.json`) containing only entries from the year 2024.
    -   Handles date parsing and JSON decoding errors.
-   **`preprocess_text(text)`:**
    -   Preprocesses text data by converting to lowercase, removing special characters, lemmatizing, and removing stop words.
    -   Stop words are extended with common words in scientific papers and lemmatized for better matching.

### Main Analysis

-   **`analyze_trends_for_year(input_file, n_clusters=8, top_n_terms=50)`:**
    -   Reads the filtered JSON data into a Pandas DataFrame.
    -   Applies TF-IDF vectorization to the preprocessed text data.
    -   Performs K-means clustering to group similar documents.
    -   Identifies the top `top_n_terms` terms for each cluster based on their average TF-IDF scores.
    -   Returns a dictionary containing the top terms for each cluster.

### Visualization

-   **`plot_wordcloud_and_barchart(words_with_scores, top_n)`:**
    -   Generates and displays a word cloud from the trending words.
    -   Creates a bar chart showing the top `top_n` trending words and their TF-IDF scores.

## Output

-   **`filtered_arxiv_data_2024.json`:** A new JSON file containing only the arXiv entries from the year 2024.
-   **Word Cloud:** A visualization showing the most frequent and significant terms in the 2024 arXiv dataset.
-   **Bar Chart:** A bar chart displaying the top trending words and their TF-IDF scores, indicating their importance.

## Notes

-   The script is designed to be run in a single execution flow. Ensure all prerequisites are met and instructions are followed carefully.
-   The processing time can vary based on system resources and dataset size.
-   The choice of `n_clusters` and `top_n_terms` in `analyze_trends_for_year` can be adjusted to explore different clustering granularities and numbers of top terms.

## Acknowledgements

-   This project was developed by Guglielmo Luraschi Sicca for the "Unsupervised Learning" course at IU International University of Applied Sciences.
-   The project utilizes the arXiv dataset, a valuable resource for scientific research analysis.
-   Special thanks to the developers and maintainers of the Python libraries used in this project.
