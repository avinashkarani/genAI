# GenAI Classification

This project uses LangChain and OpenAI to classify companies and retrieve related information.

## Setup

1. **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up API keys:**
    - Create a `config.ini` file in the root directory with the following content:
        ```ini
        [API_KEYS]
        TAVILY_API_KEY = your_tavily_api_key
        OPENAI_API_KEY = your_openai_api_key
        ```

5. **Prepare data files:**
    - Ensure `taxanomy.csv` and `data.csv` are present in the root directory.

## Running the Code

1. **Run the script:**
    ```sh
    python genAI_clasification.py
    ```

## Error Handling

- The script includes error handling for missing files and missing columns in the data.
- If any errors occur during processing, they will be printed to the console.

## Dependencies

- langchain-openai
- langchain-community
- langchain-core
- langgraph
- pandas
- configparser