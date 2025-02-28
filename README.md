# RFP_BE

## Setup Instructions

Follow these steps to set up the project locally and run it using uvicorn:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/RFP_BE.git
    cd RFP_BE
    ```

2. **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the application using uvicorn:**
    ```sh
    uvicorn main:app --reload
    ```

5. **Access the application:**
    Open your browser and navigate to `http://127.0.0.1:8000`
