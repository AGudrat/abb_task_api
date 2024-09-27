Backend API for AI-Driven Conversational Application

This project is an AI-powered backend built using Django REST Framework and LangChain. It provides endpoints for interacting with an AI model, session management, file uploads, and web scraping.

## Features: 

    1. AI Conversational API: Communicate with an AI model using LangChain's conversational features. 
    2. Session Management: Store and retrieve conversation history between users and the AI. 
    3. File Upload: Upload files and process them to generate responses via LangChain. 
    4. Web Scraping: Scrape text from web pages and save it to a text file. 
    5. Feedback on AI Responses: Like or dislike AI responses in a session. 
    6. Swagger API Documentation: Interactive API documentation for testing endpoints.
## Tech Stack: 

    1. Django: The web framework for building the backend. 
    2. Django REST Framework: For creating API views and handling HTTP requests. 
    3. LangChain: To manage AI conversations and retrieval of context-based answers. 
    4. FAISS: For fast vector storage and search. 
    5. OpenAI: To generate embeddings and AI responses. 
    6. BeautifulSoup: For web scraping. 
    7. Swagger: API documentation with drf-yasg. 
    8. Docker: Containerization for easy deployment.

## Installation: 

#### Install dependencies:
```bash
pip install -r requirements.txt
```
#### Set up environment variables: 
```bash
Create a .env file in the project root and add your OpenAI API key and other environment variables: 

OPENAI_API_KEY=your-openai-api-key
```
#### Run database migrations:
```bash
python manage.py migrate
```
#### Start the server:
```bash    
python manage.py runserver
```

#### Access the Swagger documentation: Visit http://localhost:8000/swagger/ to see the API documentation.

## Running the Project with Docker:

#### Build the Docker image:
```bash     
docker-compose build
```
#### Run the Docker containers:
```bash  
docker-compose up
```