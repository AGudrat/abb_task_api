import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

@api_view(['POST'])
def scrape_text(request):
    url = request.data.get('url')
    
    if not url:
        return Response({"error": "URL is required"}, status=status.HTTP_400_BAD_REQUEST)
    
    # Ensure the base URL is valid
    parsed_url = urlparse(url)
    if not bool(parsed_url.scheme) or not bool(parsed_url.netloc):
        return Response({"error": "Invalid URL provided"}, status=status.HTTP_400_BAD_REQUEST)

    # Function to clean up and filter visible text, excluding text but keeping links from headers, footers, and navs
    def clean_text(soup):
        unwanted_tags = ['script', 'style', 'noscript', 'meta', 'link']
        for tag in unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()  # Remove unwanted tags
        
        # Only remove the text from header, footer, and nav elements but keep the links intact
        ignored_sections = ['header', 'footer', 'nav', 'aside']  # Add other sections if needed
        for section in ignored_sections:
            for element in soup.find_all(section):
                # Clear text inside header, footer, or nav but leave the structure (so links are preserved)
                for child in element.find_all(text=True):
                    child.extract()

        # Extract visible text from the rest of the page
        return ' '.join(text.strip() for text in soup.stripped_strings if text)

    visited_urls = set()
    text_elements = []
    page_limit = 30  # Set the page limit to 30
    page_count = 0  # Initialize page count

    def scrape_page(url):
        nonlocal page_count

        if page_count >= page_limit:
            return  # Stop scraping once the page limit is reached

        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract and process visible text from the page
        clean_page_text = clean_text(soup)

        # Log the current URL being processed for debugging
        print(f"Scraping URL: {url}, extracted text length: {len(clean_page_text)}")

        # Increment the page count
        page_count += 1
        
        # Append non-empty text to the text_elements list
        if clean_page_text:
            text_elements.append(clean_page_text)

        # Extract and follow internal links, even from headers, footers, and nav elements
        base_url = url
        for link in soup.find_all('a', href=True):
            link_url = urljoin(base_url, link['href'])
            # Check if the link belongs to the same domain and avoid re-visiting the same links
            if link_url not in visited_urls and link_url.startswith(parsed_url.scheme + '://' + parsed_url.netloc):
                visited_urls.add(link_url)
                scrape_page(link_url)  # Recursive scraping for internal links

    # Start scraping from the provided URL
    visited_urls.add(url)
    scrape_page(url)

    # Check if any text was scraped
    if not text_elements:
        return Response({"error": "No text found in the provided URL"}, status=status.HTTP_404_NOT_FOUND)

    # Combine the cleaned text from all pages
    full_text = '\n'.join(text_elements)

    # Define the path to the .txt file
    output_dir = os.path.join(settings.BASE_DIR, 'scraped_texts')
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    filename = "scraped_content.txt"
    file_path = os.path.join(output_dir, filename)
    
    # Write the cleaned text content to the .txt file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(full_text)
    
    # Return a response with the filename
    return Response({"message": "Text successfully scraped and written to file", "file": filename}, status=status.HTTP_200_OK)
