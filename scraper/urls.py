from django.urls import path
from .views import scrape_text

urlpatterns = [
    path('scrape-text/', scrape_text, name='scrape_text'),
]
