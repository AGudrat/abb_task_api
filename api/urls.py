from django.urls import path
from .views import FileUploadView, QuestionAnsweringView,SessionListView,SessionDetailView,InteractionListView

urlpatterns = [
    path('upload/', FileUploadView.as_view(), name='file-upload'),
    path('ask/', QuestionAnsweringView.as_view(), name='question-answering'),
    path('sessions/', SessionListView.as_view(), name='session-list'),
    path('sessions/<uuid:session_id>/', SessionDetailView.as_view(), name='session-detail'),
    path('interactions/', InteractionListView.as_view(), name='interaction-list'),
]
