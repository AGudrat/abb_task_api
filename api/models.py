from django.db import models
import uuid

class Session(models.Model):
    session_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    file_content = models.TextField()
    conversation_history = models.JSONField(default=list, blank=True, null=True)  
    created_at = models.DateTimeField(auto_now_add=True)