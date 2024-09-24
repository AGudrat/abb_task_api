from rest_framework import serializers
from .models import Session
from rest_framework import serializers
from .models import Session

class InteractionSerializer(serializers.Serializer):
    session_id = serializers.UUIDField()
    question = serializers.CharField()
    answer = serializers.CharField()
    timestamp = serializers.DateTimeField()
    
class FileUploadSerializer(serializers.Serializer):
    file = serializers.FileField()

    def validate_file(self, value):
        if not value.content_type.startswith('text/'):
            raise serializers.ValidationError('Only text files are allowed.')
        return value

class QuestionSerializer(serializers.Serializer):
    session_id = serializers.UUIDField()
    question = serializers.CharField()

class SessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Session
        fields = ['session_id', 'conversation_history', 'created_at']