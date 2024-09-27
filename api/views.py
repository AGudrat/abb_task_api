from langchain import LLMChain
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from drf_yasg.utils import swagger_auto_schema
from langchain.prompts import PromptTemplate
from drf_yasg import openapi
from rest_framework.parsers import MultiPartParser, FormParser
import chardet
from langchain.schema import messages_from_dict, messages_to_dict,AIMessage, HumanMessage
from rest_framework.generics import ListAPIView
from .models import Session
from .serializers import SessionSerializer,FileUploadSerializer, QuestionSerializer,InteractionSerializer
from datetime import datetime
from collections import Counter
from django.db.models import Count
import re
load_dotenv()

class LikeDislikeView(APIView):
    @swagger_auto_schema(
        operation_description="Like or dislike an AI response in a session",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'session_id': openapi.Schema(type=openapi.TYPE_STRING, description='Session ID'),
                'message_index': openapi.Schema(type=openapi.TYPE_INTEGER, description='Index of the message in the conversation'),
                'liked': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='Set to true to like or false to dislike the message'),
            },
            required=['session_id', 'message_index', 'liked'],
        ),
        responses={
            200: openapi.Response(description="Status updated successfully"),
            400: openapi.Response(description="Invalid data provided"),
            404: openapi.Response(description="Session not found"),
        }
    )
    def post(self, request, format=None):
        session_id = request.data.get('session_id')
        message_index = request.data.get('message_index')
        liked = request.data.get('liked')

        if session_id is None or message_index is None or liked is None:
            return Response({'error': 'Invalid data provided'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            session = Session.objects.get(session_id=session_id)
        except Session.DoesNotExist:
            return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND)

        # Retrieve conversation history
        conversation = messages_from_dict(session.conversation_history)

        # Check if the index is valid
        if message_index < 0 or message_index >= len(conversation):
            return Response({'error': 'Invalid message index'}, status=status.HTTP_400_BAD_REQUEST)

        # Find the message and update liked/disliked status
        message = conversation[message_index]
        if message.type == 'ai':
            message.additional_kwargs['liked'] = liked

            # Save the updated conversation back to the session
            session.conversation_history = messages_to_dict(conversation)
            session.save()

            return Response({'message': 'Status updated successfully'}, status=status.HTTP_200_OK)

        return Response({'error': 'Can only like/dislike AI messages'}, status=status.HTTP_400_BAD_REQUEST)

from collections import defaultdict

class InteractionListView(APIView):
    def get(self, request, format=None):
        interactions = []
        sessions = Session.objects.all()
        
        # Data for tracking the additional features
        question_lengths = []
        word_counter = Counter()
        liked_disliked_interactions = {'liked': [], 'disliked': [], 'default': []}
        
        # Dictionary to track daily interactions
        interactions_by_date = defaultdict(int)

        for session in sessions:
            conversation = messages_from_dict(session.conversation_history)
            for idx, message in enumerate(conversation):
                if message.type == 'human':
                    question = message.content
                    timestamp_str = message.additional_kwargs.get('timestamp')
                    timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else session.created_at
                    
                    # Track interaction by date
                    date_key = timestamp.date()
                    interactions_by_date[date_key] += 1
                    
                    # Update question length distribution
                    question_length = len(question.split())
                    question_lengths.append(question_length)
                    
                    # Count words for frequency analysis
                    words = re.findall(r'\w+', question.lower())
                    word_counter.update(words)

                    # Find the corresponding AI response
                    ai_response = None
                    for next_msg in conversation[idx + 1:]:
                        if next_msg.type == 'ai':
                            ai_response = next_msg
                            break
                    
                    answer = ai_response.content if ai_response else '' 
                    
                    # Track liked and disliked answers for the AI response
                    if ai_response:
                        if 'liked' in ai_response.additional_kwargs:
                            if ai_response.additional_kwargs['liked']:
                                liked_disliked_interactions['liked'].append({
                                    'session_id': session.session_id,
                                    'question': question,
                                    'answer': answer,
                                    'timestamp': timestamp,
                                })
                            else:
                                liked_disliked_interactions['disliked'].append({
                                    'session_id': session.session_id,
                                    'question': question,
                                    'answer': answer,
                                    'timestamp': timestamp,
                                })
                        else:
                            liked_disliked_interactions['default'].append({
                                'session_id': session.session_id,
                                'question': question,
                                'answer': answer,
                                'timestamp': timestamp,
                            })

                    interactions.append({
                        'session_id': session.session_id,
                        'question': question,
                        'answer': answer,
                        'timestamp': timestamp,
                    })

        # Convert interactions by date to a list sorted by date
        interactions_over_time = [{'date': date, 'count': count} for date, count in sorted(interactions_by_date.items())]

        # Get the top 10 most frequent words
        most_frequent_words = word_counter.most_common(10)
        
        # Prepare the response data
        data = {
            'interactions': interactions,
            'interactions_over_time': interactions_over_time,
            'question_length_distribution': {
                'min': min(question_lengths) if question_lengths else 0,
                'max': max(question_lengths) if question_lengths else 0,
                'average': sum(question_lengths) / len(question_lengths) if question_lengths else 0
            },
            'top_10_most_frequent_words': most_frequent_words,
            'liked_disliked_interactions': liked_disliked_interactions
        }

        return Response(data, status=status.HTTP_200_OK)

# View to list all sessions
class SessionListView(ListAPIView):
    serializer_class = SessionSerializer

    def get_queryset(self):
        return Session.objects.all()
class SessionDetailView(APIView):
    @swagger_auto_schema(
        operation_description="Retrieve session details and conversation history.",
        responses={
            200: openapi.Response(
                description="Session details retrieved successfully.",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'session_id': openapi.Schema(type=openapi.TYPE_STRING),
                        'conversation_history': openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Items(type=openapi.TYPE_OBJECT)),
                    },
                ),
            ),
            404: 'Not Found',
        },
    )
    def get(self, request, session_id, format=None):
        try:
            session = Session.objects.get(session_id=session_id)
        except Session.DoesNotExist:
            return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # Deserialize the conversation history from the session
        conversation = messages_from_dict(session.conversation_history)
        
        # Create an updated conversation history response with "liked" status for AI messages
        conversation_response = []
        for idx, message in enumerate(conversation):
            # Base message data
            message_data = {
                "id": idx,
                "type": message.type,
                "content": message.content,
                "timestamp": message.additional_kwargs.get('timestamp', session.created_at.isoformat()),
            }

            # For AI messages, include the "liked" status
            if message.type == 'ai':
                message_data["liked"] = message.additional_kwargs.get('liked', None)  # True, False, or None

            # Append to the response list
            conversation_response.append(message_data)
        
        # Return the session details with the updated conversation history
        return Response({
            'session_id': str(session.session_id),
            'conversation_history': conversation_response,
        }, status=status.HTTP_200_OK)

    @swagger_auto_schema(
        operation_description="Delete a session.",
        responses={
            204: 'No Content',
            404: 'Not Found',
        },
    )
    def delete(self, request, session_id, format=None):
        try:
            session = Session.objects.get(session_id=session_id)
            session.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except Session.DoesNotExist:
            return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND)

file_param = openapi.Parameter(
    'file',
    openapi.IN_FORM,
    type=openapi.TYPE_FILE,
    required=True
)

class FileUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    
    @swagger_auto_schema(
        manual_parameters=[file_param],
        responses={
            200: openapi.Response(
                description="File uploaded successfully.",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'session_id': openapi.Schema(type=openapi.TYPE_STRING, description='Session ID for subsequent requests'),
                        'response': openapi.Schema(type=openapi.TYPE_STRING, description='Initial summary of the uploaded file'),
                    },
                ),
            ),
            400: 'Bad Request',
        },
    )
    def post(self, request, format=None):
        serializer = FileUploadSerializer(data=request.data)
        if serializer.is_valid():
            uploaded_file = serializer.validated_data['file']
            uploaded_file_content = uploaded_file.read()

            # Detect encoding using chardet
            result = chardet.detect(uploaded_file_content)
            encoding = result['encoding']

            # Try decoding with the detected encoding, fallback to common encodings
            possible_encodings = [encoding, 'utf-8', 'ISO-8859-1', 'Windows-1254']
            file_content = None
            for enc in possible_encodings:
                if enc:
                    try:
                        file_content = uploaded_file_content.decode(enc)
                        break  # Break if successful
                    except (UnicodeDecodeError, TypeError):
                        continue

            if not file_content:
                return Response(
                    {'error': f'Unable to decode file content using {possible_encodings} encodings.'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Process the file content (LangChain is optional)
            response = self.process_with_langchain(file_content)
            
            # Create a new session
            session = Session(file_content=file_content)
            session.save()

            # Return the session ID and the initial response (if any)
            return Response({
                'session_id': str(session.session_id),
                'response': response,
                'created_at': session.created_at.isoformat(),
            }, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def process_with_langchain(self, text):
        # Initialize LangChain components
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Task: You are responsible for answering the following question using only the context provided. Your reply must strictly adhere to the guidelines below:
1. Stay Within the Context:

    Ensure your answer is based exclusively on the given context.
    If the context doesn’t have enough information, mention that explicitly.
    Avoid making assumptions or using external sources to fill gaps.
    If the context is unrelated to the question, respectfully explain that.

2. Dealing with Unclear Questions:

    If the question is vague or unclear, ask for further explanation rather than guessing.
    Offer polite, informative responses when essential details are missing.

3. Addressing Errors or Inconsistencies:

    Should the context contain inconsistencies or unclear data, point them out.
    Where possible, suggest how missing or unclear information could be resolved.

4. Tone and Language:

    Keep a respectful and courteous tone in all replies.
    Start with "Salam" (in Azerbaijani) as a greeting.
    Respond entirely in Azerbaijani unless otherwise directed.

5. Structure and Clarity:

    Make your answers concise and focused. Avoid including unnecessary information.
    Use sections or bullet points where needed to improve readability.

6. Follow-up and Clarification:

    If additional details are necessary to complete the response, politely ask for them.
    Avoid assuming answers where critical information is lacking.

7. Relevance and Timeliness:

    If the provided context seems outdated or incomplete, make a note of this.
    Pay attention to time-related aspects and highlight any differences where applicable.

8. Simplifying Complex Information:

    If the context contains technical terms, break them down into simpler concepts.
    Ensure clarity without altering the original meaning.

9. Accuracy and Confidentiality:

    Always protect any sensitive or personal information found in the context.
    Refrain from sharing personal details unnecessarily.

10. Explaining Visual Data:

    If charts, tables, or other visual elements are mentioned in the context, explain their relevance clearly.

11. Identifying Relationships:

    Point out relationships between various pieces of information within the context to ensure a full understanding.

12. Highlighting Missing Information:

    If there are gaps in the data, state them clearly and explain their impact on your answer.

Context: {context}
Question: {question}

Important: Responses must be in Azerbaijani unless otherwise noted. If anything is unclear or incomplete, make sure to mention this and request clarification."""
        )
        llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4o-mini")
        chain = LLMChain(llm=llm, prompt=prompt)

        # Process the text
        response = chain.run({
            'context': text,  # Here you pass the text as context
            'question': 'Text faylı nədən ibarətdir?'  # Replace this with a valid question
        })
        return response
class TimestampedConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs, outputs):
        # Add timestamp to the human message
        human_message = HumanMessage(content=inputs['question'], additional_kwargs={'timestamp': datetime.now().isoformat()})
        self.chat_memory.add_message(human_message)

        # Add timestamp to the AI message
        ai_message = AIMessage(content=outputs['answer'], additional_kwargs={'timestamp': datetime.now().isoformat()})
        self.chat_memory.add_message(ai_message)
class QuestionAnsweringView(APIView):
    @swagger_auto_schema(
        operation_description="Ask a question based on the uploaded file content.",
        request_body=QuestionSerializer,
        responses={
            200: openapi.Response(
                description="Answer to the provided question.",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'answer': openapi.Schema(type=openapi.TYPE_STRING, description='Answer to the question'),
                    },
                ),
            ),
            400: 'Bad Request',
        },
    )
    def post(self, request, format=None):
        serializer = QuestionSerializer(data=request.data)
        if serializer.is_valid():
            session_id = serializer.validated_data['session_id']
            question = serializer.validated_data['question']

            # Retrieve the session and associated file content
            try:
                session = Session.objects.get(session_id=session_id)
            except Session.DoesNotExist:
                return Response({'error': 'Invalid session ID'}, status=status.HTTP_400_BAD_REQUEST)

            # Process the question using LangChain
            answer = self.process_question(session, question)

            return Response({'answer': answer}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def process_question(self, session, question):
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        texts = text_splitter.split_text(session.file_content)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
        vector_store = FAISS.from_texts(texts, embeddings)

        # Initialize OpenAI LLM
        llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'), temperature=0.2, model="gpt-4o-mini")

        # Retrieve conversation history from session
        memory = TimestampedConversationBufferMemory(memory_key="chat_history", return_messages=True)
        if session.conversation_history:
            # Deserialize messages from stored dicts
            memory.chat_memory.messages = messages_from_dict(session.conversation_history)
        
        # Create a conversation chain
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
        )

        # Get the answer
        result = qa({"question": question})

        session.conversation_history = messages_to_dict(memory.chat_memory.messages)
        session.save()

        return result['answer']

