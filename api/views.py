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
load_dotenv()


class InteractionListView(APIView):
    def get(self, request, format=None):
        interactions = []
        sessions = Session.objects.all()

        for session in sessions:
            conversation = messages_from_dict(session.conversation_history)
            for idx, message in enumerate(conversation):
                if message.type == 'human':
                    question = message.content
                    timestamp_str = message.additional_kwargs.get('timestamp')
                    timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else session.created_at

                    # Find the corresponding AI response
                    ai_response = None
                    for next_msg in conversation[idx + 1:]:
                        if next_msg.type == 'ai':
                            ai_response = next_msg
                            break

                    answer = ai_response.content if ai_response else ''
                    interactions.append({
                        'session_id': session.session_id,
                        'question': question,
                        'answer': answer,
                        'timestamp': timestamp,
                    })

        serializer = InteractionSerializer(interactions, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
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
        
        # Return the session details and conversation history
        return Response({
            'session_id': str(session.session_id),
            'conversation_history': session.conversation_history,
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
            input_variables=["text"],
            template="Summarize the following text:\n\n{text}"
        )
        llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4o-mini")
        chain = LLMChain(llm=llm, prompt=prompt)

        # Process the text
        summary = chain.run(text=text)
        return summary
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

        # Update the conversation history in the session
        # Serialize messages to store them in the database
        session.conversation_history = messages_to_dict(memory.chat_memory.messages)
        session.save()

        return result['answer']

