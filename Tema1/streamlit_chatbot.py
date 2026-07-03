from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import streamlit as st


#Configurar la pagina de la app
st.set_page_config(page_title="Chatbot con LangChain y Google Generative AI", page_icon=":beer:")
st.title("Chatbot con LangChain y Google Generative AI")
st.markdown("Bienvenido al chatbot creado con LangChain y Google Generative AI. Puedes hacer preguntas y recibir respuestas generadas por el modelo de lenguaje.")


#Creamos un sidebar para la configuracion del chatbot
with st.sidebar:
    st.header("Configuracion del Chatbot")
    temperature = st.slider("Temperatura del modelo", 0.0, 1.0, 0.5, 0.1)
    model_name = st.selectbox("Selecciona el modelo de lenguaje", ["gemini-2.5-flash", "gemini-2.5", "gemini-1.5", "gemini-1.0"])
    
    #Recrear el modelo cpn nuevos parámetros
    chat_model = ChatGoogleGenerativeAI(model = model_name, temperature = temperature)


#Inicializar el historial de mensajes en la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []


#Definir la plantilla de prompt para el modelo de lenguaje
prompt_template = PromptTemplate(
    input_variables=["mensaje","historial"],
    template = """
    Eres un asistente de un técnico de QA de una empresa farmacética. Tu tarea consiste en
    determinar cuales son las mejor medidas preventivas y correctivas para los problemas de calidad que se presentan en la empresa.
    Tu respuesta debe ser clara, concisa y profesional. Debes proporcionar recomendaciones específicas y detalladas para abordar 
    los problemas de calidad, incluyendo pasos a seguir, recursos necesarios y posibles resultados.
    
    Historial de la conversación:
    {historial}
    
    Responde de manera profesional y detallada a la siguiente pregunta:
    {mensaje}"""
)


#Crear una cadena usando LCEL (LangChain Expression Language) para manejar la conversación con el modelo de lenguaje
cadena =  prompt_template | chat_model


    
#Mostrar mensajes previos en la interfaz de usuario
for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue  # No mostrar mensajes del sistema en la interfaz de usuario
    role = "assistant" if isinstance(message, AIMessage) else "user"
    
    with st.chat_message(role):
        st.markdown(message.content)
        
if st.button("Reiniciar conversación"):
    st.session_state.messages = []
    st.rerun()
        
        
#Cuadro de entrada de texto para el usuario
pregunta = st.chat_input("Escribe tu pregunta aquí...")

if pregunta:
    #Mostrar inmediatamente la pregunta del usuario en la interfaz de chat
    with st.chat_message("user"):
        st.markdown(pregunta)

    #Generar y mostar la respuesta del modelo de lenguaje
    try:
        with st.chat_message("assistance"):
            response_placeholder = st.empty()
            full_response = ""
            
            #Streaming de las respuesta
            for chunk in cadena.stream({"mensaje": pregunta, "historial": st.session_state.messages}):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
                
            response_placeholder.markdown(full_response)
        st.session_state.messages.append(HumanMessage(content=pregunta))
        st.session_state.messages.append(AIMessage(content=full_response))
    except Exception as e:
        st.error(f"Error al generar la respuesta: {str(e)}")
        st.info("Por favor, verifica tu conexión a internet y la configuración del modelo de lenguaje.")
            
        
    
        
    