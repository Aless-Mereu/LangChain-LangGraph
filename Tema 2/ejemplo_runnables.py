from langchain_core.runnables import RunnableLambda,RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
import json

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


#Creamos una función que limpuie y prepare el texto
def preprocess_text(text):
    #Convertir el texto eliminando espacios extra y limitando la longitud a 500 caracteres
    text = text.strip()
    if len(text) > 500:
        text = text[:500]
    return text

#Convertir la función en un Runnable
preprocessor = RunnableLambda(preprocess_text)


#Generador de resúmenes
def generate_summary(text):
    #Generar un resumen del texto usando el modelo de lenguaje
    prompt = f"Por favor, genera un resumen conciso del siguiente texto:\n\n{text}"
    response = llm.invoke(prompt)
    return response.content

#Convertir la función en un Runnable
summary_branch = RunnableLambda(generate_summary)


#Analizador de sentimiento
def analyze_sentiment(text):
    #Analiza el sentimiento del texto y devuelve el resultado estructurado en un JSON
    prompt = f"""Analiza el sentimiento del siguiente texto y devuelve el resultado UNICAMENTE en formato JSON.
    El JSON debe contener las claves "sentimiento" y "razon".
    Los valores para "sentimiento" solo pueden ser: "positivo", "negativo" o "neutral".
    
    Ejemplo de salida:
    {{
        "sentimiento": "positivo",
        "razon": "El cliente está muy satisfecho con la calidad y rapidez del servicio."
    }}
    
    Texto a analizar: "{text}" """
    
    response = llm.invoke(prompt)
    try:
        content = response.content.strip()
        # The model might wrap the JSON in a markdown code block.
        json_str = content.lstrip("```json").rstrip("```").strip()
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"error": "No se pudo analizar el sentimiento correctamente."}
    
#Convertir la función en un Runnable    
Sentiment_branch = RunnableLambda(analyze_sentiment)


#Unificar los resultados de resumen y análisis de sentimiento en un solo diccionario
def merge_results(data):
    """Combina los resultados de ambas ramas en un formato unificado"""
    sentimiento_data = data.get("sentimiento_data", {})
    if "error" in sentimiento_data:
        return {
            "resumen": data.get("resumen"),
            "sentimiento": "error",
            "razon": sentimiento_data["error"],
        }
    return {
        "resumen": data.get("resumen"),
        "sentimiento": sentimiento_data.get("sentimiento", "desconocido"),
        "razon": sentimiento_data.get("razon", "No se proporcionó razón."),
    }
    
#Convertir la función en un Runnable
merger = RunnableLambda(merge_results)

parallel_analysis = RunnableParallel({
    "resumen": summary_branch,
    "sentimiento_data": Sentiment_branch
})

#cadena completa
chain = preprocessor | parallel_analysis | merger

reviews = [
    "Me encanta este producto! Funciona perfectamente y llegó muy rápido.",
    "El producto es de mala calidad y no funciona como se espera.",
    "El producto es aceptable, pero podría mejorar en algunos aspectos.",
    ]

for review in reviews:
    result = chain.invoke(review)
    print(f"Review: {review}")
    print(f"Resultado: {result}")
    print("-" * 50)