from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0.7)

pregunta = "¿En que año llegó el ser humano a la luna por primera vez y quienes fueron los astronautas a bordo?"
print ("pregunta: ", pregunta)

respuesta = llm.invoke(pregunta)
print ("respuesta: ", respuesta.content) 