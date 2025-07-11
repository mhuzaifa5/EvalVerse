
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(r'api.env')

api_key = os.getenv('GEMINI_API_KEY')  # Load Gemini API key from environment variable
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-pro')

def llm_summary(prompt: str) -> str:
    try:
        response = model.generate_content(
            f"Generate this concise response of just features included in {prompt} and must provide a model based on included feature in prompt that model have greatest value of that feature please in one sentence only no tables please. "
        )
        return response.text
    except Exception as e:
        return f"An error occurred: {e}\nPlease ensure your API key is correctly set and you have internet access."

def get_dashboard_summary(selected_models,model_data) -> str:
    try:
        response = model.generate_content([
            f"Analyze this data which AI model is performing best overall based on the metrics of {selected_models} only from {model_data},it should not include models that are not selected ."
        ])
        return response.text
    except Exception as e:
        return f"An error occurred: {e}\nPlease ensure your API key is correctly set and you have internet access."

def chat_with_llm(message, selected_models, model_data):
    try:
        prompt = (
        "You are an expert AI model analyst. "
        "The user has selected the following models: "
        f"{', '.join(selected_models)}. "
        "You must answer using only the metadata of these selected models. "
        "Do NOT reference, compare, or even mention any unselected models — your entire response must be strictly limited to the provided data of the selected models. "
        "Avoid generic statements, comparisons, or disclaimers about other models. "
        "Your response must be highly concise (maximum 5 lines), value-dense, and based solely on the metadata below.\n\n"
        f"Selected Model Metadata{selected_models}:\n{model_data} only \n\n"
        f"User's Question:\n{message}\n\n"
        f"Answer concisely and strictly from {selected_models} data only: give main answer in top line and then supporting details in the next 2-3 lines if needed .\n "
         )
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}" 

