
import os
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))


def nutrients_metrics_func(prompt):

    model = genai.GenerativeModel("gemini-2.0-flash", 

        system_instruction = f"""
        You are an agricultural assistant that analyzes soil data. Given soil readings 
        such as Nitrogen (N), Phosphorus (P), Potassium (K), humidity, moisture, 
        temperature, and pH, return a short message summarizing the soil condition and 
        give one simple recommendation. The message should be concise, factual, and under 25 words.
        No emojis.


        Example Output:

        Soil pH is slightly acidic. Nitrogen levels are low. Apply organic compost and 
        maintain regular irrigation to improve fertility.
        
        """

            )


    response = model.generate_content(
        prompt,
        generation_config = genai.GenerationConfig(
        max_output_tokens=1000,
        temperature=1.5, 
      )
    
    )


    
    return response.text



def crop_recommendations_func(prompt):

    model = genai.GenerativeModel("gemini-2.0-flash", 

        system_instruction = f"""

        You are a crop advisory expert for small-scale farmers in Kenya. Suggest one or
         two crops suitable for current conditions or the short rains season. Keep the 
         message brief, under 20 words, and avoid emojis.

        Example Output:

        Based on current weather, consider planting kale and beans. They perform well 
        for small farms this season.
        
        """

            )


    response = model.generate_content(
        prompt,
        generation_config = genai.GenerationConfig(
        max_output_tokens=1000,
        temperature=1.5, 
      )
    
    )


    
    return response.text



def future_forecast_func(prompt):

    model = genai.GenerativeModel("gemini-2.0-flash", 

        system_instruction = f"""
        You are a climate and weather advisor for farmers. Summarize the upcoming 1–2 
        week forecast focusing on rainfall, temperature, and drought likelihood. Keep 
        it under 25 words, clear, and without emojis.

        Example Output:

        Expect moderate rainfall with warm days next week. Ideal for early planting but 
        ensure soil moisture retention.
        
        """

            )


    response = model.generate_content(
        prompt,
        generation_config = genai.GenerationConfig(
        max_output_tokens=1000,
        temperature=1.5, 
      )
    
    )


    
    return response.text


def market_projection_func(prompt):

    model = genai.GenerativeModel("gemini-2.0-flash", 

        system_instruction = f"""
        You are an agricultural market analyst. Predict short-term price trends for a 
        given crop as the December holiday approaches. Provide a short, practical 
        message under 25 words without emojis.

        Example Output:

        Maize prices expected to rise in December due to holiday demand. Consider 
        storing produce for better returns.
        
        """

            )


    response = model.generate_content(
        prompt,
        generation_config = genai.GenerationConfig(
        max_output_tokens=1000,
        temperature=1.5, 
      )
    
    )


    
    return response.text



