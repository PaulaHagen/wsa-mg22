import os
import requests
import dotenv
import logging

logging.basicConfig(level=logging.DEBUG)

api_versions = {
    "gpt-4-turbo": "turbo-2024-04-09",
    "gpt-4o": "2024-05-13",
    "gpt-4o-mini": "2024-07-18",
}

def chatgpt(user_prompt, model_name):

    # print("Chatting with GPT {}...".format(model_name))

    dotenv.load_dotenv(override=True)

    # Configuration
    API_KEY = os.getenv("WSR_OPENAI_API_KEY")
    logging.debug(f"API_KEY: {API_KEY}")
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }

    # Create system and user prompts
    system_prompt = 'you are a skillful researcher that is writing abstracts'

    # Payload for the request
    payload = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ],
        "temperature": 2.0, # This are very high temperature and top_p values to make the abstracts more diverse (otherwise they are very similar)
        "top_p": 0.95,
        "max_tokens": 800,
    }

    ENDPOINT = f"https://wsr-openai.openai.azure.com/openai/deployments/{model_name}/chat/completions?api-version=2024-08-01-preview"

    # Send request
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

    # Handle the response as needed (e.g., print or process)
    return response.json()['choices'][0]['message']['content']

if __name__ == "__main__":
    # Example usage
    text = "AmpUp is proud to announce its participation in the second Google for Startups Climate Change Accelerator program. The 10-week digital program offers intensive training, mentorship, technical assistance and other resources to climate-focused Seed to Series A technology startups located in North America. The program also allows startups to present their top technical challenges to Google and receive guidance from relevant Google experts to solve these challenges. As part of Google’s commitment to address climate change, the company has made it a priority to offer support to entrepreneurs and startups solving a broad range of sustainability-related challenges in the hope of accelerating the transition to a low-carbon and circular economy. AmpUp is honored to be part of a cohort tackling pressing climate issues ranging from creating plastic-alternative biomaterials, decarbonizing materials, and of course electrification of transportation ."
    response_content = chatgpt(text, "gpt-4o-mini")
    print(response_content)
