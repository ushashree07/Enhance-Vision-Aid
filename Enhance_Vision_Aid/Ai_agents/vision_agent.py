"""
Vision Agent Module
===================
Object detection agent using Landing AI's agentic model.
Detects roads, obstacles, vehicles, people, and traffic signs in images.
"""

import requests

def detect_objects(image_path, api_key, prompt_text=None):
    """
    Detect objects in image using Landing AI vision model.
    
    Args:
        image_path: Path to image file
        api_key: Landing AI API key
        prompt_text: Detection prompt (default: road/obstacle/vehicle/people/signs)
    
    Returns:
        Detection results as JSON
    """
    if not prompt_text:
        prompt_text = "Detect objects such as road, obstacles, vehicles, people, and traffic signs."
    
    url = "https://api.landing.ai/v1/tools/agentic-object-detection"
    
    with open(image_path, "rb") as image_file:
        files = {"image": image_file}
        data = {
            "prompts": prompt_text,
            "model": "agentic"
        }
        headers = {
            "Authorization": f"Basic {api_key}"
        }
        response = requests.post(url, files=files, data=data, headers=headers)
    
    return response.json()


# Example usage
if __name__ == "__main__":
    image_path = "testing_images/im1.jpg"  
    api_key = "bTBwdDMwZ3FuaW5maTZ3Y2E2bmRwOmZOR2k2Wmc2NDd2YWhadzA2VjNyNVh3Z2FHZUR6bUpT"
    
    result = detect_objects(image_path, api_key)
    print(result)
