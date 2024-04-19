from openai import OpenAI
import json
import ast
import time

client = OpenAI(api_key="Your OpenAI API Key Here")

with open('./llm_instruction/generate_style_instruction.txt', 'r') as file:
    generate_style_instruction = file.read()

def run_conversation():

    concept_file_path = "./llm_instruction/original_concept.txt"
    image_file_path = "./llm_instruction/original_file_name.txt"

    with open(concept_file_path, 'r') as file:
        content = file.read()
        concept_list = ast.literal_eval(content)
    
    with open(image_file_path, 'r') as file:
        content = file.read()
        image_list = ast.literal_eval(content)

    messages = []

    print("concept_list: ", concept_list)
    print("image_list: ", image_list)
    generate_description(concept_list,image_list, messages, "gpt-3.5-turbo-0125")


def generate_description(concept_list, image_list, past_messages, llm_version):

    for each_concept, each_image in zip(concept_list, image_list):
        messages = []
        new_message = [
            {"role": "system", "content": generate_style_instruction},
            {"role": "user", "content": f"Asset: {each_concept}"},
            {"role": "user", "content": "Variants: "},
        ]
        messages.extend(new_message)

        call_gpt_flag = True
        while call_gpt_flag:
            try:
                gpt_response = client.chat.completions.create(
                        model=llm_version,
                        messages=messages,
                        max_tokens=4096,
                    )
                gpt_response_message = gpt_response.choices[0].message.content
                print(f"gpt_response_message for {each_image} ({each_concept}): {gpt_response_message}")
                representations_for_concept = ast.literal_eval(gpt_response_message)
                # Verify the response is a list
                if isinstance(representations_for_concept, list):
                    print("gpt_response_message is a list. Saving to file...")
                    with open(f"./description/{each_image}.txt", 'w') as f:
                        f.write(gpt_response_message)
                    call_gpt_flag = False
                else:
                    print(f"gpt_response_message for {each_image} is not a list. Retrying the API call in 5 time units.")
                    time.sleep(5)
            except Exception as e:  # Replace Exception with the specific exception for quota exceed if available
                print(f"Encountered an error: {e}. Retrying in 5 time units.")
                time.sleep(5)  # Wait for 5 seconds before retrying

print(run_conversation())