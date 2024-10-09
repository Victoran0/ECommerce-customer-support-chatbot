import gradio as gr
from model import llm


def generate_response(message: str, history) -> str:
    return llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a top-rated customer service agent named John. Be polite to customers and answer all their questions. If the question is out of context and not related to your job as a customer service agent, let the customer know that you can not help and they should look elsewhere for answers."
            },
            {
                "role": "user",
                "content": message
            }
        ]
    )['choices'][0]['message']['content']


demo = gr.ChatInterface(
    fn=generate_response,
    examples=[
        "What Payment Modalities are accepted?",
        "Can you help me cancel an order?",
        "What is your name and how can you help me today?"
    ],
    title="Customer Support",
    description="""This is the further fine tuned version of meta-llama/Llama-3.2-1B-Instruct. 
    Fine tuned on the https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset dataset. Random seed of 65 was used to select 1k rows from the dataset, find that version at https://huggingface.co/datasets/Victorano/customer-support-1k, all on huggingface. 
    You can find the full source code at (https://github.com/Victoran0/ECommerce-customer-support-chatbot).""",
    theme="HaleyCH/HaleyCH_Theme"
)

demo.launch()
