from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import onnxruntime as ort
import numpy as np

# Function to load the model and tokenizer
def load_model_and_tokenizer(model_name):
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load the model with DirectML execution provider for AMD GPUs
        providers = ['DmlExecutionProvider']  # Use DirectML for AMD GPUs
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None

# Tokenize the input question
def tokenize_input(question, tokenizer):
    try:
        inputs = tokenizer(question, return_tensors="pt")
        # Move tensors to GPU if available
        inputs = inputs.to("cuda") if torch.cuda.is_available() else inputs
        return inputs
    except Exception as e:
        print(f"Error tokenizing input: {e}")
        return None

# Generate the response
def generate_response(model, tokenizer, question):
    inputs = tokenize_input(question, tokenizer)
    if inputs:
        try:
            # Generate the response using DirectML
            outputs = model.generate(**inputs, max_new_tokens=50)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer
        except Exception as e:
            print(f"Error generating response: {e}")
    return None

# Main function to run the question-answering
def main():
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Example model
    question = "What is the capital of Bangladesh?"
    
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    if model and tokenizer:
        # Get the answer
        answer = generate_response(model, tokenizer, question)
        if answer:
            print("Answer:", answer)
        else:
            print("Failed to generate an answer.")
    else:
        print("Failed to load the model or tokenizer.")

# Run the main function
if __name__ == "__main__":
    main()
