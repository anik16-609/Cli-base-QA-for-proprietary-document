import torch
import PyPDF2
from transformers import AutoModelForCausalLM, AutoTokenizer

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

# Function to answer questions
def answer_question(model, tokenizer, context, question):
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    # Set the pad_token to the eos_token to avoid errors
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = inputs.to("cpu")  # Use "cuda" if GPU is available
    outputs = model.generate(**inputs, max_new_tokens=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Main function
def main():
    pdf_path = input("Enter the path to the PDF file: ")
    context = extract_text_from_pdf(pdf_path)
    
    if context:
        print("\nPDF content extracted successfully!\n")
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.float32)

        print("Model loaded! You can now ask questions.")
        while True:
            question = input("\nAsk a question (or type 'exit' to quit): ")
            if question.lower() == "exit":
                break
            answer = answer_question(model, tokenizer, context, question)
            print(f"\nAnswer: {answer}")
    else:
        print("Failed to extract text from the PDF.")

# Run the program
if __name__ == "__main__":
    main()
