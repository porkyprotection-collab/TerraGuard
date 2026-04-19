import argparse
from transformers import pipeline

def main():
    parser = argparse.ArgumentParser(description='Generate code using Hugging Face model.')
    parser.add_argument('prompt', type=str, help='The prompt to generate code from.')
    args = parser.parse_args()

    # Load a code generation model
    generator = pipeline('text-generation', model='Salesforce/codegen-350M-mono')
    generated = generator(args.prompt, max_length=100, num_return_sequences=1)
    print(generated[0]['generated_text'])

if __name__ == '__main__':
    main()