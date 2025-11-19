import argparse
from got.controller import Controller
from got.llm_interface import MockLLM, LlamaLLM

def main():
    parser = argparse.ArgumentParser(description="Standup Comedy Generator using Graph of Thoughts")
    parser.add_argument("--topic", type=str, default="Artificial Intelligence", help="Topic for the comedy routine")
    parser.add_argument("--model", type=str, default="mock", choices=["mock", "llama"], help="Model to use (mock or llama)")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Path to base model")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--quantization", type=str, default=None, choices=["4bit", "8bit"], help="Quantization mode (4bit, 8bit)")
    parser.add_argument("--branches", type=int, default=3, help="Number of branches (candidates) per step")
    
    args = parser.parse_args()

    if args.model == "llama":
        try:
            llm = LlamaLLM(
                model_path=args.base_model,
                adapter_path=args.adapter_path,
                quantization=args.quantization
            )
        except Exception as e:
            print(f"Failed to load Llama model: {e}")
            return
    else:
        llm = MockLLM()

    controller = Controller(llm=llm)
    jokes = controller.run(args.topic, num_branches=args.branches)

    print("\n=== Final Generated Jokes ===")
    for i, joke in enumerate(jokes):
        print(f"Joke {i+1}:")
        print(f"Setup: {joke['setup']}")
        print(f"Punchline: {joke['punchline']}")
        print(f"Score: {joke['score']}")
        print("=============================")

if __name__ == "__main__":
    main()
