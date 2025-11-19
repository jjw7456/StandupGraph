import argparse
from got.controller import Controller
from got.llm_interface import MockLLM, LlamaLLM

def main():
    parser = argparse.ArgumentParser(description="Standup Comedy Generator using Graph of Thoughts")
    parser.add_argument("--topic", type=str, default="Artificial Intelligence", help="Topic for the comedy routine")
    parser.add_argument("--model", type=str, default="mock", choices=["mock", "llama"], help="Model to use (mock or llama)")
    parser.add_argument("--branches", type=int, default=3, help="Number of branches (candidates) per step")
    
    args = parser.parse_args()

    if args.model == "llama":
        try:
            llm = LlamaLLM() # Will load default 8B model
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
