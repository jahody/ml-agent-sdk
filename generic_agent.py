"""
Generic ML Agent using Claude Agent SDK
Supports multiple datasets: Titanic and MNIST
"""

from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeAgentOptions, ClaudeSDKClient
import os
from typing import Literal
from research import find_papers_for_task
import json


def create_titanic_agent():
    """Create agent for Titanic dataset"""
    from tools.titanic_tools import TitanicMLTools

    ml = TitanicMLTools()

    # Define Titanic tools
    @tool("load_data", "Load train and test CSV files", {"train_path": str, "test_path": str})
    async def load_data(args):
        result = ml.load_data(args['train_path'], args['test_path'])
        return {"content": [{"type": "text", "text": str(result)}]}

    @tool("explore", "Explore dataset statistics", {"dataset": str})
    async def explore(args):
        result = ml.explore_data(args['dataset'])
        return {"content": [{"type": "text", "text": str(result)}]}

    @tool("preprocess", "Preprocess and engineer features", {"operations": list})
    async def preprocess(args):
        result = ml.preprocess_data(args['operations'])
        return {"content": [{"type": "text", "text": str(result)}]}

    @tool("train", "Train ML model", {"model_type": str, "n_estimators": int, "max_depth": int})
    async def train(args):
        params = {k: v for k, v in args.items() if k in ['n_estimators', 'max_depth'] and v}
        result = ml.train_model(args['model_type'], params or None)
        return {"content": [{"type": "text", "text": str(result)}]}

    @tool("predict", "Generate predictions", {"output_path": str})
    async def predict(args):
        result = ml.predict(args['output_path'])
        return {"content": [{"type": "text", "text": str(result)}]}

    @tool("research_papers", "Search academic papers for ML architectures and best practices", {"task_description": str, "max_papers": int})
    async def research_papers(args):
        try:
            papers = find_papers_for_task(
                args['task_description'],
                args.get('max_papers', 5)
            )
            result = json.dumps([{
                "title": p.get('title'),
                "year": p.get('year'),
                "citations": p.get('citationCount'),
                "relevance_score": p.get('relevance_score'),
                "reasoning": p.get('reasoning'),
                "paper_id": p.get('paperId')
            } for p in papers], indent=2)
            return {"content": [{"type": "text", "text": result}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Research error: {str(e)}"}]}

    server = create_sdk_mcp_server(
        name="titanic",
        version="1.0.0",
        tools=[load_data, explore, preprocess, train, predict, research_papers]
    )

    return server, ["mcp__titanic__load_data", "mcp__titanic__explore",
                    "mcp__titanic__preprocess", "mcp__titanic__train", "mcp__titanic__predict",
                    "mcp__titanic__research_papers"]


def create_mnist_agent():
    """Create agent for MNIST dataset"""
    from tools.mnist_tools import MNISTMLTools

    ml = MNISTMLTools()

    # Define MNIST tools
    @tool("load_mnist", "Load MNIST IDX files", {
        "train_images_path": str,
        "train_labels_path": str,
        "test_images_path": str,
        "test_labels_path": str
    })
    async def load_mnist(args):
        result = ml.load_data(
            args['train_images_path'],
            args['train_labels_path'],
            args['test_images_path'],
            args['test_labels_path']
        )
        return {"content": [{"type": "text", "text": str(result)}]}

    @tool("explore_mnist", "Explore MNIST dataset", {"dataset": str, "num_samples": int})
    async def explore_mnist(args):
        result = ml.explore_data(args['dataset'], args.get('num_samples', 5))
        return {"content": [{"type": "text", "text": str(result)}]}

    @tool("preprocess_mnist", "Preprocess MNIST images", {"operations": list})
    async def preprocess_mnist(args):
        result = ml.preprocess_data(args['operations'])
        return {"content": [{"type": "text", "text": str(result)}]}

    @tool("train_mnist", "Train MNIST model", {"model_type": str, "hyperparameters": dict})
    async def train_mnist(args):
        # Handle hyperparameters - might come as string or dict
        hyperparams = args.get('hyperparameters')
        if isinstance(hyperparams, str):
            import json
            try:
                hyperparams = json.loads(hyperparams)
            except:
                hyperparams = None
        result = ml.train_model(args['model_type'], hyperparams)
        return {"content": [{"type": "text", "text": str(result)}]}

    @tool("predict_mnist", "Generate MNIST predictions", {"output_path": str})
    async def predict_mnist(args):
        result = ml.predict(args.get('output_path'))
        return {"content": [{"type": "text", "text": str(result)}]}

    @tool("research_papers", "Search academic papers for ML architectures and best practices", {"task_description": str, "max_papers": int})
    async def research_papers(args):
        try:
            papers = find_papers_for_task(
                args['task_description'],
                args.get('max_papers', 5)
            )
            result = json.dumps([{
                "title": p.get('title'),
                "year": p.get('year'),
                "citations": p.get('citationCount'),
                "relevance_score": p.get('relevance_score'),
                "reasoning": p.get('reasoning'),
                "paper_id": p.get('paperId')
            } for p in papers], indent=2)
            return {"content": [{"type": "text", "text": result}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Research error: {str(e)}"}]}

    server = create_sdk_mcp_server(
        name="mnist",
        version="1.0.0",
        tools=[load_mnist, explore_mnist, preprocess_mnist, train_mnist, predict_mnist, research_papers]
    )

    return server, ["mcp__mnist__load_mnist", "mcp__mnist__explore_mnist",
                    "mcp__mnist__preprocess_mnist", "mcp__mnist__train_mnist", "mcp__mnist__predict_mnist",
                    "mcp__mnist__research_papers"]


async def run_titanic_agent(train_path: str, test_path: str, output_path: str):
    """Run Titanic ML pipeline"""
    server, allowed_tools = create_titanic_agent()
    
    options = ClaudeAgentOptions(
        mcp_servers={"titanic": server},
        allowed_tools=allowed_tools,
        system_prompt=f"""Build Titanic ML pipeline:
1. Load data from {train_path} and {test_path}
2. Explore train data (check missing values)
3. Preprocess with: fill_age, fill_embarked, fill_fare, family_size, is_alone, encode_sex, encode_embarked, extract_title
4. Use research_papers tool to search academic papers for ML architectures and best practices for Titanic survival prediction (tabular classification with mixed features). Ask for papers on "machine learning for tabular classification survival prediction" or similar.
5. Based on research findings, train 3 different models with appropriate hyperparameters and compare their performance
6. Use best model to predict to {output_path}"""
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("Execute the ML pipeline.")
        async for msg in client.receive_response():
            if hasattr(msg, 'content'):
                for block in msg.content:
                    if hasattr(block, 'text'):
                        print(block.text)


async def run_mnist_agent(data_dir: str, output_path: str):
    """Run MNIST ML pipeline"""
    server, allowed_tools = create_mnist_agent()
    
    train_images = os.path.join(data_dir, "train-images.idx3-ubyte")
    train_labels = os.path.join(data_dir, "train-labels.idx1-ubyte")
    test_images = os.path.join(data_dir, "t10k-images.idx3-ubyte")
    test_labels = os.path.join(data_dir, "t10k-labels.idx1-ubyte")
    
    options = ClaudeAgentOptions(
        mcp_servers={"mnist": server},
        allowed_tools=allowed_tools,
        system_prompt=f"""Build MNIST digit classification pipeline:
1. Load MNIST data from {data_dir}
2. Explore train dataset (show statistics and sample images)
3. Preprocess with: normalize, flatten
4. Use research_papers tool to search academic papers for ML architectures and best practices for MNIST digit classification (image classification task). Ask for papers on "deep learning for handwritten digit recognition MNIST" or similar.
5. Based on research findings, train 3 different models with appropriate hyperparameters and compare their performance
6. Use best model to predict and save to {output_path}"""
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("Execute the MNIST classification pipeline.")
        async for msg in client.receive_response():
            if hasattr(msg, 'content'):
                for block in msg.content:
                    if hasattr(block, 'text'):
                        print(block.text)


async def main(dataset: Literal["titanic", "mnist"]):
    """Main entry point - select dataset to run"""
    
    if dataset == "titanic":
        print("Running Titanic ML Agent...")
        await run_titanic_agent(
            train_path="data/titanic/train.csv",
            test_path="data/titanic/test.csv",
            output_path="predictions_titanic.csv"
        )
    elif dataset == "mnist":
        print("Running MNIST ML Agent...")
        await run_mnist_agent(
            data_dir="data/mnist",
            output_path="predictions_mnist.txt"
        )
    else:
        print(f"Unknown dataset: {dataset}")
        print("Available datasets: titanic, mnist")


if __name__ == "__main__":
    import anyio
    import sys
    
    # Get dataset from command line argument or prompt user
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        # Prompt user for dataset selection (synchronous, before async context)
        print("\n" + "="*50)
        print("Generic ML Agent - Dataset Selection")
        print("="*50)
        print("\nAvailable datasets:")
        print("  1. Titanic - Passenger survival prediction")
        print("  2. MNIST - Handwritten digit classification")
        print("\nWhich dataset would you like to work on?")
        
        while True:
            choice = input("Enter your choice (1 for Titanic, 2 for MNIST): ").strip()
            if choice == "1" or choice.lower() == "titanic":
                dataset = "titanic"
                break
            elif choice == "2" or choice.lower() == "mnist":
                dataset = "mnist"
                break
            else:
                print("Invalid choice. Please enter 1 or 2 (or 'titanic' or 'mnist').")
        print()
    
    anyio.run(main, dataset)
