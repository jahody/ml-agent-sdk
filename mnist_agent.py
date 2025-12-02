"""
MNIST ML Agent using Claude Agent SDK
Dedicated agent for MNIST handwritten digit classification
"""

from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeAgentOptions, ClaudeSDKClient
import os
from research import find_papers_for_task
import json


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


async def main():
    """Main entry point for MNIST agent"""
    print("=" * 60)
    print("MNIST ML Agent - Handwritten Digit Classification")
    print("=" * 60)
    print()

    # Default paths
    data_dir = "data/mnist"
    output_path = "predictions_mnist.txt"

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        print("Please ensure MNIST data files are in the correct location:")
        print(f"  - {data_dir}/train-images.idx3-ubyte")
        print(f"  - {data_dir}/train-labels.idx1-ubyte")
        print(f"  - {data_dir}/t10k-images.idx3-ubyte")
        print(f"  - {data_dir}/t10k-labels.idx1-ubyte")
        return

    print(f"Data directory: {data_dir}")
    print(f"Output file: {output_path}")
    print()
    print("Starting MNIST classification pipeline...")
    print()

    await run_mnist_agent(data_dir=data_dir, output_path=output_path)

    print()
    print("=" * 60)
    print("Pipeline complete!")
    print(f"Predictions saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    import anyio
    anyio.run(main)
