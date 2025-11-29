"""
Titanic ML Agent - Simplified version
Complete ML pipeline: data loading → preprocessing → training → prediction
"""

from ml_tools import TitanicMLTools
import json


class TitanicMLAgent:
    """Simple agent that orchestrates the ML pipeline"""

    def __init__(self):
        """Initialize the agent with ML tools"""
        self.ml_tools = TitanicMLTools()

    def run(self,
            train_path: str,
            test_path: str,
            output_path: str = "predictions.csv",
            model_type: str = "random_forest",
            n_estimators: int = 100,
            max_depth: int = 5) -> dict:
        """
        Run the complete ML pipeline

        Args:
            train_path: Path to training CSV
            test_path: Path to test CSV
            output_path: Path to save predictions
            model_type: 'random_forest' or 'logistic_regression'
            n_estimators: Number of trees for random forest
            max_depth: Max depth for random forest

        Returns:
            dict with results from each step
        """
        results = {}

        print("Starting Titanic ML Pipeline\n")

        # Step 1: Load data
        print("Step 1: Loading data...")
        load_result = self.ml_tools.load_data(train_path, test_path)
        if not load_result['success']:
            return {"success": False, "error": "Failed to load data", "details": load_result}

        print(f"   > Train: {load_result['train_shape']}")
        print(f"   > Test: {load_result['test_shape']}")
        results['load'] = load_result

        # Step 2: Explore training data
        print("\nStep 2: Exploring data...")
        explore_result = self.ml_tools.explore_data('train')
        if explore_result['success']:
            missing = explore_result['missing_counts']
            print(f"   > Missing Age: {missing.get('Age', 0)}")
            print(f"   > Missing Embarked: {missing.get('Embarked', 0)}")
            print(f"   > Missing Cabin: {missing.get('Cabin', 0)}")
        results['explore'] = explore_result

        # Step 3: Preprocess data
        print("\nStep 3: Preprocessing and feature engineering...")
        operations = [
            "fill_age",           # Fill missing ages with median
            "fill_embarked",      # Fill missing embarked with mode
            "fill_fare",          # Fill missing fares with median
            "family_size",        # Create FamilySize = SibSp + Parch + 1
            "is_alone",           # Create IsAlone feature
            "encode_sex",         # Encode Sex: male=0, female=1
            "encode_embarked",    # One-hot encode Embarked
            "extract_title"       # Extract title from Name
        ]

        preprocess_result = self.ml_tools.preprocess_data(operations)
        if not preprocess_result['success']:
            return {"success": False, "error": "Failed to preprocess", "details": preprocess_result}

        print(f"   > Features: {', '.join(preprocess_result['feature_columns'][:5])}...")
        print(f"   > Train shape: {preprocess_result['X_train_shape']}")
        results['preprocess'] = preprocess_result

        # Step 4: Train model
        print(f"\nStep 4: Training {model_type} model...")
        hyperparameters = {}
        if model_type == "random_forest":
            hyperparameters = {
                'n_estimators': n_estimators,
                'max_depth': max_depth
            }

        train_result = self.ml_tools.train_model(model_type, hyperparameters)
        if not train_result['success']:
            return {"success": False, "error": "Failed to train", "details": train_result}

        print(f"   > Train accuracy: {train_result['train_accuracy']:.4f}")
        print(f"   > CV accuracy: {train_result['cv_mean']:.4f} (+/-{train_result['cv_std']:.4f})")
        results['train'] = train_result

        # Step 5: Generate predictions
        print(f"\nStep 5: Generating predictions...")
        predict_result = self.ml_tools.predict(output_path)
        if not predict_result['success']:
            return {"success": False, "error": "Failed to predict", "details": predict_result}

        print(f"   > Predictions: {predict_result['predictions_count']}")
        print(f"   > Survival rate: {predict_result['survival_rate']:.2%}")
        print(f"   > Saved to: {output_path}")
        results['predict'] = predict_result

        print("\nPipeline completed successfully!\n")

        return {
            "success": True,
            "results": results,
            "summary": {
                "model": model_type,
                "train_accuracy": train_result['train_accuracy'],
                "cv_accuracy": train_result['cv_mean'],
                "cv_std": train_result['cv_std'],
                "predictions_count": predict_result['predictions_count'],
                "survival_rate": predict_result['survival_rate']
            }
        }

    def save_results(self, results: dict, output_path: str = "results.json"):
        """Save pipeline results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {output_path}")


def main():
    """Example usage"""

    # Initialize agent
    agent = TitanicMLAgent()

    # Run pipeline
    results = agent.run(
        train_path='g:/My Drive/AZ/llm_evo/titanic/data/train.csv',
        test_path='g:/My Drive/AZ/llm_evo/titanic/data/test.csv',
        output_path='g:/My Drive/AZ/llm_evo/titanic/predictions.csv',
        model_type='random_forest',
        n_estimators=100,
        max_depth=5
    )

    # Display summary
    if results['success']:
        print("=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        summary = results['summary']
        print(f"Model: {summary['model']}")
        print(f"Train Accuracy: {summary['train_accuracy']:.2%}")
        print(f"CV Accuracy: {summary['cv_accuracy']:.2%} (±{summary['cv_std']:.2%})")
        print(f"Predictions: {summary['predictions_count']}")
        print(f"Survival Rate: {summary['survival_rate']:.2%}")
        print("=" * 60)

        # Save full results
        agent.save_results(results)
    else:
        print(f"\nPipeline failed: {results.get('error')}")
        if 'details' in results:
            print(f"Details: {results['details']}")


if __name__ == "__main__":
    main()
