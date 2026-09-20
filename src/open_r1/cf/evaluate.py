"""Run the unchanged CLEVRER evaluator with explicit local paths."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    model, dataset = Path(args.model).resolve(strict=True), Path(args.dataset).resolve(strict=True)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    from eval.eval_clevrer import ModelEvaluator
    evaluator = ModelEvaluator(model_path=str(model), dataset_path=str(dataset),
                               output_path=str(output), batch_size=args.batch_size,
                               test_samples=None, save_token_info=False)
    evaluator.evaluate()


if __name__ == '__main__':
    main()
