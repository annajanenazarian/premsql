from pathlib import Path

from premsql.datasets import Text2SQLDataset
from premsql.executors import SQLiteExecutor
from premsql.generators import Text2SQLGeneratorHF


def main():
    # Download/load BirdBench dataset (train split)
    dataset = Text2SQLDataset(
        dataset_name="bird",
        split="train",
        dataset_folder="./data",
        force_download=False,
    ).setup_dataset(filter_by=("db_id", "address"), num_rows=1)

    generator = Text2SQLGeneratorHF(
        model_or_name_or_path="premai-io/prem-1B-SQL",
        experiment_name="birdbench_local_demo",
        device="cpu",
        type="demo",
    )

    executor = SQLiteExecutor()

    # Generate SQL with execution-guided decoding
    results = generator.generate_and_save_results(
        dataset=dataset,
        temperature=0.0,
        max_new_tokens=256,
        executor=executor,
        max_retries=1,
        force=True,
    )

    # Print a small sample
    for item in results[:3]:
        print("Question:", item["question"])
        print("Generated SQL:", item.get("generated"))
        print("DB:", item.get("db_path"))
        print("-")


if __name__ == "__main__":
    main()
