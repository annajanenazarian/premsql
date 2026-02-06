from pathlib import Path

import gradio as gr

from premsql.datasets import Text2SQLDataset
from premsql.datasets.base import Text2SQLBaseInstance
from premsql.executors import SQLiteExecutor
from premsql.generators import Text2SQLGeneratorHF
from premsql.prompts import BASE_TEXT2SQL_PROMPT


DATASET_FOLDER = Path("./data")
DB_ROOT = DATASET_FOLDER / "bird" / "train" / "train_databases"


def ensure_birdbench_downloaded():
    # Triggers download if needed.
    Text2SQLDataset(
        dataset_name="bird",
        split="train",
        dataset_folder=str(DATASET_FOLDER),
        force_download=False,
    )


def list_db_ids():
    if not DB_ROOT.exists():
        return []
    db_ids = []
    for db_dir in sorted(DB_ROOT.iterdir()):
        db_path = db_dir / f"{db_dir.name}.sqlite"
        if db_path.exists():
            db_ids.append(db_dir.name)
    return db_ids


def build_prompt(question: str, db_id: str, db_path: Path) -> str:
    blob = {
        "question": question,
        "SQL": "",
        "db_path": str(db_path),
        "db_id": db_id,
    }
    instance = Text2SQLBaseInstance(dataset=[blob])
    return instance.apply_prompt(
        num_fewshot=None, prompt_template=BASE_TEXT2SQL_PROMPT
    )[0]["prompt"]


def format_rows(rows, limit: int = 20) -> str:
    if rows is None:
        return ""
    lines = [str(r) for r in rows[:limit]]
    if len(rows) > limit:
        lines.append(f"... ({len(rows) - limit} more rows)")
    return "\n".join(lines)


def main():
    ensure_birdbench_downloaded()
    db_ids = list_db_ids()
    if not db_ids:
        raise RuntimeError("No BirdBench sqlite databases found.")

    generator = Text2SQLGeneratorHF(
        model_or_name_or_path="premai-io/prem-1B-SQL",
        experiment_name="birdbench_gradio_demo",
        device="cpu",
        type="demo",
    )
    executor = SQLiteExecutor()

    def run(question, db_id):
        if not question:
            return "", "", "Enter a question."

        db_path = DB_ROOT / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            return "", "", f"Missing DB file: {db_path}"

        prompt = build_prompt(question, db_id, db_path)
        data_blob = {
            "prompt": prompt,
            "question": question,
            "SQL": "",
            "db_path": str(db_path),
            "db_id": db_id,
        }
        sql = generator.generate(
            data_blob=data_blob, temperature=0.0, max_new_tokens=128
        )
        result = executor.execute_sql(sql=sql, dsn_or_db_path=str(db_path))
        return sql, format_rows(result["result"]), result["error"] or ""

    with gr.Blocks(title="BirdBench Text-to-SQL Demo") as demo:
        gr.Markdown("# BirdBench Text-to-SQL Demo")
        with gr.Row():
            question = gr.Textbox(
                label="Question",
                placeholder="Ask a question about the selected database...",
            )
            db_id = gr.Dropdown(
                label="Database",
                choices=db_ids,
                value=db_ids[0],
            )
        run_btn = gr.Button("Generate SQL")
        sql_out = gr.Textbox(label="Generated SQL")
        result_out = gr.Textbox(label="Query Result (first 20 rows)")
        error_out = gr.Textbox(label="Error (if any)")

        run_btn.click(run, inputs=[question, db_id], outputs=[sql_out, result_out, error_out])

    demo.launch()


if __name__ == "__main__":
    main()
