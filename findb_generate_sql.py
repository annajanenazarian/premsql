import json
from pathlib import Path

import gradio as gr

from premsql.generators import Text2SQLGeneratorHF
from premsql.prompts import BASE_TEXT2SQL_PROMPT


DATA_PATH = Path(r"C:\Users\user\Documents\Reporting Project\premsql\data\findb_testfile\train.copy.json")

def build_prompt(question: str, knowledge: str | None) -> str:
    # No schema available; rely on the provided evidence/knowledge only.
    return BASE_TEXT2SQL_PROMPT.format(
        schemas="(schema unavailable for this demo)",
        additional_knowledge="" if not knowledge else f"# Additional Knowledge:\n{knowledge}",
        few_shot_examples="",
        question=question,
    )


def main():
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    question_map = {row["question"]: row for row in data}

    generator = Text2SQLGeneratorHF(
        model_or_name_or_path="premai-io/prem-1B-SQL",
        experiment_name="findb_generate_sql_gradio",
        device="cpu",
        type="demo",
    )

    def run(question):
        row = question_map.get(question, {})
        prompt = build_prompt(question, row.get("knowledge"))
        sql = generator.generate(
            data_blob={"prompt": prompt},
            temperature=0.0,
            max_new_tokens=256,
        )
        return sql, row.get("SQL", ""), row.get("knowledge", "")

    with gr.Blocks(title="FindB Text-to-SQL Demo") as demo:
        gr.Markdown("# FindB Text-to-SQL Demo")
        question = gr.Textbox(
            label="Ask a question",
            placeholder="Type your question...",
        )
        run_btn = gr.Button("Generate SQL")
        gen_out = gr.Textbox(label="Generated SQL")
        #gold_out = gr.Textbox(label="Gold SQL (from file)")
        #know_out = gr.Textbox(label="Evidence / Knowledge")

        run_btn.click(run, inputs=[question], outputs=[gen_out])#, gold_out, know_out])

    demo.launch()


if __name__ == "__main__":
    main()
