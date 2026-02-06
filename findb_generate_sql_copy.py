import json
from pathlib import Path

import gradio as gr

from premsql.generators import Text2SQLGeneratorHF
from premsql.prompts import BASE_TEXT2SQL_PROMPT


DATA_PATH = Path(
    r"C:\Users\user\Documents\Reporting Project\premsql\data\findb_testfile\expandedtrain.json"
)

SCHEMA_BLOCK = """TRN_HDR_DBF(
  M_NB, M_OPT_MOPCNT, M_BTRADER, M_STRADER, M_COMMENT_BS,
  M__DT_TS, M_TRN_DATE, M_OPT_MOPLST, M_TRN_STATUS
)

TRN_EXT_DBF(
  M_TRADE_REF, M_VERSION, M_ACTOR, M__DT_EXTS, M__DT_TS, M_ACTION
)

MX_USER_DBF(
  M_REFERENCE, M_LABEL, M_DESC
)

Joins:
TRN_EXT_DBF.M_TRADE_REF = TRN_HDR_DBF.M_NB
TRN_EXT_DBF.M_ACTOR = MX_USER_DBF.M_REFERENCE
"""


def build_prompt(question: str, knowledge: str | None) -> str:
    return BASE_TEXT2SQL_PROMPT.format(
        schemas=SCHEMA_BLOCK,
        additional_knowledge=(
            "# Additional Knowledge:\n"
            "If missing info, return INSUFFICIENT_CONTEXT.\n"
            + (knowledge if knowledge else "")
        ),
        few_shot_examples="",
        question=question,
    )


def main():
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    question_map = {row["question"]: row for row in data}

    generator = Text2SQLGeneratorHF(
        model_or_name_or_path="premai-io/prem-1B-SQL",
        experiment_name="findb_generate_sql_copy_gradio",
        device="cpu",
        type="demo",
    )

    def run(question):
        if not question:
            return "", ""
        row = question_map.get(question, {})
        prompt = build_prompt(question, row.get("knowledge"))
        sql = generator.generate(
            data_blob={"prompt": prompt},
            temperature=0.0,
            max_new_tokens=256,
        )
        return sql, row.get("knowledge", "")

    with gr.Blocks(title="Reporting Tool Text-to-SQL Demo") as demo:
        gr.Markdown("# Reporting Tool Text-to-SQL Demo")
        question = gr.Textbox(
            label="Ask a question",
            placeholder="Type your question...",
        )
        run_btn = gr.Button("Generate SQL")
        gen_out = gr.Textbox(label="Generated SQL")
       # know_out = gr.Textbox(label="Evidence / Knowledge (if matched)")

        run_btn.click(run, inputs=[question], outputs=[gen_out])#, know_out])

    demo.launch()


if __name__ == "__main__":
    main()
