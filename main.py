import asyncio
import csv
import json
import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

INPUT_CSV = "Data/data-QA-JEE-Data_preprocessed.csv"
OUTPUT_CSV = "Data/JEE_explanations.csv"
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-6-luna")
DEFAULT_N = 5  # new questions to generate per run
BATCH_SIZE = 5  # questions per abatch call
MAX_CONCURRENCY = 5  # parallel requests inside one batch

# Columns the Superteacher viewer reads, added on top of the original CSV columns
EXTRA_COLUMNS = ["correct_display_id", "correct_answer", "explanation_markdown"]


class SolutionMarkdown(BaseModel):
    markdown_explanation: str = Field(
        ...,
        description="Solution markdown in the Superteacher card format.",
    )


SYSTEM_PROMPT = """You are an expert STEM educator writing short, high-yield JEE solutions for the Superteacher mobile card.
The output is rendered as a compact card, so keep every section brief and scannable.

Output EXACTLY this layout. Each heading is a markdown heading on its own line. 
Leave one blank line after the heading, and start the content on the next line. 
Never put content on the same line as a heading.

## Concept
One or two short sentences (under 30 words) stating the core idea. For setup or derivation problems, use "## Key setup" instead of "## Concept".

## Formula
The key formula(s) only, one per line, in LaTeX using $...$ (inline) or $$...$$ (display). Omit this section only if no formula applies.

## Steps
1. One short calculation step with LaTeX
2. ...
(3 to 5 numbered steps. Each step is one line showing the working, ending with the intermediate or final result.)

## Answer
**Answer:** <final value> → (<option id>)
If the answer is not an option, write only: **Answer:** <final value>

## Note
One punchy line (under 20 words) with a common mistake or key takeaway.

Rules:
- Solve directly for the correct answer. Do NOT explain incorrect options.
- Do NOT add an introduction, a conclusion, code fences, or extra headings.
- Keep LaTeX crisp. Use \\text{} for units. Write numbers with units where helpful (e.g., $18\\ \\text{W}$).
- The final answer must match the Correct Option / Correct Value given by the user.
- Work out the solution privately. The written steps must be a clean, linear derivation that reaches the Correct Option / Correct Value directly.
- NEVER write self-corrections or trial attempts (no "wait", "let me re-check", "re-evaluating", "if ... then ...", "discard", or retried values). If a setup seems inconsistent, silently re-derive it before writing.
"""

USER_TEMPLATE = """Question:
{question}

Subject: {subject}
Type: {q_type}
Correct Option: {correct_display}
Correct Value/Text: {correct_text}

Generate the solution in the specified format."""


class RowInput(BaseModel):
    pass  # placeholder to keep type hints readable; inputs are plain dicts


def build_chain():
    llm = ChatOpenAI(
        model=MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.01,
        reasoning_effort="medium",
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SYSTEM_PROMPT),  # literal, braces are not parsed
            ("user", USER_TEMPLATE),  # only this one has variables
        ]
    )
    return prompt | llm.with_structured_output(
        SolutionMarkdown, method="function_calling"
    )


def extract_correct_answer_details(row: dict[str, str]) -> tuple[str, str]:
    options_raw = row.get("options", "")
    correct_options_raw = row.get("correct_options", "")
    answer_raw = row.get("answer", "")

    correct_option_id = ""
    if correct_options_raw:
        try:
            corr_list = json.loads(correct_options_raw)
            if corr_list:
                correct_option_id = str(corr_list[0])
        except Exception:  # noqa: BLE001, S110
            pass

    display_id = ""
    ans_text = ""

    if options_raw:
        try:
            options_dict = json.loads(options_raw)
            for opt in options_dict.get("english", []):
                if str(opt.get("id")) == correct_option_id:
                    display_id = opt.get("displayId", "")
                    ans_text = opt.get("text", "")
                    break
        except Exception:  # noqa: BLE001, S110
            pass

    if not ans_text and answer_raw:
        try:
            ans_text = json.loads(answer_raw).get("english", "")
        except Exception:  # noqa: BLE001
            ans_text = answer_raw

    return display_id, ans_text


def load_existing_output(path: str) -> tuple[set[str], list[str] | None]:
    """Return ids that already have an explanation, and the existing header (if any)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set(), None
    with open(path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        done_ids = {
            row["id"]
            for row in reader
            if row.get("id") and (row.get("explanation_markdown") or "").strip()
        }
    return done_ids, header


def append_rows(path: str, header: list[str], rows: list[dict[str, str]]) -> None:
    """Append rows without touching existing content."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_has_content = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if not file_has_content:
            writer.writeheader()
        writer.writerows(rows)


def chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


async def process(n: int) -> None:
    done_ids, existing_header = load_existing_output(OUTPUT_CSV)

    # Collect the next n questions that have no image and no explanation yet
    with open(INPUT_CSV, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        input_fields = reader.fieldnames or []
        pending: list[dict[str, str]] = []
        for row in reader:
            if row.get("has_image", "").strip().lower() != "false":
                continue
            if row.get("id") in done_ids:
                continue
            pending.append(row)
            if len(pending) == n:
                break

    if not pending:
        print(
            "Nothing new to generate. All eligible questions already have explanations."
        )
        return

    # Keep the existing header if the file exists, so it is never rewritten
    if existing_header is None:
        header = list(input_fields) + [
            c for c in EXTRA_COLUMNS if c not in input_fields
        ]
    else:
        header = existing_header
        missing = [c for c in input_fields if c not in header]
        if missing:
            print(
                f"Warning: existing output lacks columns {missing}. "
                "Delete the output file once if you want them included."
            )

    chain = build_chain()
    processed = 0

    for batch in chunked(pending, BATCH_SIZE):
        inputs = []
        meta = []
        for row in batch:
            disp_id, ans_text = extract_correct_answer_details(row)
            inputs.append(
                {
                    "question": row.get("english", ""),
                    "subject": row.get("subject", ""),
                    "q_type": row.get("type", ""),
                    "correct_display": disp_id or "N/A",
                    "correct_text": ans_text,
                }
            )
            meta.append((row, disp_id, ans_text))

        print(f"Batch of {len(batch)}: " + ", ".join(r.get("id", "") for r in batch))
        results = await chain.abatch(
            inputs,
            config={"max_concurrency": MAX_CONCURRENCY},
            return_exceptions=True,
        )

        new_rows = []
        for (row, disp_id, ans_text), result in zip(meta, results):
            if isinstance(result, Exception):
                print(f"  FAILED ID {row.get('id')}: {result}")
                continue
            out = {col: row.get(col, "") for col in input_fields}
            out.update(
                {
                    "correct_display_id": disp_id,
                    "correct_answer": ans_text,
                    "explanation_markdown": result.markdown_explanation,
                }
            )
            new_rows.append(out)

        # Save after each batch so progress is never lost
        append_rows(OUTPUT_CSV, header, new_rows)
        processed += len(new_rows)
        print(f"  Saved {len(new_rows)} row(s).")

    print(f"Done. Added {processed} new record(s) to {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(process(DEFAULT_N))
