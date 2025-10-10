from datetime import datetime

from shared.clients import get_langfuse_client, get_openai_client
from langfuse_helpers import (
    run_with_chat_prompt,
    run_with_text_prompt,
    parse_judge_response,
    format_reasoning_summary
)

# Initialize clients
langfuse = get_langfuse_client()
oai_client = get_openai_client()

# Configuration
DATASET_NAME = "dataset_docgen"
PROMPT_A_NAME = "prompt_doc_a"
PROMPT_B_NAME = "prompt_doc_b"
PROMPT_JUDGE_NAME = "llm_judge_pairwise"

timestamp = datetime.now().strftime("%H%M")

if __name__ == "__main__":
    print("="*80)
    print("LANGFUSE PAIRWISE EVALUATION")
    print("="*80)

    # 1. Load prompts from Langfuse
    print("\n1. Loading prompts from Langfuse...")
    prompt_a = langfuse.get_prompt(PROMPT_A_NAME, label="production")
    prompt_b = langfuse.get_prompt(PROMPT_B_NAME, label="production")
    prompt_judge = langfuse.get_prompt(PROMPT_JUDGE_NAME, label="evaluation")
    print(f"   ✓ Loaded: {PROMPT_A_NAME}")
    print(f"   ✓ Loaded: {PROMPT_B_NAME}")
    print(f"   ✓ Loaded: {PROMPT_JUDGE_NAME}")

    # 2. Load dataset
    print(f"\n2. Loading dataset '{DATASET_NAME}'...")
    dataset = langfuse.get_dataset(DATASET_NAME)
    items_list = list(dataset.items)
    print(f"   ✓ Dataset loaded with {len(items_list)} items")

    # 3. Run experiments
    print(f"\n3. Running pairwise evaluation (timestamp: {timestamp})...")
    print("-"*80)

    for idx, item in enumerate(items_list, 1):
        print(f"\n📄 Item {idx}/{len(items_list)}")
        print(f"   Input: {str(item.input)[:100]}...")

        # Run Prompt A
        print("   → Running Prompt A...")
        experiment_a_name = f"ExperimentA_{timestamp}"
        with item.run(
            run_name=experiment_a_name,
            run_metadata={"experiment": "DocPromptA", "prompt": "A", "timestamp": timestamp}
        ) as run_a:
            output_a = run_with_chat_prompt(prompt_a, item.input, oai_client)
            trace_id_a = run_a.trace_id
            print(f"      ✓ Generated {len(output_a)} chars (trace: {trace_id_a[:8]}...)")

        # Run Prompt B
        print("   → Running Prompt B...")
        experiment_b_name = f"ExperimentB_{timestamp}"
        with item.run(
            run_name=experiment_b_name,
            run_metadata={"experiment": "DocPromptB", "prompt": "B", "timestamp": timestamp}
        ) as run_b:
            output_b = run_with_chat_prompt(prompt_b, item.input, oai_client)
            trace_id_b = run_b.trace_id
            print(f"      ✓ Generated {len(output_b)} chars (trace: {trace_id_b[:8]}...)")

        # Run Pairwise Judge
        print("   → Running Pairwise Judge...")
        with item.run(
            run_name=f"PairwiseJudge_{timestamp}",
            run_metadata={"experiment": "PairwiseJudge", "type": "evaluation", "timestamp": timestamp}
        ) as run_judge:
            # Prepare judge inputs
            judge_response = run_with_text_prompt(
                prompt_judge,
                oai_client,
                code=str(item.input.get("files", "")),
                reference=str(item.expected_output.get("reference", "")),
                answer_a=output_a,
                answer_b=output_b
            )

            # Parse judge decision
            decision, reasoning = parse_judge_response(judge_response)

            # Extract scores from reasoning
            score_a = reasoning.get("score_total_a", "?")
            score_b = reasoning.get("score_total_b", "?")

            # Determine winner text for display
            winner_text = experiment_a_name if decision == "A" else (experiment_b_name if decision == "B" else "TIE")

            # Score the judge run itself - use string value to show experiment name
            run_judge.score(
                name="Winner",
                value=winner_text,  # String value shows in the cell
                data_type="CATEGORICAL",
                comment=f"Scores: A={score_a}/50, B={score_b}/50\n\n{format_reasoning_summary(reasoning)}"
            )

            print(f"      ✓ Decision: {decision} (A: {score_a}/50, B: {score_b}/50)")

        # Add scores to the original runs
        print("   → Adding comparison scores to runs...")

        # Score for Prompt A - use categorical value to show result text
        result_a = "Won" if decision == "A" else ("Tie" if decision == "TIE" else "Lost")
        langfuse.create_score(
            trace_id=trace_id_a,
            name="Pairwise Result",
            value=result_a,  # String value shows in the cell
            data_type="CATEGORICAL",
            comment=f"{result_a} against {experiment_b_name}\n\n{format_reasoning_summary(reasoning)}"
        )

        # Score for Prompt B - use categorical value to show result text
        result_b = "Won" if decision == "B" else ("Tie" if decision == "TIE" else "Lost")
        langfuse.create_score(
            trace_id=trace_id_b,
            name="Pairwise Result",
            value=result_b,  # String value shows in the cell
            data_type="CATEGORICAL",
            comment=f"{result_b} against {experiment_a_name}\n\n{format_reasoning_summary(reasoning)}"
        )

        print(f"      ✓ Scores added")

    # Flush to ensure all data is sent
    print("\n4. Flushing data to Langfuse...")
    langfuse.flush()
    print("   ✓ All data sent")

    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)
    print(f"\nView results in Langfuse UI:")
    print(f"  → Datasets: http://localhost:3000/project/[project-id]/datasets/{DATASET_NAME}")
    print(f"  → Filter by timestamp: {timestamp}")
    print(f"\nDataset runs created:")
    print(f"  - ExperimentA_{timestamp}")
    print(f"  - ExperimentB_{timestamp}")
    print(f"  - PairwiseJudge_{timestamp}")
