from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import torch

tokenizer = PreTrainedTokenizerFast.from_pretrained("digit82/kobart-summarization")
model = BartForConditionalGeneration.from_pretrained("digit82/kobart-summarization")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def summarize_text(text: str) -> str:
    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    # token_type_ids 제외
    inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
    inputs = {k: v.to(device) for k, v in inputs.items()}

    summary_ids = model.generate(
        **inputs,
        max_length=130,
        min_length=30,
        do_sample=False
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print("\n=== 입력 원문 ===\n", text)
    print("\n=== 요약 결과 ===\n", summary)

    return summary
