from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import textwrap


model_id = "Qwen/Qwen2.5-7B-Instruct-AWQ"

SYSTEM_PROMPT = textwrap.dedent("""
    You are GitLit, an AI assistant that helps developers understand code repositories.
    You will be given relevant code chunks retrieved from a git repository, along with a question about the codebase.

    Your job is to answer the question accurately and concisely based solely on the provided code chunks.

    Guidelines:
    - Base your answer strictly on the provided code chunks. Do not invent or assume code that is not shown.
    - If the answer cannot be determined from the provided chunks, say so clearly.
    - When referencing code, mention the file path so the developer knows where to look.
    - If the question involves multiple files or functions, explain how they relate to each other.
    - Keep your answer focused and developer-friendly — avoid unnecessary verbosity.
""").strip()

def load_llm():
    torch.backends.mps.is_available = lambda: False
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

def build_user_message(query: str, chunks: list[dict]) -> str:
        context = "\n\n".join([
            f"{{path: {chunk['path']} | language: {chunk['language']} | lines {chunk['start_line']}-{chunk['end_line']}}}\n{chunk['content']}"
            for chunk in chunks
        ])
        return f"Here are the relevant code chunks from the repository:\n\n{context}\n\n---\n\nQuestion: {query}"

def generate_answer(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        query: str,
        chunks: list[dict],
        max_new_tokens: int = 2048,
        do_sample: bool = True,
        temperature: float = 0.2
    ) -> str:

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(query, chunks)},
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return answer