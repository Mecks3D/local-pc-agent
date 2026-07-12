#!/usr/bin/env python3
"""Test rapido: quanto spesso Qwen2.5-0.5B-Instruct produce un tool-call JSON valido al primo colpo."""

from llama_cpp import Llama
from agent import build_system_prompt, extract_json, MODEL_PATH

TEST_PROMPTS = [
    "che ore sono?",
    "elenca i file nella mia home",
    "quanto spazio disco libero c'è?",
    "mostrami il contenuto di /etc/hostname",
    "cancella tutti i file nella cartella /tmp con rm -rf",
    "dimmi una barzelletta",
    "qual è il mio indirizzo IP locale?",
    "quanta RAM è libera?",
    "crea una cartella chiamata test_agent nella home",
    "chi sono io su questo sistema (whoami)?",
]

def main():
    llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_threads=4, verbose=False)
    system_prompt = build_system_prompt()

    ok, bad = 0, 0
    for prompt in TEST_PROMPTS:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = llm.create_chat_completion(
            messages=messages, temperature=0.2, max_tokens=256, stop=["<|im_end|>"]
        )
        raw = response["choices"][0]["message"]["content"]
        parsed = extract_json(raw)

        valid = parsed is not None and "tool" in parsed and parsed["tool"] in (
            "run_shell", "list_dir", "read_file", "final_answer"
        )
        ok += valid
        bad += not valid

        status = "OK " if valid else "FAIL"
        print(f"[{status}] '{prompt}'")
        print(f"       raw: {raw!r}")
        print(f"       parsed: {parsed}\n")

    print(f"\nTotale: {ok} validi / {bad} non validi su {len(TEST_PROMPTS)}")

if __name__ == "__main__":
    main()
