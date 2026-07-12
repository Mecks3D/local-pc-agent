#!/usr/bin/env python3
"""
Agente locale di controllo PC via linguaggio naturale.
Modello: Qwen2.5-0.5B-Instruct (GGUF, Q8_0), inferenza CPU con llama-cpp-python.

Loop: prompt utente -> il modello risponde con un tool call in JSON
      -> conferma se il comando e' pericoloso -> esecuzione via subprocess
      -> il risultato torna al modello -> il modello formula la risposta finale.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

from llama_cpp import Llama

MODEL_PATH = str(Path.home() / "models" / "qwen2.5-0.5b-instruct-q8_0.gguf")
MAX_TURNS = 6          # quante iterazioni di tool-calling per richiesta, prima di forzare una risposta
CTX_SIZE = 4096
MAX_TOOL_OUTPUT = 4000  # tronca stdout/stderr lunghissimi prima di rimandarli al modello

# ---------------------------------------------------------------------------
# Definizione dei tool disponibili al modello
# ---------------------------------------------------------------------------

TOOLS = {
    "run_shell": {
        "description": "Esegue un comando shell sul sistema e ne restituisce stdout/stderr/exit code.",
        "params": {"command": "stringa: il comando shell da eseguire"},
    },
    "list_dir": {
        "description": "Elenca il contenuto di una directory.",
        "params": {"path": "stringa: percorso della directory (default '.')"},
    },
    "read_file": {
        "description": "Legge il contenuto testuale di un file (primi 4000 caratteri).",
        "params": {"path": "stringa: percorso del file"},
    },
    "final_answer": {
        "description": "Usalo quando hai finito e vuoi rispondere all'utente in linguaggio naturale, senza eseguire altri comandi.",
        "params": {"text": "stringa: la risposta finale per l'utente"},
    },
}

# Comandi/pattern considerati pericolosi: richiedono conferma esplicita anche
# se l'utente ha gia' accettato l'esecuzione generica dei comandi shell.
DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bpoweroff\b",
    r"\bhalt\b",
    r"\b(sudo|su)\b",
    r">\s*/dev/sd",
    r"\bchmod\s+-R\s+777\b",
    r"\bchown\s+-R\b",
    r":\(\)\{.*\};:",       # fork bomb
    r"\bkill\s+-9\s+1\b",
    r"\bmv\s+.*\s+/dev/null",
    r"\biptables\b",
    r"\buserdel\b",
    r"\bpasswd\b",
    r"\bcurl\b.*\|\s*sh",
    r"\bwget\b.*\|\s*sh",
]


def is_dangerous(command: str) -> bool:
    return any(re.search(p, command, re.IGNORECASE) for p in DANGEROUS_PATTERNS)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    tools_desc = json.dumps(
        {name: spec["description"] for name, spec in TOOLS.items()}, ensure_ascii=False, indent=2
    )
    return f"""Sei un agente che controlla un PC Linux eseguendo comandi per conto dell'utente.

Hai accesso a questi tool:
{tools_desc}

REGOLE FERREE:
1. Rispondi SEMPRE E SOLO con un oggetto JSON valido, su una riga, senza testo prima o dopo, senza markdown, senza backtick.
2. Il formato deve essere esattamente: {{"tool": "<nome_tool>", "args": {{...}}}}
3. Usa ESATTAMENTE i nomi di parametro definiti per ogni tool (es. "command" per run_shell, "path" per list_dir/read_file, "text" per final_answer). Non inventare nomi di parametro diversi.
4. Per rispondere all'utente senza eseguire nulla (chiacchiera, domande generiche, opinioni), usa SEMPRE il tool "final_answer".
5. Non inventare mai l'esito di un comando: se ti serve un'informazione dal sistema, chiama un tool per ottenerla, non supporre il risultato.
6. Un tool alla volta. Aspetta il risultato prima di decidere il prossimo passo.

Esempi:
Utente: che ore sono?
Tu: {{"tool": "run_shell", "args": {{"command": "date"}}}}

Utente: elenca i file nella home
Tu: {{"tool": "list_dir", "args": {{"path": "~"}}}}

Utente: mostrami il contenuto di /etc/hostname
Tu: {{"tool": "read_file", "args": {{"path": "/etc/hostname"}}}}

Utente: dimmi una barzelletta
Tu: {{"tool": "final_answer", "args": {{"text": "Perché i programmatori confondono Halloween e Natale? Perché OCT 31 == DEC 25."}}}}

Utente: quanto spazio disco libero c'è?
Tu: {{"tool": "run_shell", "args": {{"command": "df -h"}}}}
"""


# ---------------------------------------------------------------------------
# Parsing dell'output del modello
# ---------------------------------------------------------------------------

def extract_json(text: str):
    """Estrae il primo oggetto JSON valido dal testo generato dal modello."""
    text = text.strip()
    # rimuove eventuali code-fence markdown, il modello a volte le aggiunge comunque
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

    # tentativo diretto
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # fallback: cerca la prima { ... } bilanciata nel testo
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


# ---------------------------------------------------------------------------
# Esecuzione dei tool
# ---------------------------------------------------------------------------

def confirm(prompt: str) -> bool:
    answer = input(f"{prompt} [s/N] ").strip().lower()
    return answer in ("s", "si", "sì", "y", "yes")


def execute_tool(name: str, args: dict) -> str:
    if name == "run_shell":
        command = args.get("command", "")
        if not command:
            return "ERRORE: nessun comando specificato."

        if is_dangerous(command):
            print(f"\n⚠️  Comando potenzialmente PERICOLOSO: {command}")
            if not confirm("Confermi l'esecuzione?"):
                return "ESEGUZIONE ANNULLATA dall'utente (comando pericoloso rifiutato)."
        else:
            print(f"\n→ Comando proposto: {command}")
            if not confirm("Eseguire?"):
                return "ESEGUZIONE ANNULLATA dall'utente."

        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
            output = f"exit_code={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        except subprocess.TimeoutExpired:
            output = "ERRORE: comando terminato per timeout (30s)."
        return output[:MAX_TOOL_OUTPUT]

    elif name == "list_dir":
        path = args.get("path", ".")
        try:
            entries = sorted(Path(path).expanduser().iterdir())
            return "\n".join(str(e.name) + ("/" if e.is_dir() else "") for e in entries)[:MAX_TOOL_OUTPUT]
        except Exception as e:
            return f"ERRORE: {e}"

    elif name == "read_file":
        path = args.get("path", "")
        try:
            content = Path(path).expanduser().read_text(errors="replace")
            return content[:MAX_TOOL_OUTPUT]
        except Exception as e:
            return f"ERRORE: {e}"

    elif name == "final_answer":
        return args.get("text", "")

    else:
        return f"ERRORE: tool sconosciuto '{name}'."


# ---------------------------------------------------------------------------
# Loop principale
# ---------------------------------------------------------------------------

def main():
    print(f"Caricamento modello da {MODEL_PATH} ...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=CTX_SIZE,
        n_threads=4,
        verbose=False,
    )
    print("Modello caricato. Digita 'exit' per uscire.\n")

    system_prompt = build_system_prompt()

    while True:
        try:
            user_input = input("Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in ("exit", "quit", "esci"):
            break
        if not user_input:
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        for turn in range(MAX_TURNS):
            response = llm.create_chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=512,
                stop=["<|im_end|>"],
            )
            raw = response["choices"][0]["message"]["content"]

            parsed = extract_json(raw)
            if parsed is None or "tool" not in parsed:
                print(f"Agente [output non valido]: {raw}")
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": "La tua risposta non era JSON valido nel formato richiesto. "
                        'Rispondi SOLO con {"tool": "...", "args": {...}}.',
                    }
                )
                continue

            tool_name = parsed.get("tool")
            tool_args = parsed.get("args", {})

            if tool_name == "final_answer":
                print(f"Agente: {tool_args.get('text', '')}\n")
                break

            print(f"[tool call] {tool_name}({tool_args})")
            result = execute_tool(tool_name, tool_args)
            print(f"[risultato] {result}\n")

            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": f"Risultato del tool '{tool_name}':\n{result}"}
            )
        else:
            print("Agente: (troppi passaggi, mi fermo qui)\n")


if __name__ == "__main__":
    main()
