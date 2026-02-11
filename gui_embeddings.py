"""
GUI за генериране на embeddings с OpenAI и Gemini.
(Anthropic няма собствен embedding API – препоръчват Voyage AI.)
Стартиране: python gui_embeddings.py
"""
import base64
import json
import os
import struct
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _openai_client():
    from openai_api import client as m
    return m


def _gemini_client():
    from gemini import client as m
    return m


# OpenAI embeddings: всички body параметри от POST /embeddings
OPENAI_ENCODING_FORMAT = ["float", "base64"]  # encoding_format
# Модели за embeddings
OPENAI_EMBED_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]
GEMINI_EMBED_MODELS = ["gemini-embedding-001"]
GEMINI_TASK_TYPES = [
    "",
    "SEMANTIC_SIMILARITY",
    "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY",
    "CLASSIFICATION",
    "CLUSTERING",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
    "CODE_RETRIEVAL_QUERY",
]


class EmbeddingGui:
    def __init__(self):
        self.win = tk.Tk()
        self.win.title("Embeddings – OpenAI & Gemini")
        self.win.geometry("880x680")
        self.win.minsize(600, 500)

        self.api_var = tk.StringVar(value="OpenAI")
        self.opt_widgets = {}
        self.request_text = None
        self.response_text = None
        self._full_response_text = ""  # пълен отговор с всички числа за копиране
        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.win, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Провайдър
        row0 = ttk.Frame(main)
        row0.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row0, text="Провайдър:").pack(side=tk.LEFT, padx=(0, 6))
        api_combo = ttk.Combobox(
            row0, textvariable=self.api_var,
            values=["OpenAI", "Gemini"],
            state="readonly", width=14
        )
        api_combo.pack(side=tk.LEFT)
        api_combo.bind("<<ComboboxSelected>>", self._on_api_change)
        ttk.Label(row0, text="  (Anthropic няма embedding API)", font=("", 9)).pack(side=tk.LEFT)
        ttk.Label(
            main,
            text="Провайдър: кой API да генерира векторите – OpenAI (text-embedding-3, ada-002) или Gemini (gemini-embedding-001).",
            font=("", 8),
            foreground="gray",
        ).pack(anchor=tk.W, pady=(0, 4))

        # Опции
        self.opts_frame = ttk.LabelFrame(main, text="Параметри", padding=6)
        self.opts_frame.pack(fill=tk.X, pady=(0, 8))
        self._fill_opts("OpenAI")

        # Входен текст (ред по ред = отделни текстове, или един блок)
        ttk.Label(main, text="Текст(ове) за embedding:").pack(anchor=tk.W)
        self.input_text = tk.Text(main, height=5, wrap=tk.WORD, font=("Segoe UI", 10))
        self.input_text.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(
            main,
            text="Един ред = един текст (един вектор). Или напишете един блок – ще се третира като един вход. Текстът не трябва да надвишава лимита за токени на модела.",
            font=("", 8),
            foreground="gray",
        ).pack(anchor=tk.W, pady=(0, 8))

        btn_row = ttk.Frame(main)
        btn_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(btn_row, text="Генерирай embeddings", command=self._generate).pack(side=tk.LEFT, padx=(0, 8))

        # Заявка / Отговор
        paned = ttk.PanedWindow(main, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        req_f = ttk.LabelFrame(paned, text="Заявка (request)", padding=4)
        self.request_text = scrolledtext.ScrolledText(req_f, height=4, wrap=tk.WORD, font=("Consolas", 9), state=tk.DISABLED)
        self.request_text.pack(fill=tk.BOTH, expand=True)
        paned.add(req_f, weight=1)
        res_f = ttk.LabelFrame(paned, text="Отговор (response) – размер на векторите и първи стойности", padding=4)
        res_head = ttk.Frame(res_f)
        res_head.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(res_head, text="Копирай целия отговор", command=self._copy_response).pack(side=tk.RIGHT)
        self.response_text = scrolledtext.ScrolledText(res_f, height=8, wrap=tk.WORD, font=("Consolas", 9), state=tk.DISABLED)
        self.response_text.pack(fill=tk.BOTH, expand=True)
        paned.add(res_f, weight=2)

        self.status_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.status_var, font=("", 9)).pack(anchor=tk.W)

    def _clear_opts(self):
        for w in self.opts_frame.winfo_children():
            w.destroy()
        self.opt_widgets = {}

    def _fill_opts(self, api: str):
        self._clear_opts()
        self.opt_widgets = {}
        desc_font = ("", 8)
        if api == "OpenAI":
            r1 = ttk.Frame(self.opts_frame)
            r1.pack(fill=tk.X, pady=2)
            ttk.Label(r1, text="Модел:").pack(side=tk.LEFT, padx=(0, 6))
            c = ttk.Combobox(r1, values=OPENAI_EMBED_MODELS, state="readonly", width=26)
            c.set("text-embedding-3-small")
            c.pack(side=tk.LEFT, padx=(0, 16))
            self.opt_widgets["model"] = c
            ttk.Label(r1, text="dimensions:").pack(side=tk.LEFT, padx=(0, 6))
            dim_entry = ttk.Entry(r1, width=6)
            dim_entry.pack(side=tk.LEFT, padx=(0, 16))
            self.opt_widgets["dimensions"] = dim_entry
            ttk.Label(r1, text="encoding_format:").pack(side=tk.LEFT, padx=(0, 6))
            enc = ttk.Combobox(r1, values=OPENAI_ENCODING_FORMAT, state="readonly", width=8)
            enc.set("float")
            enc.pack(side=tk.LEFT, padx=(0, 16))
            self.opt_widgets["encoding_format"] = enc
            ttk.Label(r1, text="user (optional):").pack(side=tk.LEFT, padx=(0, 6))
            user_entry = ttk.Entry(r1, width=16)
            user_entry.pack(side=tk.LEFT)
            self.opt_widgets["user"] = user_entry
            # Описания на български под параметрите
            r1_desc = ttk.Frame(self.opts_frame)
            r1_desc.pack(fill=tk.X, pady=(0, 4))
            ttk.Label(
                r1_desc,
                text="Модел: кой embedding модел да се използва.  dimensions: брой измерения на вектора (само за text-embedding-3).  encoding_format: float = числа, base64 = кодиран низ.  user: по избор – идентификатор на потребител за мониторинг.",
                font=desc_font,
                foreground="gray",
            ).pack(anchor=tk.W)
        else:
            row = ttk.Frame(self.opts_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text="Модел:").pack(side=tk.LEFT, padx=(0, 6))
            c = ttk.Combobox(row, values=GEMINI_EMBED_MODELS, state="readonly", width=24)
            c.set("gemini-embedding-001")
            c.pack(side=tk.LEFT, padx=(0, 16))
            self.opt_widgets["model"] = c
            ttk.Label(row, text="task_type:").pack(side=tk.LEFT, padx=(0, 6))
            c2 = ttk.Combobox(row, values=GEMINI_TASK_TYPES, state="readonly", width=22)
            c2.set("")
            c2.pack(side=tk.LEFT, padx=(0, 16))
            self.opt_widgets["task_type"] = c2
            ttk.Label(row, text="output_dimensionality:").pack(side=tk.LEFT, padx=(0, 6))
            dim_entry = ttk.Entry(row, width=6)
            dim_entry.insert(0, "")
            dim_entry.pack(side=tk.LEFT)
            self.opt_widgets["output_dimensionality"] = dim_entry
            # Описания на български под параметрите
            row_desc = ttk.Frame(self.opts_frame)
            row_desc.pack(fill=tk.X, pady=(0, 4))
            ttk.Label(
                row_desc,
                text="Модел: embedding модел на Gemini.  task_type: за какво ще ползвате векторите (търсене, сходство, класификация и др.) – празно = по подразбиране.  output_dimensionality: размер на вектора (напр. 768, 1536, 3072) – празно = по подразбиране.",
                font=desc_font,
                foreground="gray",
            ).pack(anchor=tk.W)

    def _on_api_change(self, event=None):
        self._fill_opts(self.api_var.get())

    def _update_request(self, payload: dict, hint: str = ""):
        if not self.request_text:
            return
        self.request_text.config(state=tk.NORMAL)
        self.request_text.delete(1.0, tk.END)
        lines = [f"{hint}\n"] if hint else []
        try:
            lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception:
            lines.append(str(payload))
        self.request_text.insert(tk.END, "".join(lines))
        self.request_text.config(state=tk.DISABLED)

    def _update_response(self, text: str):
        if not self.response_text:
            return
        self.response_text.config(state=tk.NORMAL)
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, text)
        self.response_text.config(state=tk.DISABLED)

    def _copy_response(self):
        """Копира целия отговор с всички числа на векторите в клипборда."""
        text = self._full_response_text.strip() if getattr(self, "_full_response_text", None) else ""
        if not text and self.response_text:
            text = self.response_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showinfo("Копиране", "Няма съдържание за копиране.")
            return
        self.win.clipboard_clear()
        self.win.clipboard_append(text)
        self.win.update_idletasks()
        self.status_var.set("Целият отговор (с всички числа) е копиран в клипборда.")

    def _get_inputs(self) -> list[str]:
        raw = self.input_text.get(1.0, tk.END).strip()
        if not raw:
            return []
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(lines) == 1 and "\n" not in raw and len(raw) > 80:
            return [raw]
        return lines if lines else [raw]

    def _generate(self):
        inputs = self._get_inputs()
        if not inputs:
            messagebox.showwarning("Грешка", "Въведи поне един текст.")
            return
        self.status_var.set("Генериране...")

        def run():
            try:
                api = self.api_var.get()
                if api == "OpenAI":
                    self._do_openai(inputs)
                else:
                    self._do_gemini(inputs)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                self.win.after(0, lambda: self._show_error(err))

        threading.Thread(target=run, daemon=True).start()

    def _do_openai(self, inputs: list[str]):
        mod = _openai_client()
        model = self.opt_widgets.get("model")
        model = model.get() if model else "text-embedding-3-small"
        dim_w = self.opt_widgets.get("dimensions")
        dimensions = None
        if dim_w and dim_w.get().strip():
            try:
                dimensions = int(dim_w.get().strip())
            except ValueError:
                pass
        enc_w = self.opt_widgets.get("encoding_format")
        encoding_format = enc_w.get() if enc_w else "float"
        user_w = self.opt_widgets.get("user")
        user = (user_w.get() or "").strip() or None
        payload = {"model": model, "input": inputs[0] if len(inputs) == 1 else inputs}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        payload["encoding_format"] = encoding_format
        if user:
            payload["user"] = user
        self.win.after(0, lambda: self._update_request(payload, "OpenAI POST /embeddings\n"))
        resp = mod.embeddings_create(
            inputs[0] if len(inputs) == 1 else inputs,
            model=model,
            dimensions=dimensions,
            encoding_format=encoding_format,
            user=user,
        )
        # Суров отговор – брой вектори, размер, първи стойности (при base64 декодираме)
        try:
            data = resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()
            vectors = []
            for d in data.get("data", []):
                emb = d.get("embedding", [])
                if isinstance(emb, str):
                    decoded = base64.b64decode(emb)
                    emb = list(struct.unpack(f"<{len(decoded)//4}f", decoded))
                vectors.append(emb if isinstance(emb, list) else list(emb))
        except Exception:
            vectors = []
            for obj in resp.data:
                e = getattr(obj, "embedding", None)
                if isinstance(e, str):
                    decoded = base64.b64decode(e)
                    e = list(struct.unpack(f"<{len(decoded)//4}f", decoded))
                vectors.append(e if e is not None else [])
        summary = self._format_embedding_response(vectors, len(inputs))
        full_text = self._format_embedding_response(vectors, len(inputs), full=True)
        self.win.after(0, lambda s=summary: self._update_response(s))
        self.win.after(0, lambda ft=full_text: setattr(self, "_full_response_text", ft))
        self.win.after(0, lambda: self.status_var.set("Готово."))

    def _do_gemini(self, inputs: list[str]):
        mod = _gemini_client()
        model = self.opt_widgets.get("model")
        model = model.get() if model else "gemini-embedding-001"
        task_w = self.opt_widgets.get("task_type")
        task_type = (task_w.get() or "").strip() or None
        dim_w = self.opt_widgets.get("output_dimensionality")
        output_dimensionality = None
        if dim_w and dim_w.get().strip():
            try:
                output_dimensionality = int(dim_w.get().strip())
            except ValueError:
                pass
        payload = {
            "model": model,
            "contents": inputs[0] if len(inputs) == 1 else inputs,
        }
        if task_type:
            payload["task_type"] = task_type
        if output_dimensionality is not None:
            payload["output_dimensionality"] = output_dimensionality
        self.win.after(0, lambda: self._update_request(payload, "Gemini embed_content\n"))
        vectors = mod.embed_content(
            inputs[0] if len(inputs) == 1 else inputs,
            model=model,
            task_type=task_type,
            output_dimensionality=output_dimensionality,
        )
        summary = self._format_embedding_response(vectors, len(inputs))
        full_text = self._format_embedding_response(vectors, len(inputs), full=True)
        self.win.after(0, lambda s=summary: self._update_response(s))
        self.win.after(0, lambda ft=full_text: setattr(self, "_full_response_text", ft))
        self.win.after(0, lambda: self.status_var.set("Готово."))

    def _format_embedding_response(
        self, vectors: list[list[float]], num_inputs: int, *, full: bool = False
    ) -> str:
        lines = [f"Брой вектори: {len(vectors)}", f"Брой входни текста: {num_inputs}", ""]
        for i, vec in enumerate(vectors):
            dim = len(vec)
            if full:
                lines.append(f"Вектор {i + 1}: размер {dim}")
                lines.append(json.dumps(vec, ensure_ascii=False))
            else:
                preview = vec[:10] if len(vec) >= 10 else vec
                lines.append(f"Вектор {i + 1}: размер {dim}, първи стойности: {preview}")
                if len(vec) > 10:
                    lines.append(f"  ... (още {dim - 10} стойности)")
        return "\n".join(lines)

    def _show_error(self, msg: str):
        self.status_var.set("")
        self._full_response_text = ""
        self._update_response(f"Грешка: {msg}")
        messagebox.showerror("Грешка", msg)

    def run(self):
        self.win.mainloop()


if __name__ == "__main__":
    app = EmbeddingGui()
    app.run()
