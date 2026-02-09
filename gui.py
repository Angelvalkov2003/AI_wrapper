"""
Просто GUI за тестване на Gemini, OpenAI и Anthropic API.
Показва всички опции, изпратената заявка и суровия отговор.
Стартиране от папката ai_apis: python gui.py
"""
import json
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font as tkfont

# Да се намери коренът на проекта и да се импортират клиентите
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Лениви импорти за клиентите (при избор на API)
def _import_gemini():
    from gemini import client as m
    return m

def _import_openai():
    from openai import client as m
    return m

def _import_anthropic():
    from anthropic import client as m
    return m


def _format_request(api: str, payload: dict) -> str:
    """Форматира заявката за показ в GUI."""
    try:
        return json.dumps(payload, indent=2, ensure_ascii=False)
    except Exception:
        return str(payload)


# Предопределени модели за dropdown
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
]
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]
ANTHROPIC_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307",
]


def _format_response(obj) -> str:
    """Форматира отговора (текст или обект) за показ."""
    if isinstance(obj, str):
        return obj
    try:
        if hasattr(obj, "model_dump"):
            return json.dumps(obj.model_dump(), indent=2, ensure_ascii=False, default=str)
        if hasattr(obj, "__dict__"):
            d = getattr(obj, "__dict__", {})
            return json.dumps(d, indent=2, ensure_ascii=False, default=str)
        return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


class ApiGui:
    def __init__(self):
        self.win = tk.Tk()
        self.win.title("AI APIs – Заявка и отговор")
        self.win.geometry("1000x800")
        self.win.minsize(700, 500)

        self.api_var = tk.StringVar(value="Gemini")
        self.stream_var = tk.BooleanVar(value=False)
        self.request_text = None
        self.response_text = None
        self.prompt_text = None
        self.system_text = None
        self.opt_frame = None
        self.entries = {}
        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.win, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        # --- Горен ред: API + Стрийминг + БУТОН ИЗПРАТИ (винаги видим) ---
        row0 = ttk.Frame(main)
        row0.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row0, text="API:").pack(side=tk.LEFT, padx=(0, 6))
        api_combo = ttk.Combobox(
            row0, textvariable=self.api_var,
            values=["Gemini", "OpenAI", "Anthropic"],
            state="readonly", width=14
        )
        api_combo.pack(side=tk.LEFT)
        api_combo.bind("<<ComboboxSelected>>", self._on_api_change)
        ttk.Checkbutton(row0, text="Стрийминг", variable=self.stream_var).pack(side=tk.LEFT, padx=(20, 0))
        ttk.Button(row0, text="  Изпрати заявка  ", command=self._send).pack(side=tk.LEFT, padx=(24, 0))

        # --- Скролваемо съдържание (параметри, prompt, заявка/отговор) ---
        scroll_container = ttk.Frame(main)
        scroll_container.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(scroll_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = tk.Canvas(scroll_container, highlightthickness=0, yscrollcommand=scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.canvas.yview)

        scroll_inner = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=scroll_inner, anchor=tk.NW)

        def _on_frame_configure(event=None):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        def _on_canvas_configure(event):
            self.canvas.itemconfig(self.canvas_window, width=event.width)
        scroll_inner.bind("<Configure>", _on_frame_configure)
        self.canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.canvas.bind("<MouseWheel>", _on_mousewheel)
        scroll_inner.bind("<MouseWheel>", _on_mousewheel)

        # --- Параметри ---
        self.opt_frame = ttk.LabelFrame(scroll_inner, text="Параметри", padding=6)
        self.opt_frame.pack(fill=tk.X, pady=(0, 8))
        self._fill_options("Gemini")

        # --- Prompt / System ---
        prompt_frame = ttk.Frame(scroll_inner)
        prompt_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(prompt_frame, text="System (опционално) – инструкции за ролята/стила на асистента:").pack(anchor=tk.W)
        self.system_text = scrolledtext.ScrolledText(prompt_frame, height=2, wrap=tk.WORD, font=("Consolas", 10))
        self.system_text.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(prompt_frame, text="Съобщение (prompt) – въпрос или задача към модела:").pack(anchor=tk.W)
        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, height=4, wrap=tk.WORD, font=("Consolas", 10))
        self.prompt_text.pack(fill=tk.X, pady=(0, 6))

        # --- Заявка / Отговор ---
        paned = ttk.PanedWindow(scroll_inner, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        req_frame = ttk.LabelFrame(paned, text="Изпратена заявка (request)", padding=4)
        self.request_text = scrolledtext.ScrolledText(req_frame, height=8, wrap=tk.WORD, font=("Consolas", 9), state=tk.DISABLED)
        self.request_text.pack(fill=tk.BOTH, expand=True)
        paned.add(req_frame, weight=1)

        res_frame = ttk.LabelFrame(paned, text="Суров отговор (response)", padding=4)
        self.response_text = scrolledtext.ScrolledText(res_frame, height=12, wrap=tk.WORD, font=("Consolas", 9), state=tk.DISABLED)
        self.response_text.pack(fill=tk.BOTH, expand=True)
        paned.add(res_frame, weight=2)

    def _clear_options(self):
        for w in self.opt_frame.winfo_children():
            w.destroy()
        self.entries.clear()

    def _add_row(self, label: str, key: str, default="", width=20):
        """Ред с етикет и текстово поле. label може да е многоредов (описание под името)."""
        row = ttk.Frame(self.opt_frame)
        row.pack(fill=tk.X, pady=4)
        lbl_frame = ttk.Frame(row)
        lbl_frame.pack(side=tk.LEFT, padx=(0, 8), anchor=tk.N)
        lines = label.split("\n")
        for i, line in enumerate(lines):
            lbl = ttk.Label(lbl_frame, text=line, anchor=tk.W, font=("", 9 if i > 0 else 10))
            if i > 0:
                lbl.config(wraplength=320)
            lbl.pack(anchor=tk.W)
        e = ttk.Entry(row, width=width)
        e.insert(0, default)
        e.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entries[key] = e

    def _add_row_combo(self, label: str, key: str, values: list, default: str | None = None):
        """Ред с етикет и падащ списък (dropdown) за избор на стойност."""
        row = ttk.Frame(self.opt_frame)
        row.pack(fill=tk.X, pady=4)
        lbl_frame = ttk.Frame(row)
        lbl_frame.pack(side=tk.LEFT, padx=(0, 8), anchor=tk.N)
        lines = label.split("\n")
        for i, line in enumerate(lines):
            lbl = ttk.Label(lbl_frame, text=line, anchor=tk.W, font=("", 9 if i > 0 else 10))
            if i > 0:
                lbl.config(wraplength=320)
            lbl.pack(anchor=tk.W)
        val = default if default and default in values else (values[0] if values else "")
        c = ttk.Combobox(row, values=values, state="readonly", width=36)
        c.set(val)
        c.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entries[key] = c

    def _fill_options(self, api: str):
        self._clear_options()
        if api == "Gemini":
            self._add_row_combo(
                "Модел\nКой Gemini модел да ползва (flash = бърз, pro = по-качествен)",
                "model", GEMINI_MODELS, "gemini-2.0-flash"
            )
            self._add_row(
                "Temperature (0–1)\nСлучайност: 0 = фиксиран отговор, 1 = по-разнообразно",
                "temperature", "0.7", 10
            )
            self._add_row(
                "Top P (0–1)\nЯдро на вероятности: колко от най-вероятните токени да се ползват",
                "top_p", "0.95", 10
            )
            self._add_row(
                "Top K (цяло число)\nМакс. брой токени при избор на следващ (сэмплиране)",
                "top_k", "40", 10
            )
            self._add_row(
                "Max output tokens\nМаксимален брой токени в отговора",
                "max_output_tokens", "2048", 10
            )
            self._add_row(
                "Stop sequences\nТекстове, при които да спре генерирането (разделени със запетая)",
                "stop_sequences", "", 35
            )
            self._add_row(
                "Response MIME type\nПразно = обикновен текст; application/json за JSON",
                "response_mime_type", "", 25
            )
        elif api == "OpenAI":
            self._add_row_combo(
                "Модел\nКой OpenAI модел да ползва (gpt-4o, gpt-4o-mini, …)",
                "model", OPENAI_MODELS, "gpt-4o-mini"
            )
            self._add_row(
                "Temperature (0–2)\nСлучайност на отговора; 0 = детерминистично",
                "temperature", "0.7", 10
            )
            self._add_row(
                "Max tokens\nМаксимален брой токени в отговора",
                "max_tokens", "1024", 10
            )
            self._add_row(
                "Top P (0–1)\nАлтернатива на temperature за сэмплиране",
                "top_p", "1.0", 10
            )
            self._add_row(
                "Frequency penalty (-2–2)\nНамалява повторенията на токени в отговора",
                "frequency_penalty", "0.0", 10
            )
            self._add_row(
                "Presence penalty (-2–2)\nСтимулира нови теми",
                "presence_penalty", "0.0", 10
            )
            self._add_row(
                "Stop\nЕдин или няколко стринга (със запетая), при които да спре",
                "stop", "", 28
            )
            self._add_row(
                "Seed (цяло число)\nЗа възпроизводимост; празно = не се ползва",
                "seed", "", 10
            )
        else:  # Anthropic
            self._add_row_combo(
                "Модел\nКой Claude модел да ползва (sonnet, opus, haiku)",
                "model", ANTHROPIC_MODELS, "claude-sonnet-4-20250514"
            )
            self._add_row(
                "Max tokens\nМаксимален брой токени в отговора",
                "max_tokens", "1024", 10
            )
            self._add_row(
                "Temperature (0–1)\nСлучайност; празно = по подразбиране от API",
                "temperature", "", 10
            )
            self._add_row(
                "Top P (0–1)\nЯдро на вероятности; празно = по подразбиране",
                "top_p", "", 10
            )
            self._add_row(
                "Top K (цяло число)\nСэмплиране; празно = по подразбиране",
                "top_k", "", 10
            )
            self._add_row(
                "Stop sequences\nТекстове за спиране (разделени със запетая)",
                "stop_sequences", "", 35
            )

    def _on_api_change(self, event=None):
        self._fill_options(self.api_var.get())

    def _get_opt(self, key: str, default=None):
        if key not in self.entries:
            return default
        v = self.entries[key].get().strip()
        return v if v else default

    def _get_float(self, key: str, default=None):
        v = self._get_opt(key)
        if v is None or v == "":
            return default
        try:
            return float(v)
        except ValueError:
            return default

    def _get_int(self, key: str, default=None):
        v = self._get_opt(key)
        if v is None or v == "":
            return default
        try:
            return int(v)
        except ValueError:
            return default

    def _update_request_display(self, payload: dict):
        self.request_text.config(state=tk.NORMAL)
        self.request_text.delete(1.0, tk.END)
        self.request_text.insert(tk.END, _format_request(self.api_var.get(), payload))
        self.request_text.config(state=tk.DISABLED)

    def _update_response_display(self, text: str):
        self.response_text.config(state=tk.NORMAL)
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, text)
        self.response_text.config(state=tk.DISABLED)

    def _send(self):
        api = self.api_var.get()
        prompt = self.prompt_text.get(1.0, tk.END).strip()
        system = self.system_text.get(1.0, tk.END).strip() or None
        if not prompt:
            messagebox.showwarning("Грешка", "Въведи съобщение (prompt).")
            return
        stream = self.stream_var.get()

        def run():
            try:
                if api == "Gemini":
                    self._do_gemini(prompt, system, stream)
                elif api == "OpenAI":
                    self._do_openai(prompt, system, stream)
                else:
                    self._do_anthropic(prompt, system, stream)
            except Exception as exc:
                err_msg = f"Грешка:\n{type(exc).__name__}: {exc}"
                self.win.after(0, lambda msg=err_msg: self._update_response_display(msg))

        threading.Thread(target=run, daemon=True).start()

    def _do_gemini(self, prompt: str, system: str | None, stream: bool):
        mod = _import_gemini()
        model = self._get_opt("model", "gemini-2.0-flash")
        temperature = self._get_float("temperature", 0.7)
        top_p = self._get_float("top_p", 0.95)
        top_k = self._get_int("top_k", 40)
        max_out = self._get_int("max_output_tokens", 2048)
        stop_raw = self._get_opt("stop_sequences")
        stop_sequences = [s.strip() for s in stop_raw.split(",") if s.strip()] if stop_raw else None
        response_mime = self._get_opt("response_mime_type") or None

        request_payload = {
            "model": model,
            "contents": prompt,
            "system_instruction": system,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_out,
            "stream": stream,
        }
        if stop_sequences:
            request_payload["stop_sequences"] = stop_sequences
        if response_mime:
            request_payload["response_mime_type"] = response_mime

        self.win.after(0, lambda: self._update_request_display(request_payload))

        if stream:
            out = []
            for chunk in mod.generate_content_stream(
                prompt, model=model, system_instruction=system,
                temperature=temperature, max_output_tokens=max_out,
            ):
                out.append(chunk)
            result = "".join(out)
        else:
            result = mod.generate_content(
                prompt, model=model, system_instruction=system,
                temperature=temperature, top_p=top_p, top_k=top_k,
                max_output_tokens=max_out, stop_sequences=stop_sequences,
                response_mime_type=response_mime,
            )
        self.win.after(0, lambda: self._update_response_display(result))

    def _do_openai(self, prompt: str, system: str | None, stream: bool):
        mod = _import_openai()
        model = self._get_opt("model", "gpt-4o-mini")
        temperature = self._get_float("temperature", 0.7)
        max_tokens = self._get_int("max_tokens", 1024)
        top_p = self._get_float("top_p", 1.0)
        freq = self._get_float("frequency_penalty", 0.0)
        pres = self._get_float("presence_penalty", 0.0)
        stop_raw = self._get_opt("stop")
        stop = [s.strip() for s in stop_raw.split(",") if s.strip()] if stop_raw else None
        if stop and len(stop) == 1:
            stop = stop[0]
        seed_raw = self._get_opt("seed")
        seed = int(seed_raw) if seed_raw and seed_raw.strip() else None

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        request_payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": freq,
            "presence_penalty": pres,
            "stream": stream,
        }
        if stop is not None:
            request_payload["stop"] = stop
        if seed is not None:
            request_payload["seed"] = seed

        self.win.after(0, lambda: self._update_request_display(request_payload))

        if stream:
            out = []
            for chunk in mod.chat_stream(
                prompt, model=model, system=system,
                temperature=temperature, max_tokens=max_tokens,
            ):
                out.append(chunk)
            result = "".join(out)
        else:
            r = mod.chat_completion(
                messages, model=model, temperature=temperature,
                max_tokens=max_tokens, top_p=top_p,
                frequency_penalty=freq, presence_penalty=pres,
                stop=stop, seed=seed,
            )
            raw_display = _format_response(r)
            self.win.after(0, lambda: self._update_response_display(raw_display))
            return
        self.win.after(0, lambda: self._update_response_display(result))

    def _do_anthropic(self, prompt: str, system: str | None, stream: bool):
        mod = _import_anthropic()
        model = self._get_opt("model", "claude-sonnet-4-20250514")
        max_tokens = self._get_int("max_tokens", 1024)
        temperature = self._get_float("temperature")
        top_p = self._get_float("top_p")
        top_k_raw = self._get_opt("top_k")
        top_k = int(top_k_raw) if top_k_raw and top_k_raw.strip() else None
        stop_raw = self._get_opt("stop_sequences")
        stop_sequences = [s.strip() for s in stop_raw.split(",") if s.strip()] if stop_raw else None

        request_payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "system": system,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if temperature is not None:
            request_payload["temperature"] = temperature
        if top_p is not None:
            request_payload["top_p"] = top_p
        if top_k is not None:
            request_payload["top_k"] = top_k
        if stop_sequences:
            request_payload["stop_sequences"] = stop_sequences

        self.win.after(0, lambda: self._update_request_display(request_payload))

        if stream:
            out = []
            for chunk in mod.message_stream(
                prompt, model=model, system=system,
                max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, stop_sequences=stop_sequences,
            ):
                out.append(chunk)
            result = "".join(out)
            self.win.after(0, lambda: self._update_response_display(result))
            return
        r = mod.message_create(
            [{"role": "user", "content": prompt}],
            model=model, system=system, max_tokens=max_tokens,
            temperature=temperature, top_p=top_p, top_k=top_k,
            stop_sequences=stop_sequences,
        )
        text = next((b.text for b in r.content if hasattr(b, "text") and b.text), "") if r.content else ""
        raw_display = _format_response(r)
        self.win.after(0, lambda: self._update_response_display(raw_display))

    def run(self):
        self.win.mainloop()


if __name__ == "__main__":
    app = ApiGui()
    app.run()
