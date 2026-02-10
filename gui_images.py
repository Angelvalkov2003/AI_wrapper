"""
GUI за генериране на изображения с OpenAI (DALL-E) и Gemini.
Показва точната заявка, суровия отговор и всички параметри.
Стартиране: python gui_images.py
"""
import base64
import io
import json
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = ImageTk = None  # ще покажем съобщение да инсталира pilllow

# Лениви импорти
def _openai_images():
    from openai_api import client as m
    return m

def _gemini_images():
    from gemini import client as m
    return m


# Модели и параметри
OPENAI_IMAGE_MODELS = ["dall-e-3", "dall-e-2"]
OPENAI_SIZES = ["1024x1024", "1792x1024", "1024x1792", "512x512", "256x256"]
OPENAI_QUALITY = ["standard", "hd"]  # само DALL-E 3
OPENAI_RESPONSE_FORMAT = ["b64_json", "url"]
GEMINI_IMAGE_MODELS = ["gemini-2.5-flash-image", "gemini-3-pro-image-preview"]
GEMINI_ASPECT_RATIOS = ["1:1", "9:16", "16:9", "4:3", "3:4", "2:3", "3:2", "4:5", "5:4", "21:9"]
GEMINI_IMAGE_SIZES = ["1K", "2K", "4K"]  # за 3-pro


class ImageGenGui:
    def __init__(self):
        self.win = tk.Tk()
        self.win.title("Генериране на изображения – OpenAI & Gemini")
        self.win.geometry("950x820")
        self.win.minsize(700, 600)

        self.api_var = tk.StringVar(value="OpenAI")
        self.current_photo = None
        self.current_image_bytes = None
        self.request_text = None
        self.response_text = None
        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.win, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Ред 1: API
        row0 = ttk.Frame(main)
        row0.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row0, text="Провайдър:").pack(side=tk.LEFT, padx=(0, 6))
        api_combo = ttk.Combobox(
            row0, textvariable=self.api_var,
            values=["OpenAI", "Gemini"],
            state="readonly", width=12
        )
        api_combo.pack(side=tk.LEFT)
        api_combo.bind("<<ComboboxSelected>>", self._on_api_change)

        # Опции според API (всички възможни параметри)
        self.opts_frame = ttk.LabelFrame(main, text="Параметри (всички за endpoint-а)", padding=6)
        self.opts_frame.pack(fill=tk.X, pady=(0, 8))
        self._fill_opts("OpenAI")

        # Описание (prompt)
        ttk.Label(main, text="Описание на изображението (prompt):").pack(anchor=tk.W)
        self.prompt_text = tk.Text(main, height=2, wrap=tk.WORD, font=("Segoe UI", 10))
        self.prompt_text.pack(fill=tk.X, pady=(0, 8))

        # Бутони
        btn_row = ttk.Frame(main)
        btn_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(btn_row, text="Генерирай изображение", command=self._generate).pack(side=tk.LEFT, padx=(0, 8))
        self.save_btn = ttk.Button(btn_row, text="Запази като...", command=self._save_image, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT)

        # Заявка / Отговор (малки полета)
        paned = ttk.PanedWindow(main, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        req_f = ttk.LabelFrame(paned, text="Точна заявка (request) до endpoint-а", padding=4)
        self.request_text = scrolledtext.ScrolledText(req_f, height=5, wrap=tk.WORD, font=("Consolas", 9), state=tk.DISABLED)
        self.request_text.pack(fill=tk.BOTH, expand=True)
        paned.add(req_f, weight=1)
        res_f = ttk.LabelFrame(paned, text="Суров отговор (response) от API", padding=4)
        self.response_text = scrolledtext.ScrolledText(res_f, height=4, wrap=tk.WORD, font=("Consolas", 9), state=tk.DISABLED)
        self.response_text.pack(fill=tk.BOTH, expand=True)
        paned.add(res_f, weight=1)

        # Област за изображение
        img_frame = ttk.LabelFrame(main, text="Генерирано изображение", padding=6)
        img_frame.pack(fill=tk.BOTH, expand=True)
        self.img_label = ttk.Label(img_frame, text="Тук ще се покаже изображението.", anchor=tk.CENTER)
        self.img_label.pack(fill=tk.BOTH, expand=True)
        self.status_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.status_var, font=("", 9)).pack(anchor=tk.W)

    def _clear_opts(self):
        for w in self.opts_frame.winfo_children():
            w.destroy()
        self.opt_widgets = {}

    def _fill_opts(self, api: str):
        self._clear_opts()
        self.opt_widgets = {}
        if api == "OpenAI":
            r1 = ttk.Frame(self.opts_frame)
            r1.pack(fill=tk.X, pady=2)
            ttk.Label(r1, text="Модел:").pack(side=tk.LEFT, padx=(0, 6))
            c = ttk.Combobox(r1, values=OPENAI_IMAGE_MODELS, state="readonly", width=14)
            c.set("dall-e-3")
            c.pack(side=tk.LEFT, padx=(0, 16))
            self.opt_widgets["model"] = c
            ttk.Label(r1, text="Размер:").pack(side=tk.LEFT, padx=(0, 6))
            c2 = ttk.Combobox(r1, values=OPENAI_SIZES, state="readonly", width=12)
            c2.set("1024x1024")
            c2.pack(side=tk.LEFT, padx=(0, 16))
            self.opt_widgets["size"] = c2
            ttk.Label(r1, text="Качество:").pack(side=tk.LEFT, padx=(0, 6))
            cq = ttk.Combobox(r1, values=OPENAI_QUALITY, state="readonly", width=10)
            cq.set("standard")
            cq.pack(side=tk.LEFT, padx=(0, 16))
            self.opt_widgets["quality"] = cq
            ttk.Label(r1, text="Стил:").pack(side=tk.LEFT, padx=(0, 6))
            c3 = ttk.Combobox(r1, values=["vivid", "natural"], state="readonly", width=10)
            c3.set("vivid")
            c3.pack(side=tk.LEFT, padx=(0, 16))
            ttk.Label(r1, text="response_format:").pack(side=tk.LEFT, padx=(0, 6))
            cr = ttk.Combobox(r1, values=OPENAI_RESPONSE_FORMAT, state="readonly", width=10)
            cr.set("b64_json")
            cr.pack(side=tk.LEFT)
            self.opt_widgets["response_format"] = cr
        else:
            r1 = ttk.Frame(self.opts_frame)
            r1.pack(fill=tk.X, pady=2)
            ttk.Label(r1, text="Модел:").pack(side=tk.LEFT, padx=(0, 6))
            c = ttk.Combobox(r1, values=GEMINI_IMAGE_MODELS, state="readonly", width=28)
            c.set("gemini-2.5-flash-image")
            c.pack(side=tk.LEFT, padx=(0, 16))
            self.opt_widgets["model"] = c
            ttk.Label(r1, text="Aspect ratio:").pack(side=tk.LEFT, padx=(0, 6))
            c2 = ttk.Combobox(r1, values=GEMINI_ASPECT_RATIOS, state="readonly", width=8)
            c2.set("1:1")
            c2.pack(side=tk.LEFT)
            self.opt_widgets["aspect_ratio"] = c2
            r2 = ttk.Frame(self.opts_frame)
            r2.pack(fill=tk.X, pady=2)
            ttk.Label(r2, text="Image size (за 3-pro):").pack(side=tk.LEFT, padx=(0, 6))
            c3 = ttk.Combobox(r2, values=GEMINI_IMAGE_SIZES, state="readonly", width=6)
            c3.set("2K")
            c3.pack(side=tk.LEFT)
            self.opt_widgets["image_size"] = c3

    def _on_api_change(self, event=None):
        self._fill_opts(self.api_var.get())

    def _update_request_display(self, payload: dict, endpoint_hint: str = ""):
        """Попълва полето за точната заявка."""
        if not self.request_text:
            return
        self.request_text.config(state=tk.NORMAL)
        self.request_text.delete(1.0, tk.END)
        lines = []
        if endpoint_hint:
            lines.append(f"Endpoint: {endpoint_hint}\n")
        try:
            lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception:
            lines.append(str(payload))
        self.request_text.insert(tk.END, "".join(lines))
        self.request_text.config(state=tk.DISABLED)

    def _update_response_display(self, text: str):
        """Попълва полето за суровия отговор."""
        if not self.response_text:
            return
        self.response_text.config(state=tk.NORMAL)
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, text)
        self.response_text.config(state=tk.DISABLED)

    def _generate(self):
        prompt = self.prompt_text.get(1.0, tk.END).strip()
        if not prompt:
            messagebox.showwarning("Грешка", "Въведи описание на изображението.")
            return
        if Image is None:
            messagebox.showerror("Грешка", "Инсталирай Pillow: pip install Pillow")
            return
        self.status_var.set("Генериране...")
        self.save_btn.config(state=tk.DISABLED)
        self.img_label.config(image="", text="Генериране...")

        def run():
            try:
                api = self.api_var.get()
                if api == "OpenAI":
                    self._do_openai(prompt)
                else:
                    self._do_gemini(prompt)
            except Exception as e:
                err = f"Грешка: {type(e).__name__}: {e}"
                self.win.after(0, lambda: self._show_error(err))

        threading.Thread(target=run, daemon=True).start()

    def _do_openai(self, prompt: str):
        mod = _openai_images()
        model = self.opt_widgets.get("model")
        model = model.get() if model else "dall-e-3"
        size = self.opt_widgets.get("size")
        size = size.get() if size else "1024x1024"
        style = self.opt_widgets.get("style")
        style = style.get() if style else "vivid"
        qw = self.opt_widgets.get("quality")
        quality = qw.get() if qw else "standard"
        rfw = self.opt_widgets.get("response_format")
        response_format = rfw.get() if rfw else "b64_json"
        params = dict(
            prompt=prompt,
            model=model,
            size=size,
            n=1,
            response_format=response_format,
        )
        if model == "dall-e-3":
            params["quality"] = quality
            params["style"] = style
        self.win.after(0, lambda: self._update_request_display(
            params, "POST images/generations (OpenAI)"
        ))
        r = mod.images_generate(**params)
        # Суров отговор: сериализираме, но b64_json съкращаваме
        try:
            raw = r.model_dump() if hasattr(r, "model_dump") else r.dict()
            for item in raw.get("data", []):
                if "b64_json" in item and item["b64_json"]:
                    item["b64_json"] = f"<base64, {len(item['b64_json'])} chars>"
            resp_str = json.dumps(raw, ensure_ascii=False, indent=2)
        except Exception:
            resp_str = str(r)
        self.win.after(0, lambda s=resp_str: self._update_response_display(s))
        # Показване на изображение: от b64 или от url
        first = r.data[0]
        if getattr(first, "b64_json", None):
            img_bytes = base64.b64decode(first.b64_json)
        elif getattr(first, "url", None):
            import urllib.request
            with urllib.request.urlopen(first.url) as resp:
                img_bytes = resp.read()
        else:
            img_bytes = None
        if img_bytes:
            self.win.after(0, lambda: self._show_image(img_bytes))
        else:
            self.win.after(0, lambda: self._show_error("Няма данни за изображение в отговора."))

    def _do_gemini(self, prompt: str):
        mod = _gemini_images()
        model_w = self.opt_widgets.get("model")
        model = model_w.get() if model_w else "gemini-2.5-flash-image"
        ar = self.opt_widgets.get("aspect_ratio")
        aspect_ratio = ar.get() if ar else "1:1"
        payload = {
            "model": model,
            "contents": prompt,
            "config": {
                "response_modalities": ["IMAGE"],
                "image_config": {"aspect_ratio": aspect_ratio},
            },
        }
        self.win.after(0, lambda: self._update_request_display(
            payload, "generate_content (Gemini)"
        ))
        img_bytes = mod.generate_image(prompt, model=model, aspect_ratio=aspect_ratio)
        resp_summary = json.dumps({
            "image_size_bytes": len(img_bytes),
            "model": model,
            "aspect_ratio": aspect_ratio,
            "note": "Пълният отговор съдържа binary image data; тук е обобщение.",
        }, ensure_ascii=False, indent=2)
        self.win.after(0, lambda s=resp_summary: self._update_response_display(s))
        self.win.after(0, lambda: self._show_image(img_bytes))

    def _show_image(self, img_bytes: bytes):
        self.current_image_bytes = img_bytes
        self.status_var.set("Готово.")
        self.save_btn.config(state=tk.NORMAL)
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img.thumbnail((600, 600), Image.Resampling.LANCZOS)
            self.current_photo = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.current_photo, text="")
        except Exception as e:
            self.img_label.config(text=f"Грешка при показ: {e}")
            self.current_photo = None

    def _show_error(self, msg: str):
        self.status_var.set("")
        self.save_btn.config(state=tk.DISABLED)
        self.img_label.config(image="", text=msg)
        self.current_image_bytes = None
        messagebox.showerror("Грешка", msg)

    def _save_image(self):
        if not self.current_image_bytes:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Всички", "*.*")]
        )
        if path:
            try:
                with open(path, "wb") as f:
                    f.write(self.current_image_bytes)
                self.status_var.set(f"Запазено: {path}")
                messagebox.showinfo("Готово", f"Изображението е запазено в:\n{path}")
            except Exception as e:
                messagebox.showerror("Грешка", str(e))

    def run(self):
        self.win.mainloop()


if __name__ == "__main__":
    app = ImageGenGui()
    app.run()
