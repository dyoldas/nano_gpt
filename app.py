import argparse
import threading
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from chat_engine import ChatEngine


class NanoGPTApp:
    """Native desktop chat window for your local nanoGPT model.

    This uses Tkinter, which creates a real desktop window instead of
    opening a browser tab like Gradio.

    The UI calls ChatEngine, and ChatEngine handles:
        - model loading
        - prompt formatting
        - generation
        - response cleanup
    """

    def __init__(self, root, engine):
        self.root = root
        self.engine = engine

        # Chat history uses message dictionaries:
        #
        # [
        #     {"role": "user", "content": "..."},
        #     {"role": "assistant", "content": "..."},
        # ]
        self.history = []

        self.root.title("nanoGPT Chat")
        self.root.geometry("800x700")

        self.build_ui()

    def build_ui(self):
        """Create all UI elements."""

        # Main frame gives padding around the whole app.
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            main,
            text="nanoGPT Chat",
            font=("Arial", 18, "bold"),
        )
        title.pack(anchor="w", pady=(0, 10))

        # Chat display.
        #
        # ScrolledText gives us a large text box with a scrollbar.
        self.chat_box = ScrolledText(
            main,
            wrap=tk.WORD,
            height=25,
            state=tk.DISABLED,
        )
        self.chat_box.pack(fill=tk.BOTH, expand=True)

        # Input label.
        ttk.Label(main, text="Message:").pack(anchor="w", pady=(10, 0))

        # User input box.
        self.input_box = tk.Text(
            main,
            height=4,
            wrap=tk.WORD,
        )
        self.input_box.pack(fill=tk.X, pady=(0, 10))

        # Ctrl+Enter sends message.
        self.input_box.bind("<Control-Return>", lambda event: self.send_message())

        # Generation controls.
        controls = ttk.Frame(main)
        controls.pack(fill=tk.X, pady=(0, 10))

        # Temperature controls randomness.
        self.temperature = tk.DoubleVar(value=0.7)

        ttk.Label(controls, text="Temperature").grid(row=0, column=0, sticky="w")
        ttk.Scale(
            controls,
            from_=0.1,
            to=1.5,
            variable=self.temperature,
            orient=tk.HORIZONTAL,
        ).grid(row=0, column=1, sticky="ew", padx=10)

        self.temperature_label = ttk.Label(controls, text="0.70")
        self.temperature_label.grid(row=0, column=2, sticky="e")

        # Update visible temperature number when slider moves.
        self.temperature.trace_add("write", self.update_temperature_label)

        # top_k restricts sampling to the top-k likely tokens.
        self.top_k = tk.IntVar(value=30)

        ttk.Label(controls, text="Top-k").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(
            controls,
            from_=0,
            to=100,
            textvariable=self.top_k,
            width=8,
        ).grid(row=1, column=1, sticky="w", padx=10)

        # max_new_tokens controls response length.
        self.max_new_tokens = tk.IntVar(value=200)

        ttk.Label(controls, text="Max tokens").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(
            controls,
            from_=50,
            to=1000,
            increment=50,
            textvariable=self.max_new_tokens,
            width=8,
        ).grid(row=2, column=1, sticky="w", padx=10)

        controls.columnconfigure(1, weight=1)

        # Buttons.
        buttons = ttk.Frame(main)
        buttons.pack(fill=tk.X)

        self.send_button = ttk.Button(
            buttons,
            text="Send",
            command=self.send_message,
        )
        self.send_button.pack(side=tk.LEFT)

        self.clear_button = ttk.Button(
            buttons,
            text="Clear",
            command=self.clear_chat,
        )
        self.clear_button.pack(side=tk.LEFT, padx=10)

        hint = ttk.Label(
            buttons,
            text="Shortcut: Ctrl+Enter to send",
        )
        hint.pack(side=tk.RIGHT)

    def update_temperature_label(self, *_):
        """Update the small temperature number next to the slider."""

        value = self.temperature.get()
        self.temperature_label.config(text=f"{value:.2f}")

    def append_chat(self, speaker, text):
        """Append one message to the chat display."""

        self.chat_box.config(state=tk.NORMAL)

        if speaker == "You":
            self.chat_box.insert(tk.END, f"\nYou:\n{text}\n")
        else:
            self.chat_box.insert(tk.END, f"\nnanoGPT:\n{text}\n")

        self.chat_box.insert(tk.END, "-" * 80 + "\n")

        self.chat_box.config(state=tk.DISABLED)

        # Auto-scroll to bottom.
        self.chat_box.see(tk.END)

    def set_busy(self, busy):
        """Disable buttons while the model is generating."""

        state = tk.DISABLED if busy else tk.NORMAL
        self.send_button.config(state=state)
        self.clear_button.config(state=state)

    def send_message(self):
        """Read user input and start generation."""

        user_message = self.input_box.get("1.0", tk.END).strip()

        if user_message == "":
            return

        # Clear input immediately.
        self.input_box.delete("1.0", tk.END)

        # Show user message immediately.
        self.append_chat("You", user_message)

        # Disable buttons while generation runs.
        self.set_busy(True)

        # Run generation in a background thread.
        #
        # Without this, Tkinter would freeze until the model finishes.
        thread = threading.Thread(
            target=self.generate_in_background,
            args=(user_message,),
            daemon=True,
        )
        thread.start()

    def generate_in_background(self, user_message):
        """Generate response without freezing the UI."""

        try:
            updated_history = self.engine.respond(
                user_message=user_message,
                history=self.history,
                temperature=float(self.temperature.get()),
                top_k=int(self.top_k.get()),
                max_new_tokens=int(self.max_new_tokens.get()),
            )

            self.history = updated_history

            # Last assistant message is at the end.
            assistant_message = self.history[-1]["content"]

        except Exception as e:
            assistant_message = f"[error: {e}]"

        # Tkinter UI updates must happen on the main thread.
        self.root.after(
            0,
            lambda: self.finish_generation(assistant_message),
        )

    def finish_generation(self, assistant_message):
        """Display generated assistant message and re-enable UI."""

        self.append_chat("nanoGPT", assistant_message)
        self.set_busy(False)

    def clear_chat(self):
        """Clear chat window and reset history."""

        self.history = []

        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.delete("1.0", tk.END)
        self.chat_box.config(state=tk.DISABLED)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/instruct_mix/ckpt.pt",
        help="Path to model checkpoint.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, mps, cuda, or cpu.",
    )

    args = parser.parse_args()

    # Load model once at startup.
    engine = ChatEngine(
        ckpt_path=args.ckpt,
        device=args.device,
    )

    # Create native desktop window.
    root = tk.Tk()
    NanoGPTApp(root, engine)

    # Start Tkinter event loop.
    root.mainloop()


if __name__ == "__main__":
    main()
