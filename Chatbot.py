import tkinter as tk
from tkinter import messagebox, scrolledtext
import subprocess
import sys

def install_dependencies():
    """Install required dependencies if missing"""
    try:
        import torch
        import transformers
    except ImportError:
        if messagebox.askyesno(
            "Dependencies Missing",
            "Required packages (PyTorch & Transformers) are missing. Install now? (Internet connection required)"
        ):
            try:
                # Install PyTorch (CPU version for wider compatibility)
                subprocess.check_call([
                    sys.executable, 
                    "-m", 
                    "pip", 
                    "install", 
                    "torch", 
                    "torchvision", 
                    "torchaudio"
                ])
                
                # Install Transformers
                subprocess.check_call([
                    sys.executable, 
                    "-m", 
                    "pip", 
                    "install", 
                    "transformers"
                ])

                messagebox.showinfo(
                    "Success", 
                    "Libraries installed successfully!\nRestart the application."
                )
                sys.exit(0)
            except Exception as e:
                messagebox.showerror(
                    "Installation Failed", 
                    f"Error installing dependencies:\n{str(e)}"
                )
                sys.exit(1)

install_dependencies()

# Main Chatbot Application (after ensuring dependencies are installed)
class ChatbotApp:
    def __init__(self, root):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        self.root = root
        self.root.title("🤖 AI Chatbot")
        self.root.geometry("650x700")
        
        # Model configuration
        self.model_name = "microsoft/DialoGPT-medium"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.chat_history_ids = None
        self.max_history_length = 1024  # Prevents memory overload
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Configure the user interface"""
        # Chat Display
        self.chat_display = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            state="disabled",
            font=("Arial", 11),
            bg="#f0f0f0",
            padx=10,
            pady=10,
            width=60,
            height=20
        )
        self.chat_display.pack(pady=10, padx=10, fill="both", expand=True)
        
        # Input Frame
        input_frame = tk.Frame(self.root)
        input_frame.pack(pady=10, fill="x", padx=10)
        
        self.user_input = tk.Entry(
            input_frame,
            font=("Arial", 12),
            relief="flat",
            highlightthickness=1,
            highlightcolor="#4CAF50"
        )
        self.user_input.pack(side="left", fill="x", expand=True, ipady=8)
        self.user_input.bind("<Return>", self.send_message)
        self.user_input.focus()
        
        # Send Button
        send_button = tk.Button(
            input_frame,
            text="Send",
            command=self.send_message,
            bg="#4CAF50",
            fg="white",
            relief="flat",
            font=("Arial", 10, "bold")
        )
        send_button.pack(side="right", padx=(10, 0))
        
        # Status Bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            bd=1,
            relief="sunken",
            anchor="w"
        )
        self.status_bar.pack(fill="x")
        
        # Display welcome message
        self.update_chat("AI", "Hello! How can I help you today?")
    
    def send_message(self, event=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        user_msg = self.user_input.get().strip()
        if not user_msg:
            return
            
        self.user_input.delete(0, "end")
        self.update_chat("You", user_msg)
        
        try:
            # Tokenize input and update history
            new_input = self.tokenizer.encode(
                user_msg + self.tokenizer.eos_token,
                return_tensors="pt"
            )
            
            if self.chat_history_ids is not None:
                input_ids = torch.cat([self.chat_history_ids, new_input], dim=-1)
            else:
                input_ids = new_input
            
            # Generate response
            response = self.model.generate(
                input_ids,
                max_length=self.max_history_length,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.7,
                repetition_penalty=1.2
            )
            
            # Extract bot response (exclude input)
            response_start = input_ids.shape[-1]
            bot_response = self.tokenizer.decode(
                response[:, response_start:][0],
                skip_special_tokens=True
            )
            
            # Update history
            self.chat_history_ids = response
            
            # Enforce history length limit
            if self.chat_history_ids.shape[-1] > self.max_history_length:
                self.chat_history_ids = self.chat_history_ids[:, -self.max_history_length//2:]
                
            self.update_chat("AI", bot_response)
            
        except Exception as e:
            self.update_chat("System", f"Error: {str(e)}")
    
    def update_chat(self, sender, message):
        """Update chat display with colored sender labels"""
        tags = {
            "You": ("you", "#2c3e50"),
            "AI": ("ai", "#2980b9"),
            "System": ("sys", "#e74c3c")
        }
        
        self.chat_display.config(state="normal")
        self.chat_display.insert(
            "end",
            f"{sender}: ",
            tags[sender][0]
        )
        self.chat_display.insert(
            "end",
            message + "\n\n",
            "msg"
        )
        
        # Configure tag colors
        self.chat_display.tag_config(
            tags[sender][0],
            foreground=tags[sender][1],
            font=("Arial", 10, "bold")
        )
        self.chat_display.tag_config("msg", font=("Arial", 10))
        
        self.chat_display.config(state="disabled")
        self.chat_display.see("end")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()
