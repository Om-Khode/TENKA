# Custom Wake Word Models

Place your custom openWakeWord model files (.onnx) here.

## Training a Custom Wake Word

openWakeWord makes it easy to train custom wake words — completely free, no API key!

### Steps:

1. **Open the training notebook** (runs in your browser, no local setup):
   https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb

2. **Set the target phrases** to match your assistant's name. For the default
   `ASSISTANT_NAME=TENKA`, that's:

   ```
   ["tenka", "ten-ka", "Tenka"]
   ```

   If you renamed the assistant via `ASSISTANT_NAME=Luna` (or any other name),
   use that name's pronunciations instead.

3. **Run all cells** in the notebook — takes ~30-60 minutes

4. **Download the trained model** (.onnx file)

5. **Rename it to `{assistant_name_lower}.onnx`** and place it in this folder.
   For the default name:

   ```
   assistant/models/tenka.onnx
   ```

6. **Restart the assistant** — it will auto-detect the custom model!

## Until You Have a Custom Model

The assistant uses the built-in **"hey jarvis"** wake word as a fallback.
Just say "Hey Jarvis" — no extra setup needed.

You can change the fallback in `config.py` → `WAKE_WORD_BUILTIN`.
