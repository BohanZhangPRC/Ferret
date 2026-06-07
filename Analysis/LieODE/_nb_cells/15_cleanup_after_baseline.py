# --- GPU cleanup ---
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print('GPU memory cleared.')