import torch
print("\n" + "═"*50)
print(" 🚀 STARTING ULTIMATE HEALTH CHECK 🚀")
print("═"*50)

try:
    print(f"[1/4] PyTorch: v{torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if not torch.cuda.is_available(): raise Exception("CUDA IS NOT AVAILABLE!")
    
    print("[2/4] Testing Whisper...")
    import whisper
    w = whisper.load_model("tiny", device="cuda")
    
    print("[3/4] Testing WavLM...")
    from transformers import WavLMModel
    wlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to("cuda")
    
    print("[4/4] Testing XLM-Roberta (SentencePiece)...")
    from transformers import XLMRobertaTokenizer, XLMRobertaModel
    tok = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    xlm = XLMRobertaModel.from_pretrained("xlm-roberta-base").to("cuda")

    print("\n✅ SUCCESS! ALL MODELS LOADED PERFECTLY! ✅")
    print("You are 100% ready to run your batch script!")
    print("═"*50 + "\n")
except Exception as e:
    import traceback
    print(f"\n❌ FATAL ERROR: {e}\n")
    traceback.print_exc()
