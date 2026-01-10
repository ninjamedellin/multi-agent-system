import os
import whisper
from pathlib import Path
from tempfile import NamedTemporaryFile

try:
    # 1. Rutas
    audio_path = "test_assets/audio.mp3"
    full_path = Path(audio_path).resolve()
    
    # 2. Leer bytes y crear temporal
    with open(full_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    with NamedTemporaryFile(suffix=full_path.suffix, delete=False) as temp_f:
        temp_f.write(audio_bytes)
        temp_path = temp_f.name
        
    print(f"Archivo temporal creado: {temp_path}")

    # 3. Ejecutar Transcripción (Punto Crítico)
    model = whisper.load_model("base")
    output = model.transcribe(temp_path)["text"].strip()
    
    print("\n✅ --- TRANSCRIPCIÓN EXITOSA --- ✅")
    print(output)

except Exception as e:
    print("\n❌ --- FALLO CRÍTICO DE WHISPER --- ❌")
    print(f"Error: {e}")

finally:
    # 4. Limpieza
    if 'temp_path' in locals() and os.path.exists(temp_path):
        os.remove(temp_path)
        print(f"Limpieza finalizada.")