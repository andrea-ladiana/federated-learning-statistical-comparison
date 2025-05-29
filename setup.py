"""
Script per installare le dipendenze necessarie per il progetto FL.
"""
import sys
import subprocess
import os

def main():
    print("Installazione delle dipendenze necessarie...")
    
    # Controlla se siamo in un ambiente virtuale
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        print("ATTENZIONE: Non sei in un ambiente virtuale Python!")
        proceed = input("Vuoi procedere comunque? (s/n): ")
        if proceed.lower() != 's':
            print("Installazione annullata.")
            print("\nConsiglio: Crea un ambiente virtuale prima di installare le dipendenze:")
            print("python -m venv venv")
            print("venv\\Scripts\\activate  # su Windows")
            print("source venv/bin/activate  # su Linux/Mac")
            return
    
    # Installa le dipendenze da requirements.txt
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\nDipendenze installate con successo!")
        
        # Verifica l'installazione
        print("\nVerifica dell'installazione:")
        try:
            import flwr
            print(f"✓ Flower versione: {flwr.__version__}")
            import torch
            print(f"✓ PyTorch versione: {torch.__version__}")
            import torchvision
            print(f"✓ TorchVision versione: {torchvision.__version__}")
            print("\nInstallazione completata con successo! Il sistema è pronto per eseguire esperimenti FL.")
        except ImportError as e:
            print(f"⚠ Errore nel verificare l'installazione: {e}")
            print("Alcune dipendenze potrebbero non essere state installate correttamente.")
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'installazione: {e}")
        print("Controlla la connessione internet e i permessi di installazione.")

if __name__ == "__main__":
    main()
