#!/usr/bin/env python3
"""
Test semplice dei pattern regex.
"""
import re

def test_regex_only():
    """Test solo dei pattern regex."""
    print("Test dei pattern regex:")
    
    # Test delle righe reali contro i pattern regex
    test_lines = [
        "INFO :      [ROUND 1]",
        "[Server] Round 1 aggregate fit -> accuracy=0.2100, loss=2.1999, precision=0.1897, recall=0.2037, f1=0.1860",
        "[Server] Round 1 evaluate -> accuracy=0.4400, loss=1.7303, precision=0.4654, recall=0.4481, f1=0.3904",
        "[Client 0] fit | Received parameters, config: {'round': 1}",
        "[Client 0] fit complete | avg_loss=2.1999, accuracy=0.2100, precision=0.1897, recall=0.2037, f1=0.1860",
        "[Client 0] evaluate | Received parameters, config: {'round': 1}",
        "[Client 0] evaluate complete | avg_loss=1.7303, accuracy=0.4400, precision=0.4654, recall=0.4481, f1=0.3904",
    ]
    
    patterns = {
        "round": r"\[ROUND (\d+)\]",
        "server_fit": r"\[Server\] Round (\d+) aggregate fit -> (.+)",
        "server_eval": r"\[Server\] Round (\d+) evaluate -> (.+)",
        "client_fit": r"\[Client (\d+)\] fit complete \| (.+)",
        "client_eval": r"\[Client (\d+)\] evaluate complete \| (.+)",
        "client_config": r"\[Client (\d+)\] (?:fit|evaluate) \| Received parameters, config: \{'round': (\d+)\}",
        "metrics": r"(\w+)=([\d\.]+)"
    }
    
    total_matches = 0
    
    for line in test_lines:
        print(f"\nTestando: {line}")
        line_matches = 0
        
        for pattern_name, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                print(f"  ✅ {pattern_name}: {match.groups()}")
                line_matches += 1
                total_matches += 1
                
                # Test specifico per metrics extraction
                if pattern_name in ["server_fit", "server_eval", "client_fit", "client_eval"]:
                    if len(match.groups()) > 1:
                        metrics_str = match.groups()[-1]
                        metric_matches = re.findall(patterns["metrics"], metrics_str)
                        print(f"     Metriche estratte: {metric_matches}")
                        total_matches += len(metric_matches)
        
        if line_matches == 0:
            print("  ❌ Nessun pattern corrispondente")
    
    print(f"\n\nTotale corrispondenze trovate: {total_matches}")
    
    if total_matches > 0:
        print("✅ I pattern regex funzionano correttamente!")
        return True
    else:
        print("❌ I pattern regex non funzionano!")
        return False

if __name__ == "__main__":
    test_regex_only()
