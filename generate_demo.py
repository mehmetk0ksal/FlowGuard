import pandas as pd
import numpy as np
import random
import os

def generate_synthetic_demo_data(num_samples=30):
    # Exact column order expected by the model from JSON
    features = [
        "Dst Port", "Bwd Packets/s", "Fwd Seg Size Min", "Flow IAT Mean", 
        "Down/Up Ratio", "FWD Init Win Bytes", "Flow IAT Max", "Flow Packets/s", 
        "Packet Length Max", "Src Port", "Flow Duration", "Bwd Init Win Bytes", 
        "Fwd Packet Length Max", "Bwd IAT Mean", "PSH Flag Count", 
        "Fwd Packet Length Mean", "Packet Length Mean", "Flow Bytes/s", 
        "Bwd Packet Length Max", "Total Bwd packets", "FIN Flag Count", 
        "Bwd Packet Length Mean", "Total Fwd Packet", "Bwd IAT Total"
    ]
    
    # Distribution suitable for real-world (Total 30 rows)
    # Intentionally split into categories to test all types in the UI
    labels = (
        ['Benign'] * 20 + 
        ['DoS'] * 3 + 
        ['Exploits'] * 3 + 
        ['Reconnaissance'] * 2 + 
        ['Generic'] * 2
    )
    
    random.shuffle(labels) # Shuffle so the model doesn't memorize the order
    
    data = []
    
    for label in labels:
        row = {}
        
        # Assign logical (but synthetic) values based on the label
        if label == 'Benign':
            # Normal web traffic usually goes to port 80 or 443
            row["Dst Port"] = random.choice([80, 443, 8080])
            row["Src Port"] = random.randint(1024, 65535)
            row["Flow Duration"] = random.randint(100, 50000)
            row["Total Fwd Packet"] = random.randint(1, 20)
            row["Flow Packets/s"] = random.uniform(10.0, 500.0)
            row["Packet Length Max"] = random.randint(40, 1500)
        
        elif label == 'DoS':
            # In a DoS attack, packets per second are at crazy levels
            row["Dst Port"] = 80
            row["Src Port"] = random.randint(1024, 65535)
            row["Flow Duration"] = random.randint(10, 1000)
            row["Total Fwd Packet"] = random.randint(1000, 5000) # Massive packets
            row["Flow Packets/s"] = random.uniform(5000.0, 20000.0)
            row["Packet Length Max"] = random.randint(1000, 1500)
            
        elif label in ['Exploits', 'Generic', 'Reconnaissance']:
            # Other attacks may show weird ports, abnormal packet sizes
            row["Dst Port"] = random.choice([21, 22, 23, 445, 3389])
            row["Src Port"] = random.randint(1024, 65535)
            row["Flow Duration"] = random.randint(50000, 1000000)
            row["Total Fwd Packet"] = random.randint(5, 50)
            row["Flow Packets/s"] = random.uniform(0.1, 50.0)
            row["Packet Length Max"] = random.randint(40, 500)

        # Fill remaining features randomly (numeric to prevent model errors)
        for feature in features:
            if feature not in row:
                if 'Count' in feature or 'Ratio' in feature or 'Size' in feature:
                    row[feature] = random.randint(0, 5)
                else:
                    row[feature] = random.uniform(0.1, 1000.0)
        
        # Add true label for testing purposes
        row["Label"] = label
        
        # Add to list according to JSON order + Label
        ordered_row = [row[feat] for feat in features] + [row["Label"]]
        data.append(ordered_row)

    # Create DataFrame and save as CSV
    df = pd.DataFrame(data, columns=features + ["Label"])
    
    # Round decimals for cleaner display
    df = df.round(4)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, "demo_traffic.csv")
    df.to_csv(output_path, index=False)
    
    print(f"30-row synthetic demo data created successfully: {output_path}")
    print("Class Distribution:")
    print(df['Label'].value_counts())

if __name__ == "__main__":
    generate_synthetic_demo_data(30)