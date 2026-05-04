import pandas as pd
import os

def create_real_demo():
    # File paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_file_path = os.path.join(base_dir, "data", "processed", "X_test_ham.csv")
    output_file_path = os.path.join(base_dir, "real_demo_traffic.csv")

    print(f"Reading real test data: {test_file_path}")
    df = pd.read_csv(test_file_path)

    # Pull 6 random samples from each label class (0, 1, 2, 3, 4) -> 5 * 6 = 30 rows
    sampled_dfs = []
    for label in df['Label'].unique():
        # Filter data for the class
        class_subset = df[df['Label'] == label]
        
        # Get exactly 6 samples, or fewer if the class doesn't have enough data
        n_samples = min(6, len(class_subset))
        sampled_dfs.append(class_subset.sample(n=n_samples, random_state=42))

    # Combine all subsets and shuffle rows
    demo_df = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert Label numbers back to text for elegant UI display
    REVERSE_DICT = {0: "Benign", 1: "Reconnaissance", 2: "Exploits", 3: "DoS", 4: "Generic"}
    demo_df['Label'] = demo_df['Label'].map(REVERSE_DICT)

    # Save as CSV
    demo_df.to_csv(output_file_path, index=False)
    
    print(f"\nSuccess! 30 rows of real network traffic demo data generated.")
    print(f"File saved to: {output_file_path}")
    print("\nClass Distribution of Generated Demo Data:")
    print(demo_df['Label'].value_counts())

if __name__ == "__main__":
    create_real_demo()