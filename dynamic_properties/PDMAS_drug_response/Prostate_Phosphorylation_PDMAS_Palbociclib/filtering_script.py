import pandas as pd
import os

input_file = "/home/saishyam/Protein_dynamics/Dynamic_properties/PDMAS_drug_response/PTMD_data_in_pmads.txt"
output_dir = "data/combined"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_file, sep='\t', header=None, encoding='latin1')
df['combined_text'] = df.apply(lambda row: ' '.join(row.astype(str)).lower(), axis=1)

# Strict filters
f1 = df['combined_text'].str.contains('prostate')
f2 = df['combined_text'].str.contains('phospho|phosphorylation')
f3 = df['combined_text'].str.contains('lncap')
f4 = df['combined_text'].str.contains('palbociclib|palboicib')

strict_df = df[f1 & f2 & f3 & f4]
fallback_df = df[f1 & f2 & f3]

# Remove temporary column before saving
def save_output(subset, name):
    if 'combined_text' in subset.columns:
        subset = subset.drop(columns=['combined_text'])
    path = os.path.join(output_dir, name)
    subset.to_csv(path, sep='\t', index=False, header=False)
    return path

if not strict_df.empty:
    strict_path = save_output(strict_df, "strict_combined.txt")
    print(f"Strict results saved to {strict_path} (Count: {len(strict_df)})")
    
    up_df = strict_df[strict_df['combined_text'].str.contains('up-regulated|upregulated')]
    down_df = strict_df[strict_df['combined_text'].str.contains('down-regulated|downregulated')]
    
    up_path = save_output(up_df, "strict_up_only.txt")
    down_path = save_output(down_df, "strict_down_only.txt")
    print(f"Up-only saved to {up_path} (Count: {len(up_df)})")
    print(f"Down-only saved to {down_path} (Count: {len(down_df)})")
else:
    print("Strict filter yielded no results. Saving fallback (prostate & phospho & lncap).")
    fallback_path = save_output(fallback_df, "fallback_combined.txt")
    print(f"Fallback results saved to {fallback_path} (Count: {len(fallback_df)})")

