import os
import json
import pandas as pd
import numpy as np
from cleaner import clean_ocr_text, deduplicate_resumes
from extractor import extract_structured_fields
from synthesize_jd import generate_synthetic_jds

def load_jsonl_to_df(filepath: str) -> pd.DataFrame:
    """Loads a JSONL file into a Pandas DataFrame."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)

def main():
    raw_dataset_path = "c:/Users/bsais/OneDrive/Desktop/celabal/project/data/raw/resumes_dataset.jsonl"
    processed_parquet_path = "c:/Users/bsais/OneDrive/Desktop/celabal/project/data/processed/processed_resumes.parquet"
    jds_output_path = "c:/Users/bsais/OneDrive/Desktop/celabal/project/data/processed/job_descriptions.jsonl"
    
    print("=" * 60)
    print("Starting ResumeIQ Preprocessing & EDA Pipeline")
    print("=" * 60)
    
    # 0. Synthesize Job Descriptions
    print("\n[Step 0] Generating synthetic job descriptions...")
    generate_synthetic_jds(jds_output_path)
    
    # 1. Load Raw Resumes
    print(f"\n[Step 1] Loading raw dataset from: {raw_dataset_path}")
    if not os.path.exists(raw_dataset_path):
        raise FileNotFoundError(f"Raw resume dataset not found at {raw_dataset_path}")
        
    df = load_jsonl_to_df(raw_dataset_path)
    total_raw_count = len(df)
    print(f"Loaded {total_raw_count} raw resumes.")
    
    # 2. Profile Field Completeness (Before Cleaning)
    print("\n[Step 2] Profiling field completeness (Before cleaning):")
    missing_report = df.isnull().sum()
    for col, missing in missing_report.items():
        completeness = ((total_raw_count - missing) / total_raw_count) * 100
        print(f"  Field: {col:<15} | Completeness: {completeness:6.2f}% | Missing: {missing}")
        
    # 3. Clean Text Fields
    print("\n[Step 3] Cleaning OCR text, stripping headers, and normalizing whitespace...")
    # Keep copies of original fields for manual comparison
    df['Original_Summary'] = df['Summary']
    df['Original_Experience'] = df['Experience']
    
    df['Summary'] = df['Summary'].fillna("").apply(clean_ocr_text)
    df['Experience'] = df['Experience'].fillna("").apply(clean_ocr_text)
    df['Education'] = df['Education'].fillna("").apply(clean_ocr_text)
    df['Text'] = df['Text'].fillna("").apply(clean_ocr_text)
    
    # 4. De-duplicate Near-Identical Resumes
    print("\n[Step 4] De-duplicating near-identical resumes (TF-IDF + Cosine Similarity > 0.95)...")
    df_dedup = deduplicate_resumes(df, text_col='Text', threshold=0.95)
    dedup_count = len(df_dedup)
    print(f"Removed {total_raw_count - dedup_count} near-duplicate resumes. Remaining: {dedup_count}")
    
    # 5. Extract Structured Fields
    print("\n[Step 5] Extracting structured features (Skills, Experience, Degree level)...")
    extracted_features = []
    for idx, row in df_dedup.iterrows():
        features = extract_structured_fields(row)
        extracted_features.append(features)
        
    features_df = pd.DataFrame(extracted_features)
    
    # Combine back
    df_processed = pd.concat([df_dedup, features_df], axis=1)
    
    # 6. Report Category Balance
    print("\n[Step 6] Category Balance / Class Distribution:")
    cat_counts = df_processed['Category'].value_counts()
    for cat, count in cat_counts.items():
        percentage = (count / dedup_count) * 100
        print(f"  Category: {cat:<30} | Count: {count:<4} | Percentage: {percentage:5.2f}%")
        
    # 7. Text Length Distributions
    print("\n[Step 7] Text Length Distributions (in characters):")
    text_lengths = df_processed['Text'].str.len()
    print(f"  Min length:  {text_lengths.min():,}")
    print(f"  Max length:  {text_lengths.max():,}")
    print(f"  Mean length: {text_lengths.mean():.1f}")
    print(f"  Median length:{text_lengths.median():.1f}")
    
    # 8. Print Sample Before/After Cleaning
    print("\n[Step 8] Sample Before/After Cleaning:")
    sample_idx = 0
    if len(df_processed) > 0:
        sample_row = df_processed.iloc[sample_idx]
        print("-" * 50)
        print("ORIGINAL PHONE FIELD:")
        print(repr(sample_row.get('Phone', '')))
        print("\nORIGINAL SUMMARY FIELD (First 300 chars):")
        print(repr(sample_row.get('Original_Summary', '')[:300]))
        print("\nCLEANED SUMMARY FIELD (First 300 chars):")
        print(repr(sample_row.get('Summary', '')[:300]))
        print("\nEXTRACTED STRUCTURED FIELDS:")
        print(f"  Extracted Skills (First 10): {sample_row.get('ExtractedSkills', [])[:10]}")
        print(f"  Years of Experience:         {sample_row.get('YearsOfExperience', 0.0)}")
        print(f"  Degree Level:                {sample_row.get('DegreeLevel', 'None')}")
        print("-" * 50)
        
    # 9. Persist Cleaned Dataset
    print(f"\n[Step 9] Saving processed resumes to parquet: {processed_parquet_path}")
    os.makedirs(os.path.dirname(processed_parquet_path), exist_ok=True)
    df_processed.to_parquet(processed_parquet_path, index=False)
    print("Dataset saved successfully.")
    
    print("\n" + "=" * 60)
    print("Preprocessing & EDA Phase Completed Successfully")
    print("=" * 60)

if __name__ == "__main__":
    main()
