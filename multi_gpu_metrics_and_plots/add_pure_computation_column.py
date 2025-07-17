import pandas as pd
import glob
import os

def add_pure_computation_column(csv_filepath):
    """Add pure_computation_time column to a CSV file."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_filepath)
        
        # Check if the required columns exist
        if 'total_time_ms' not in df.columns or 'total_pure_comm_time_ms' not in df.columns:
            print(f"Warning: Required columns not found in {csv_filepath}")
            return False
        
        # Check if pure_computation_time column already exists
        if 'pure_computation_time' in df.columns:
            print(f"Column 'pure_computation_time' already exists in {csv_filepath}")
            return False
        
        # Calculate pure_computation_time = total_time_ms - total_pure_comm_time_ms
        df['pure_computation_time'] = df['total_time_ms'] - df['total_pure_comm_time_ms']
        
        # Reorder columns to put pure_computation_time after computation_time_ms
        columns = df.columns.tolist()
        
        # Find the index of computation_time_ms
        if 'computation_time_ms' in columns:
            comp_time_idx = columns.index('computation_time_ms')
            # Remove pure_computation_time from its current position
            columns.remove('pure_computation_time')
            # Insert it after computation_time_ms
            columns.insert(comp_time_idx + 1, 'pure_computation_time')
            df = df[columns]
        
        # Write back to CSV
        df.to_csv(csv_filepath, index=False)
        print(f"Successfully added 'pure_computation_time' column to {csv_filepath}")
        return True
        
    except Exception as e:
        print(f"Error processing {csv_filepath}: {str(e)}")
        return False

def process_all_csvs(directory="."):
    """Process all CSV files in the given directory."""
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in directory: {directory}")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  - {os.path.basename(csv_file)}")
    
    print("\nProcessing files...")
    
    success_count = 0
    total_count = len(csv_files)
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"\nProcessing: {filename}")
        
        if add_pure_computation_column(csv_file):
            success_count += 1
    
    print(f"\n=== Summary ===")
    print(f"Total files processed: {total_count}")
    print(f"Successfully modified: {success_count}")
    print(f"Skipped/Failed: {total_count - success_count}")

def verify_columns(directory="."):
    """Verify that all CSV files now have the pure_computation_time column."""
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    print("\n=== Verification ===")
    print("Checking all CSV files for the new column...")
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        try:
            df = pd.read_csv(csv_file)
            if 'pure_computation_time' in df.columns:
                # Show some sample values
                sample_values = df['pure_computation_time'].head(3).tolist()
                print(f"✓ {filename}: pure_computation_time column present (samples: {sample_values})")
            else:
                print(f"✗ {filename}: pure_computation_time column MISSING")
        except Exception as e:
            print(f"✗ {filename}: Error reading file - {str(e)}")

if __name__ == "__main__":
    print("Adding 'pure_computation_time' column to all CSV files...")
    print("Formula: pure_computation_time = total_time_ms - total_pure_comm_time_ms")
    print("=" * 80)
    
    # Process all CSV files in current directory
    process_all_csvs()
    
    # Verify the changes
    verify_columns()
    
    print("\nDone!") 