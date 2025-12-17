"""
Quick script to check the progress of Test.py execution
"""
import os
import glob

print("="*60)
print("Checking Test.py Progress")
print("="*60)

# Check for output files
files_to_check = {
    "Heatmaps": ["quantum_kernel_basic_heatmap.png", "quantum_kernel_zz_heatmap.png"],
    "Results": ["svm_results_summary.txt"],
    "Log": ["test_output.log"]
}

for category, files in files_to_check.items():
    print(f"\n{category}:")
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  [OK] {file} ({size:,} bytes)")
        else:
            print(f"  [ ] {file} (not created yet)")

# Check if Python is running
try:
    import subprocess
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                          capture_output=True, text=True)
    if 'python.exe' in result.stdout:
        print("\n[OK] Python process is running")
    else:
        print("\n[ ] Python process not found (may have completed)")
except:
    print("\n[?] Could not check Python process status")

print("\n" + "="*60)
print("Expected Output Files:")
print("="*60)
print("1. quantum_kernel_basic_heatmap.png - Basic feature map kernel visualization")
print("2. quantum_kernel_zz_heatmap.png - ZZ feature map kernel visualization")
print("3. svm_results_summary.txt - SVM performance results")
print("4. test_output.log - Full execution log (if logging enabled)")
print("\nNote: Full kernel computation may take 30-60+ minutes")

