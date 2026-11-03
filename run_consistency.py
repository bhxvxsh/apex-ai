import sys
import os

# Add APEX_AI and 05_Analytics to path
sys.path.insert(0, r"D:\yeah def my nyc project\APEX_AI")
sys.path.insert(0, r"D:\yeah def my nyc project\APEX_AI\05_Analytics")

# Change to the APEX_AI directory so relative paths in modules work
os.chdir(r"D:\yeah def my nyc project\APEX_AI")

# Import the consistency module
from consistency import get_all_drivers_consistency

# Call it
all_consistency = get_all_drivers_consistency()

# Print results
print("=== F1 Driver Consistency Rankings (2023-2025) ===")
print(f"{'Rank':<6} {'Driver':<12} {'Team':<10} {'Races':<6} {'Finish%':<9} {'Avg Pos':<8} {'StdDev':<8} {'Podiums'}")
print("-" * 70)
for i, row in all_consistency.iterrows():
    print(f"{i+1:<6} {row['driverId']:<12} {row['teamName']:<10} " +
          f"{row['races_analyzed']:<6} {row['finish_rate']:<9.1f} " +
          f"{row['avg_position']:<8.2f} {row['position_stddev']:<8.3f} " +
          f"{row['podium_finishes']}")

print()
print("Top 5 most consistent:")
print(all_consistency.head(5)[['driverId', 'fullName', 'position_stddev', 'finish_rate', 'podium_finishes']].to_string(index=False))