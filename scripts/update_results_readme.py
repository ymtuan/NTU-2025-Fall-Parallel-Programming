import re
import subprocess
from datetime import datetime
from pathlib import Path

# Paths
log_path = Path("runtime_log.txt")
readme_path = Path("README.md")

# Read log file
log_text = log_path.read_text()

# Extract test results
pattern = re.compile(
    r"--- Test Case (\d+) ---.*?"
    r"Execution time:\s*([\d\.]+)\s*ms.*?"
    r"Found\s+(\d+)\s+keypoints.*?"
    r"Validation Result:\s*(\w+)",
    re.DOTALL
)

# Get commit hash and timestamp
commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

# Build rows
rows = []
for match in pattern.finditer(log_text):
    case, time, keypoints, result = match.groups()
    emoji = "✅" if result == "Pass" else "❌"
    rows.append(f"| `{commit_hash}` | {timestamp} | {case} | {time} | {keypoints} | {emoji} {result} |")

# Load README and markers
readme_text = readme_path.read_text()
start_marker = "<!-- RESULT_TABLE_START -->"
end_marker = "<!-- RESULT_TABLE_END -->"

if start_marker not in readme_text or end_marker not in readme_text:
    # Insert a new table if not present
    table_header = (
        "| Commit | Date | Test Case | Time (ms) | Keypoints | Result |\n"
        "|--------|------|-----------|-----------|-----------|--------|\n"
    )
    new_table = table_header + "\n".join(rows)
    readme_text += f"\n\n{start_marker}\n{new_table}\n{end_marker}\n"
else:
    # Append rows to the existing table
    parts = readme_text.split(start_marker)
    head = parts[0]
    body = parts[1].split(end_marker)
    table_section = body[0]
    footer = end_marker + body[1]

    # Append rows after the header line
    new_table_section = table_section.strip() + "\n" + "\n".join(rows) + "\n"
    readme_text = head + start_marker + "\n" + new_table_section + footer

# Write back
readme_path.write_text(readme_text)
print(f"Appended {len(rows)} results to README.md for commit {commit_hash}")
