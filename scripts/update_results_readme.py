import re
from pathlib import Path

log_path = Path("sift_log.txt")
readme_path = Path("README.md")

log_text = log_path.read_text()

pattern = re.compile(
    r"--- Test Case (\d+) ---.*?"
    r"Execution time:\s*([\d\.]+)\s*ms.*?"
    r"Found\s+(\d+)\s+keypoints.*?"
    r"Validation Result:\s*(\w+)",
    re.DOTALL
)

rows = []
for match in pattern.finditer(log_text):
    case, time, keypoints, result = match.groups()
    emoji = "✅" if result == "Pass" else "❌"
    rows.append(f"| {case} | {time} | {keypoints} | {emoji} {result} |")

table_header = (
    "| Test Case | Time (ms) | Keypoints | Result |\n"
    "|-----------|-----------|-----------|--------|\n"
)
table = table_header + "\n".join(rows)

# Replace between markers in README
readme_text = readme_path.read_text()
start_marker = "<!-- RESULT_TABLE_START -->"
end_marker = "<!-- RESULT_TABLE_END -->"

if start_marker in readme_text and end_marker in readme_text:
    new_text = re.sub(
        f"{start_marker}.*?{end_marker}",
        f"{start_marker}\n{table}\n{end_marker}",
        readme_text,
        flags=re.DOTALL
    )
else:
    new_text = readme_text + f"\n\n{start_marker}\n{table}\n{end_marker}\n"

readme_path.write_text(new_text)
print("README.md updated with latest test results.")

