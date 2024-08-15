from pathlib import Path

FILE = Path(__file__).resolve()
ROOT1 = FILE.parent
ROOT2 = FILE.parents[1]
print()
print(ROOT1  / ROOT2 )