import csv, sys
from collections import Counter

folder = sys.argv[1]
for level in ("1", "2", "3"):
    c = Counter()
    for r in csv.DictReader(open(f"{folder}/exp1_raw.csv", encoding="utf-8")):
        if r["algo"] == "astar" and r["level"] == level:
            if r["recovered"] == "1":
                c["recovered"] += 1
            elif r["solved"] == "0":
                c["unreadable"] += 1
            else:
                c["solved but wrong"] += 1
    print(folder, "level", level, dict(c))