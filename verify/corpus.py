"""Fixed prompt corpus for pp2_receipts.py (parity, acceptance window) and pp2_soak.py.

This is a REPLACEMENT, written 2026-11-20. Both scripts used to read their prompts from
/tmp/accept_sweep.py, which was never committed and is gone from the boxes, together with
/tmp/tp2_refs.json (the TP2 reference hashes). So:

  * numbers taken with this corpus are not comparable with any receipt from before
    2026-11-20 (README: 391 tok/s, the C1..C8 soak, 0/24 parity vs TP2);
  * parity is now measured against references you record yourself on a lane you trust
    (PP2_RECORD_REFS=1 python3 verify/pp2_receipts.py), not against TP2.

Four classes of six prompts, 24 in all, the shape the old scripts expect: a dict named
CLASSES, class name -> list of prompts, with a "prose" class.
"""

CLASSES = {
    "prose": [
        "Write about 350 words describing a harbour town on the morning after a storm, from the point of view of the person who opens the fish market.",
        "Explain to a curious twelve-year-old why the Moon always shows us the same face. About 250 words, no formulas.",
        "Write a letter of about 300 words from a lighthouse keeper to her successor, covering what the job is really like and three practical tips.",
        "Summarise the causes and the consequences of the invention of the printing press in about 300 words, as flowing prose without bullet points.",
        "Describe how sourdough bread is made from starter to baked loaf, in about 350 words, for someone who has never baked.",
        "Write a 300-word scene in which two old friends meet at a railway station after twenty years and neither says what they came to say.",
    ],
    "code": [
        "Write a Python function merge_intervals(intervals) that merges overlapping closed integer intervals and returns them sorted. Include a docstring and five assert-based tests.",
        "Write a Rust function that parses a semantic version string such as 1.4.2-rc.1+build.5 into a struct and implements ordering per the SemVer 2.0 precedence rules. Include unit tests.",
        "Write a bash script that takes a directory and prints the ten largest files under it with human-readable sizes, handling file names that contain spaces and newlines.",
        "Implement an LRU cache in Go with Get and Put in O(1), safe for concurrent use, with a short example in main.",
        "Write a SQL query for PostgreSQL that returns, for each customer, their three most recent orders and each order's share of that customer's lifetime spend. State the tables you assume.",
        "Write a TypeScript function debounce(fn, waitMs) with correct generic typing for arguments and this, a cancel method, and a short usage example.",
    ],
    "reasoning": [
        "A tank holds 2400 litres. Pump A fills it in 6 hours, pump B in 4 hours, and a drain empties it in 12 hours. Starting empty with all three running, how long until it is full? Show each step.",
        "Five people sit in a row. Ana is not at either end. Ben is immediately left of Cho. Dev is somewhere right of Ana. Eli is at the left end. Give every seating that satisfies all four statements, with the reasoning.",
        "A shop raises a price by 20 percent, then offers 20 percent off the new price. Is the final price higher, lower or equal to the original, and by how much? Then generalise to a rise and cut of p percent.",
        "You have a 3-litre jug and a 5-litre jug and unlimited water. Give the shortest sequence of fills, pours and empties that leaves exactly 4 litres in the larger jug, and argue that no shorter sequence exists.",
        "A train leaves city X at 08:15 at 90 km/h. A second train leaves city Y, 510 km away, at 09:00 towards X at 120 km/h. At what clock time do they meet and how far from X? Show the arithmetic.",
        "Three boxes are labelled APPLES, ORANGES and MIXED, and every label is wrong. You may draw one fruit from one box. Which box do you draw from, and how do you then relabel all three? Explain why it works.",
    ],
    "structured": [
        "Return a JSON object, and nothing else, describing a fictional library book with the keys title, authors (array), year (integer), isbn13 (string), subjects (array of at least three) and available (boolean).",
        "Convert this to a Markdown table with columns Planet, Moons, Day length: Mercury has no moons and a day of 1408 hours; Venus has none and 5832 hours; Earth has one and 24 hours; Mars has two and 25 hours; Jupiter has 95 and 10 hours.",
        "Write a YAML configuration for a web service with sections server (host, port, tls with cert and key paths), database (url, pool with min and max), and logging (level, json flag). Add a one-line comment per section.",
        "Extract every date from this text and list them one per line in ISO 8601 form: The lease began on March 3rd, 2021, was amended 14/07/2022, renewed on the first of January 2024 and ends December 31 2026.",
        "Produce a numbered ten-step checklist for rotating the TLS certificate of a public web service with zero downtime. One sentence per step, imperative mood.",
        "Write a CSV with a header row and eight data rows for an inventory of hand tools with the columns sku, name, category, quantity, unit_price_usd. Quote fields only where CSV requires it.",
    ],
}
