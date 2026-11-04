from nltk.tokenize import WordPunctTokenizer


def build_source_units(chunks: list[str]) -> tuple[list[list[list[int | str]]], dict[int, tuple[int, int, int]]]:
    """Prepare nonoverlapping word/punctuation units in original source order."""
    words = WordPunctTokenizer()
    display = []
    lookup = {}
    for index, chunk in enumerate(chunks):
        units = []
        for start, end in words.span_tokenize(chunk):
            unit_id = len(lookup)
            units.append([unit_id, chunk[start:end]])
            lookup[unit_id] = (index, start, end)
        display.append(units)
    return display, lookup


def resolve_source_ranges(ranges, chunks: list[str], chunk_ids: list[str],
                          lookup: dict[int, tuple[int, int, int]]) -> tuple[list[dict], list[str]]:
    """Resolve explicit continuous source ranges into canonical engine quotes."""
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("fact must cite at least one current source unit range")
    selected = []
    seen = set()
    for source_range in ranges:
        if (not isinstance(source_range, dict) or set(source_range) != {"first", "last"}
                or any(type(source_range[key]) is not int or source_range[key] not in lookup
                       for key in ("first", "last"))):
            raise ValueError("fact source range requires first and last current integer unit IDs")
        pair = (source_range["first"], source_range["last"])
        if pair in seen:
            raise ValueError("fact contains duplicate source unit ranges")
        seen.add(pair)
        if pair[0] > pair[1]:
            raise ValueError("fact source range first must not exceed last")
        selected.extend(lookup[unit_id] for unit_id in range(pair[0], pair[1] + 1))
    merged = []
    for index, start, end in sorted(selected):
        if (merged and merged[-1][0] == index
                and (start <= merged[-1][2] or not chunks[index][merged[-1][2]:start].strip())):
            previous = merged[-1]
            merged[-1] = (index, previous[1], max(previous[2], end))
        else:
            merged.append((index, start, end))
    quotes = []
    for index, start, end in merged:
        quote = chunks[index][start:end]
        offset, occurrence = -len(quote), -1
        while offset < start:
            offset = chunks[index].find(quote, offset + len(quote))
            occurrence += 1
            if offset < 0 or offset > start:
                raise ValueError("selected source units are not an exact nonoverlapping quote occurrence")
        quotes.append({"chunk_id": chunk_ids[index], "quote": quote, "occurrence": occurrence})
    return quotes, list(dict.fromkeys(quote["chunk_id"] for quote in quotes))
