from collections import defaultdict

from character_pipeline.config import MALE_TITLES, NICKNAMES
from character_pipeline.name_parsing import ParsedName, parse_name
from character_pipeline.union_find import UnionFind


def _depluralize(sur: str, all_surnames: set[str]) -> str:
    for suf in ("ses", "es", "s"):
        if sur.endswith(suf) and sur[:- len(suf)] in all_surnames:
            return sur[:- len(suf)]
    return sur


def surname_keys(parsed: dict[str, ParsedName]) -> dict[str, str]:
    all_surnames = {p.surname for p in parsed.values() if p.surname}

    def key(sur: str) -> str:
        if not sur:
            return ""
        return _depluralize(sur, all_surnames).split()[-1]

    return {n: key(p.surname) for n, p in parsed.items()}


def reparse_title_plus_given(parsed: dict[str, ParsedName], sur_key: dict[str, str]) -> set[str]:
    all_givens: set[str] = set()
    for p in parsed.values():
        all_givens.update(p.given)
    for n in list(parsed.keys()):
        t, g, s = parsed[n]
        if t and not g and s and " " not in s:
            s_norm = NICKNAMES.get(s, s)
            if s_norm in all_givens:
                parsed[n] = ParsedName(t, (s_norm,), "")
                sur_key[n] = ""
    return all_givens


def _share_given(p_a: ParsedName, p_b: ParsedName) -> bool:
    if not p_a.given or not p_b.given:
        return False
    if p_a.title and p_b.title and p_a.title != p_b.title:
        return False
    ga_s, gb_s = set(p_a.given), set(p_b.given)
    return ga_s.issubset(gb_s) or gb_s.issubset(ga_s)


def _members_of(rep: str, pool: list[str], uf: UnionFind) -> list[str]:
    return [x for x in pool if uf.find(x) == rep]


def stage_a_shared_given(
    names: list[str],
    parsed: dict[str, ParsedName],
    sur_key: dict[str, str],
    uf: UnionFind,
) -> None:
    surname_pool: dict[str, list[str]] = defaultdict(list)
    for n in names:
        if sur_key[n]:
            surname_pool[sur_key[n]].append(n)
    for members in surname_pool.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if _share_given(parsed[members[i]], parsed[members[j]]):
                    uf.union(members[i], members[j])


def stage_b_title_only(
    names: list[str],
    parsed: dict[str, ParsedName],
    sur_key: dict[str, str],
    uf: UnionFind,
) -> None:
    surname_pool: dict[str, list[str]] = defaultdict(list)
    for n in names:
        if sur_key[n]:
            surname_pool[sur_key[n]].append(n)
    for n in names:
        t, g, s = parsed[n]
        if not t or g or not s:
            continue
        pool = surname_pool.get(sur_key[n], [])
        cands = set()
        for m in pool:
            r = uf.find(m)
            if r == uf.find(n):
                continue
            if any(parsed[x].title == t for x in _members_of(r, pool, uf)):
                cands.add(r)
        if len(cands) == 1:
            uf.union(next(iter(cands)), n)


def stage_c_bare_surname(
    names: list[str],
    parsed: dict[str, ParsedName],
    sur_key: dict[str, str],
    uf: UnionFind,
    counts: dict[str, int],
) -> None:
    surname_pool: dict[str, list[str]] = defaultdict(list)
    for n in names:
        if sur_key[n]:
            surname_pool[sur_key[n]].append(n)
    for n in names:
        t, g, s = parsed[n]
        if t or g or not s:
            continue
        pool = surname_pool.get(sur_key[n], [])
        title_only_counts: dict[str, int] = defaultdict(int)
        male_reps: set[str] = set()
        seen_reps: set[str] = set()
        for m in pool:
            r = uf.find(m)
            if r == uf.find(n) or r in seen_reps:
                continue
            seen_reps.add(r)
            mem = _members_of(r, pool, uf)
            has_title = any(parsed[x].title for x in mem)
            has_given = any(parsed[x].given for x in mem)
            if has_title and not has_given:
                title_only_counts[r] = sum(counts.get(x, 0) for x in mem)
            if any(parsed[x].title in MALE_TITLES for x in mem):
                male_reps.add(r)
        if title_only_counts:
            ranked = sorted(title_only_counts.items(), key=lambda kv: -kv[1])
            top_rep, top_count = ranked[0]
            if len(ranked) == 1 or top_count > 2 * ranked[1][1]:
                uf.union(top_rep, n)
                continue
        if len(male_reps) == 1:
            uf.union(next(iter(male_reps)), n)


def premerge_title_given_no_surname(
    names: list[str],
    parsed: dict[str, ParsedName],
    uf: UnionFind,
) -> None:
    tg_seed: dict[tuple[str, frozenset[str]], str] = {}
    for n in names:
        t, g, s = parsed[n]
        if not t or not g or s:
            continue
        key = (t, frozenset(g))
        if key in tg_seed:
            uf.union(tg_seed[key], n)
        else:
            tg_seed[key] = n


def stage_f_title_given_no_surname(
    names: list[str],
    parsed: dict[str, ParsedName],
    uf: UnionFind,
) -> None:
    for n in names:
        t, g, s = parsed[n]
        if not t or not g or s:
            continue
        g_set = set(g)
        cands = set()
        for m in names:
            if m == n:
                continue
            pm = parsed[m]
            if pm.title == t and (g_set & set(pm.given)):
                cands.add(uf.find(m))
        cands.discard(uf.find(n))
        if len(cands) == 1:
            uf.union(next(iter(cands)), n)


def reparse_bare_surname_as_given(
    parsed: dict[str, ParsedName],
    all_givens: set[str],
    uf: UnionFind,
) -> None:
    for n in list(parsed.keys()):
        t, g, s = parsed[n]
        if t or g or not s or " " in s:
            continue
        if uf.find(n) != n:
            continue
        s_norm = NICKNAMES.get(s, s)
        if s_norm in all_givens:
            parsed[n] = ParsedName("", (s_norm,), "")


def premerge_bare_given(names: list[str], parsed: dict[str, ParsedName], uf: UnionFind) -> None:
    bare_given_seed: dict[str, str] = {}
    for n in names:
        t, g, s = parsed[n]
        if t or s or not g:
            continue
        for tok in g:
            if tok in bare_given_seed:
                uf.union(bare_given_seed[tok], n)
            else:
                bare_given_seed[tok] = n


def stage_d_bare_given(
    names: list[str],
    parsed: dict[str, ParsedName],
    uf: UnionFind,
) -> None:
    given_to_reps: dict[str, set[str]] = defaultdict(set)
    for n in names:
        for tok in parsed[n].given:
            given_to_reps[tok].add(uf.find(n))
    for n in names:
        t, g, s = parsed[n]
        if t or s or not g:
            continue
        cands: set[str] = set()
        for tok in g:
            cands |= given_to_reps.get(tok, set())
        cands.discard(uf.find(n))
        if len(cands) == 1:
            uf.union(next(iter(cands)), n)


def _canon_score(n: str, parsed: dict[str, ParsedName], counts: dict[str, int]):
    title, given, sur = parsed[n]
    return (
        -int(bool(given) and bool(sur) and not bool(title)),
        -int(bool(title) and bool(given) and bool(sur)),
        -int(bool(given) and bool(sur)),
        -int(bool(title) and bool(sur)),
        -int(bool(sur)),
        -counts.get(n, 0),
        -len(n),
        n,
    )


def build_groups(
    names: list[str],
    uf: UnionFind,
    parsed: dict[str, ParsedName],
    counts: dict[str, int],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    reps: dict[str, list[str]] = defaultdict(list)
    for n in names:
        reps[uf.find(n)].append(n)
    mapping: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    for members in reps.values():
        canonical = min(members, key=lambda n: _canon_score(n, parsed, counts))
        groups[canonical] = sorted(members)
        for m in members:
            mapping[m] = canonical
    return mapping, groups


def cluster_rule_based(
    names: list[str],
    counts: dict[str, int],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Cluster names by structured title / given / surname matching."""
    parsed = {n: parse_name(n) for n in names}
    sur_key = surname_keys(parsed)
    all_givens = reparse_title_plus_given(parsed, sur_key)
    uf = UnionFind(names)

    stage_a_shared_given(names, parsed, sur_key, uf)
    stage_b_title_only(names, parsed, sur_key, uf)
    stage_c_bare_surname(names, parsed, sur_key, uf, counts)
    premerge_title_given_no_surname(names, parsed, uf)
    stage_f_title_given_no_surname(names, parsed, uf)
    reparse_bare_surname_as_given(parsed, all_givens, uf)
    premerge_bare_given(names, parsed, uf)
    stage_d_bare_given(names, parsed, uf)

    return build_groups(names, uf, parsed, counts)
