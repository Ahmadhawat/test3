You’re not getting BM25 scores because the Whoosh query you build from a full sentence ends up matching nothing. With QueryParser("text", schema=ix.schema) the parser interprets punctuation/boolean syntax and (depending on the default group/analyzer & stopwords) often builds an ANDed query that no doc satisfies. A single token works; a multi-token sentence silently becomes a too-strict query → hits == [] → your lex_scores dict is empty → you print 0.

Quick fix (robust & parser-free)

Tokenize the sentence with the same analyzer Whoosh used to index the field and build a plain OR of Term queries. This avoids parser quirks entirely and works well with BM25.

Replace your BM25 block with this:

from whoosh.scoring import BM25F
from whoosh import query as Q

# -------- BM25 search (robust sentence handling) --------
with whoosh_index.open_dir(lroot) as ix:
    with ix.searcher(weighting=BM25F()) as searcher:
        # tokenize with the field's analyzer to match the index
        analyzer = ix.schema["text"].analyzer
        tokens = [tok.text for tok in analyzer(qtext) if tok.text]

        if tokens:
            # OR all terms so sentences become a bag-of-words query
            q = Q.Or([Q.Term("text", t) for t in tokens])
            # (optional) require at least N terms to match:
            # q = Q.DisjunctionMax([Q.Term("text", t) for t in tokens], tiebreak=0.0)
        else:
            # nothing left after stopwords -> match all
            q = Q.Every()

        hits = searcher.search(q, limit=k_lex)
        lex_scores: Dict[int, float] = {int(h["doc_id"]): float(h.score) for h in hits}

Alternative (if you’d rather keep QueryParser)

Use an OR group and escape punctuation:

from whoosh.qparser import QueryParser, OrGroup
from whoosh.qparser import syntax

with whoosh_index.open_dir(lroot) as ix:
    with ix.searcher(weighting=BM25F()) as searcher:
        parser = QueryParser("text", ix.schema, group=OrGroup.factory(0.1))
        parser.remove_plugin_class(syntax.RangePlugin)  # optional: less strict
        q = parser.parse(qtext)  # or: parser.parse(syntax.escape(qtext))
        hits = searcher.search(q, limit=k_lex)
        lex_scores = {int(h["doc_id"]): float(h.score) for h in hits}

Either approach makes sentence queries return real BM25 hits, so your lexical score won’t be stuck at 0 anymore.

