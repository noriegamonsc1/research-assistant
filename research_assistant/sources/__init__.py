"""Source adapter registry. Add a new source = add a module + one line here + list it in fields.yaml."""

from . import crossref, openalex, pubmed, semanticscholar

REGISTRY = {
    "openalex": openalex.search,
    "semanticscholar": semanticscholar.search,
    "pubmed": pubmed.search,
    "crossref": crossref.search,
}
