"""Context: query both real Pinecone namespaces and save their original evidence.

No hosted generation is requested. Numeric coverage is a diagnostic, not proof
of correct row/year interpretation or a substitute for RAGAS context recall.
"""

import json
import os
import argparse
from pathlib import Path

from financial_rag.calculator import numeric_literals
from financial_rag.cloud import PineconeSearch
from financial_rag.core import Index
from financial_rag.embedding import Embedder
from financial_rag.generation import load_environment
from financial_rag.retrieval import flashrank


def main():
    """Context: test prefilters, stored-text verification and identical B/C candidates."""
    load_environment()
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["TRANSFORMERS_OFFLINE"]="1"
    folder=Path(__file__).resolve().parents[1]/"runs/apple-pinecone-benchmark-v1"
    parser=argparse.ArgumentParser(description="Record real Pinecone retrieval evidence without generation.")
    parser.add_argument("--strategy",choices=("all","fixed","semantic"),default="all")
    options=parser.parse_args()
    questions=json.loads((folder/"questions.json").read_text(encoding="utf-8"))
    fixed=Index.load(folder/"fixed.json")
    semantic=Index.load(folder/"semantic.json") if options.strategy!="fixed" else None
    embedder=Embedder(fixed.manifest["embedding_model"],fixed.manifest["embedding_revision"])
    output=folder/"retrieval-checks.json"
    rows=json.loads(output.read_text(encoding="utf-8")).get("rows",[]) if output.exists() else []
    for strategy,index in (("fixed",fixed),("semantic",semantic)):
        if options.strategy!="all" and options.strategy!=strategy:
            continue
        rows=[row for row in rows if row["namespace"]!=strategy]
        search=PineconeSearch("financial-rag",strategy)
        for question in questions:
            hits,pool=search(index,question["question"],embedder,question["filters"],10,10)
            variants=[("A" if strategy=="fixed" else "B",hits[:3])]
            if strategy=="semantic":
                variants.append(("C",flashrank(hits,question["question"])[:3]))
            for configuration,selected in variants:
                source_numbers=numeric_literals("\n".join(hit.text for hit in selected))
                expected=numeric_literals(" ".join(question.get("expected_numbers",[])))
                rows.append({"configuration":configuration,"question_id":question["id"],
                    "filters":question["filters"],"namespace":strategy,"candidate_pool":pool,
                    "returned_contexts":[{"id":hit.chunk.id,"citation":hit.chunk.citation(),
                                          "text":hit.text,"score":hit.score} for hit in selected],
                    "reference_numbers_present":sorted(str(value) for value in source_numbers&expected),
                    "reference_numbers_absent":sorted(str(value) for value in expected-source_numbers),
                    "numeric_presence_is_semantic_proof":False})
            (folder/"retrieval-checks.json").write_text(json.dumps({"rows":rows},indent=2),encoding="utf-8")
            print(strategy+" "+question["id"]+": "+str(len(hits))+" stored-text-verified candidates.",flush=True)


if __name__=="__main__":
    main()
