# memory.py
import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import os
import re

class BioClinicalMemoryAgent:
    """
    BioClinicalBERT retrieval with intelligent chunking, metadata-aware filtering,
    and lexical reranking to prevent wrong-domain matches.
    """

    def find_best_match(self, query: str) -> dict:
        threshold = 0.85
        domain_min = 0.6  # Minimum score for domain acceptance
        results = self.retrieve_similar_cases(
            query,
            k=1,
            return_chunks=True,
            similarity_threshold=threshold,
            use_metadata_filters=True,
            apply_lexical_rerank=True
        )
        # Extract intended body_system from query
        filters = self._extract_filters_from_query(query)
        want_sys = set(filters.get("body_system", []))
        # Check abstain rule
        if not results:
            return {
                "query": query,
                "diagnosis": None,
                "note": "No relevant cases found. Please expand history/exam for more context."
            }
        best = results[0]
        best_sys = set([best['metadata'].get('body_system')])
        best_score = best['score']
        # Abstain if no candidates match body_system or score too low
        if want_sys and not (best_sys & want_sys):
            return {
                "query": query,
                "diagnosis": None,
                "note": "No matching cases in the relevant domain. Please expand history/exam."
            }
        if best_score < domain_min:
            return {
                "query": query,
                "diagnosis": None,
                "note": "No high-confidence match found. Please expand history/exam."
            }
        diagnosis = best['metadata'].get('condition', None)
        return {"query": query, "diagnosis": diagnosis}

    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 chunk_size: int = 300, overlap: int = 50):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedding_dim = 768

        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.documents: List[str] = []
        self.doc_metadata: List[Dict[str, Any]] = []

        import faiss
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self.faiss_chunk_id_map: List[int] = []

        self._init_embedding_system()
        self._compile_query_patterns()

        print(f"✅ BioClinicalMemoryAgent Initialized")
        print(f"   Model: {model_name}")
        print(f"   Chunking: {chunk_size} tokens with {overlap} overlap")
        print(f"   Embeddings: {self.embedding_dim}D with cosine similarity")

    def _init_embedding_system(self):
        from transformers import AutoTokenizer, AutoModel
        print(f"🔄 Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.use_transformers = True
        print("✅ BioClinicalBERT loaded via transformers")
        print("✅ Medical domain embedding system initialized")

    def _compile_query_patterns(self):
        self._patterns = {
            "ophthalmology": re.compile(r"\b(eye|ocular|conjunct|red eye|photophobia|vision)\b", re.I),
            "vascular": re.compile(r"\b(raynaud|color change|triphasic|white|blue|blanch|rewarm|tingl)\w*", re.I),
            "respiratory": re.compile(r"\b(cough|hemoptysis|sputum|wheeze|dyspnea|shortness of breath|pleuritic)\b", re.I),
            "gastroenterology": re.compile(r"\b(abdominal|ruq|nausea|vomit|biliary|gallstone|reflux|heartburn)\b", re.I),
            "neurology": re.compile(r"\b(weakness|numb|tingl|droop|slurred|aphasia|syncope|seizure)\b", re.I),
            "dermatology": re.compile(r"\b(rash|itch|pruritus|scaly|lesion|eczema|dermatitis|alopecia|hair loss|scalp|thinning|bald|minoxidil)\b", re.I),
            "genitourinary": re.compile(r"\b(dysuria|frequency|urgency|flank|suprapubic|hematuria)\b", re.I),
            "cardiology": re.compile(r"\b(angina|exertional chest|chest pressure|palpit|nitro)\b", re.I),
            "endocrine": re.compile(r"\b(polyuria|polydipsia|hyperglyc|hba1c|weight loss)\b", re.I),
            "musculoskeletal": re.compile(r"\b(knee|shoulder|hip|ankle|runner|run|overuse|joint|ligament|menisc|tendon|patell|iliotibial|it band)\b", re.I),
        }
        self._boost_terms = {
            "ophthalmology": ["conjunctivitis","red_eye","itching","no_discharge","photophobia","vision_change","allergic_history"],
            "vascular": ["raynaud","vasospasm","acrocyanosis","color_change","blanching","rewarming","digits"],
            "respiratory": ["hemoptysis","sputum","wheeze","dyspnea","bronchitis","pneumonia","tb"],
            "gastroenterology": ["ruq","biliary_colic","gallstones","ultrasound","murphy","fatty_meal"],
            "neurology": ["TIA","transient_deficit","dysarthria","facial_droop","arm_weakness"],
            "dermatology": [
                "atopic","eczema","pruritus","flexural","emollients",
                "alopecia","hair_loss","scalp_thinning","bald","minoxidil","androgenetic","telogen_effluvium","alopecia_areata"
            ],
            "genitourinary": ["dysuria","frequency","suprapubic_pain","pyuria","nitrite","cystitis"],
            "cardiology": ["stable_angina","exertional_chest_pressure","relief_with_rest","stress_test","nitroglycerin"],
            "endocrine": ["hyperglycemia","elevated_hba1c","metformin","polyuria","polydipsia"],
            "musculoskeletal": ["knee_pain","running_trigger","overuse","no_trauma","patellofemoral","it_band","pes_anserine","meniscal_signs"],
        }

    def add_document(self, text: str, source: str = "manual",
                     metadata: Optional[Dict[str, Any]] = None) -> int:
        doc_id = len(self.documents)
        doc_meta = {"doc_id": doc_id, "source": source}
        if metadata:
            doc_meta.update(metadata)

        # Enrich metadata with inferred system/tags if missing
        if "body_system" not in doc_meta or "symptom_tags" not in doc_meta:
            inferred_system, tags = self._infer_system_and_tags(text)
            doc_meta.setdefault("body_system", inferred_system)
            doc_meta.setdefault("symptom_tags", tags)

        self.documents.append(text)
        self.doc_metadata.append(doc_meta)

        chunks = self._chunk_document(text)
        print(f"📄 Document {doc_id}: {len(chunks)} chunks created")

        for i, chunk in enumerate(chunks):
            chunk_meta = {
                "doc_id": doc_id,
                "chunk_id": len(self.chunks),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source": source,
                "condition": doc_meta.get("condition"),
                "body_system": doc_meta.get("body_system"),
                "symptom_tags": doc_meta.get("symptom_tags", []),
            }
            chunk_meta["word_count"] = len(chunk.split())

            embedding = self._get_embedding(chunk).astype('float32')
            self.faiss_index.add(embedding.reshape(1, -1))
            self.faiss_chunk_id_map.append(len(self.chunks))
            self.chunks.append(chunk)
            self.chunk_metadata.append(chunk_meta)

        print(f"✅ Added document {doc_id} ({doc_meta.get('condition', 'unknown')}) with {len(chunks)} chunks from {source}")
        return doc_id

    def _infer_system_and_tags(self, text: str) -> Tuple[str, List[str]]:
        tl = text.lower()
        system_scores = {sys: len(p.findall(tl)) for sys, p in self._patterns.items()}
        inferred = max(system_scores, key=system_scores.get) if system_scores else "general"
        tags = []
        for sys, terms in self._boost_terms.items():
            for t in terms:
                if t.replace("_", " ") in tl or t in tl:
                    tags.append(t)
        tags = list(dict.fromkeys(tags))[:10]
        return inferred, tags

    def _chunk_document(self, text: str) -> List[str]:
        sections = re.split(r'\n\s*\n|\n(?=[A-Z][a-z]+(?:\s[A-Z][a-z]+)*:)', text.strip())
        chunks = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', section)
            sentences = [s.strip() for s in sentences if s.strip()]
            if not sentences:
                if len(section.split()) <= self.chunk_size:
                    chunks.append(section)
                continue
            current_chunk = []
            current_tokens = 0
            for sentence in sentences:
                sentence_tokens = len(sentence.split())
                if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    overlap_sents = max(1, self.overlap // 25)
                    overlap_chunk = current_chunk[-overlap_sents:] if len(current_chunk) > overlap_sents else []
                    current_chunk = overlap_chunk + [sentence]
                    current_tokens = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        return chunks if chunks else [text]

    def _get_embedding(self, text: str) -> np.ndarray:
        return self._get_transformer_embedding(text)

    def _get_transformer_embedding(self, text: str) -> np.ndarray:
        import torch
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding / (np.linalg.norm(embedding) + 1e-8)

    def retrieve_similar_cases(
        self,
        query: str,
        k: int = 5,
        return_chunks: bool = True,
        similarity_threshold: float = 0.1,
        use_metadata_filters: bool = False,
        apply_lexical_rerank: bool = False
    ) -> List[Dict[str, Any]]:
        if not self.chunks:
            print("⚠️ No medical cases in memory")
            return []

        filters = None
        if use_metadata_filters:
            filters = self._extract_filters_from_query(query)

        query_embedding = self._get_embedding(query).astype('float32').reshape(1, -1)
        pool = max(k * 10, 50)
        D, I = self.faiss_index.search(query_embedding, min(pool, len(self.chunks)))
        similarities = 1 - 0.5 * D[0]
        indices = I[0]

        prelim = []
        for sim, idx in zip(similarities, indices):
            if idx == -1:
                continue
            if sim < similarity_threshold:
                break
            meta = self.chunk_metadata[idx]
            prelim.append({
                "chunk_idx": idx,
                "text": self.chunks[idx],
                "score": float(sim),
                "metadata": meta,
                "doc_id": meta["doc_id"],
                "source": meta["source"]
            })

        if filters:
            prelim = self._apply_metadata_filters(prelim, filters)

        if apply_lexical_rerank and prelim:
            prelim = self._lexical_rerank(query, prelim)

        results = []
        seen = set() if not return_chunks else set()
        for r in sorted(prelim, key=lambda x: x["score"], reverse=True):
            doc_id = r["doc_id"]
            if not return_chunks and doc_id in seen:
                continue
            if not return_chunks:
                seen.add(doc_id)
                text = self.documents[doc_id]
                metadata = self.doc_metadata[doc_id]
            else:
                text = r["text"]
                metadata = r["metadata"]
            out = {
                "text": text,
                "score": float(r["score"]),
                "source": metadata["source"],
                "doc_id": doc_id,
                "rank": len(results) + 1,
                "metadata": metadata
            }
            if return_chunks:
                out["chunk_info"] = {
                    "chunk_index": metadata["chunk_index"],
                    "total_chunks": metadata["total_chunks"],
                    "chunk_id": metadata["chunk_id"]
                }
            results.append(out)
            if len(results) >= k:
                break

        avg_similarity = np.mean([r["score"] for r in results]) if results else 0
        print(f"🔍 Retrieved {len(results)} {'chunks' if return_chunks else 'documents'} (avg similarity: {avg_similarity:.3f})")
        return results

    def _extract_filters_from_query(self, query: str) -> Dict[str, Any]:
        ql = query.lower()
        sys_counts = {sys: len(p.findall(ql)) for sys, p in self._patterns.items()}
        dominant_system = max(sys_counts, key=sys_counts.get) if sys_counts else None
        filters = {"body_system": set(), "symptom_tags": set()}
        if dominant_system and sys_counts[dominant_system] > 0:
            filters["body_system"].add(dominant_system)
        for sys, terms in self._boost_terms.items():
            for t in terms:
                if t.replace("_", " ") in ql or t in ql:
                    filters["symptom_tags"].add(t)
        filters["body_system"] = list(filters["body_system"])
        filters["symptom_tags"] = list(filters["symptom_tags"])
        return filters

    def _apply_metadata_filters(self, candidates: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not candidates:
            return candidates
        want_sys = set(filters.get("body_system", []))
        want_tags = set(filters.get("symptom_tags", []))
        filtered = []
        for r in candidates:
            sys_ok = (not want_sys) or (r["metadata"].get("body_system") in want_sys)
            tags_have = set(r["metadata"].get("symptom_tags", []))
            tags_ok = (not want_tags) or (len(tags_have & want_tags) > 0)
            if sys_ok and tags_ok:
                filtered.append(r)
        if not filtered and want_sys:
            filtered = [r for r in candidates if r["metadata"].get("body_system") in want_sys]
        return filtered if filtered else candidates

    def _lexical_rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ql = query.lower()
        sys_counts = {sys: len(p.findall(ql)) for sys, p in self._patterns.items()}
        dominant_system = max(sys_counts, key=sys_counts.get) if sys_counts else None
        boost_terms = self._boost_terms.get(dominant_system, []) if dominant_system else []
        penalty_terms = []
        if dominant_system == "vascular":
            penalty_terms = self._boost_terms["respiratory"]
        elif dominant_system == "respiratory":
            penalty_terms = self._boost_terms["vascular"]

        def score_adjust(text: str) -> float:
            tl = text.lower()
            boost = sum(1.0 for t in boost_terms if t in tl or t.replace("_", " ") in tl)
            penalty = sum(0.5 for t in penalty_terms if t in tl)
            length_norm = 1.0 / (1.0 + np.log1p(len(tl)))
            return boost * 0.5 - penalty * 0.5 + length_norm

        for r in candidates:
            r["score"] = r["score"] + score_adjust(r["text"])
        return sorted(candidates, key=lambda x: x["score"], reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        if not self.chunks:
            return {"status": "empty", "total_documents": 0, "total_chunks": 0}
        avg_chunk_words = np.mean([meta.get("word_count", 0) for meta in self.chunk_metadata])
        conditions = [meta.get("condition", "unknown") for meta in self.doc_metadata]
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "avg_chunks_per_doc": len(self.chunks) / len(self.documents),
            "avg_chunk_words": round(avg_chunk_words, 1),
            "embedding_dimension": self.embedding_dim,
            "model_name": self.model_name,
            "use_transformers": getattr(self, 'use_transformers', False),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.overlap,
            "similarity_method": "cosine_similarity",
            "sources": list(set(meta["source"] for meta in self.doc_metadata)),
            "conditions": list(set(conditions)),
        }

    def get_document_chunks(self, doc_id: int) -> List[Dict[str, Any]]:
        doc_chunks = []
        for i, meta in enumerate(self.chunk_metadata):
            if meta["doc_id"] == doc_id:
                doc_chunks.append({
                    "chunk_index": meta["chunk_index"],
                    "chunk_id": meta["chunk_id"],
                    "text": self.chunks[i],
                    "word_count": meta.get("word_count", 0),
                    "metadata": meta
                })
        return sorted(doc_chunks, key=lambda x: x["chunk_index"])

    def analyze_similarity_distribution(self, query: str) -> Dict[str, Any]:
        if not self.chunks:
            return {"error": "No chunks available"}
        query_embedding = self._get_embedding(query).astype('float32').reshape(1, -1)
        D, I = self.faiss_index.search(query_embedding, len(self.chunks))
        similarities = 1 - 0.5 * D[0]
        return {
            "query": query,
            "total_chunks": len(similarities),
            "max_similarity": float(np.max(similarities)),
            "min_similarity": float(np.min(similarities)),
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "quartiles": [float(np.percentile(similarities, q)) for q in [25, 50, 75]],
        }

    def save_memory(self, filepath: str = "bioclinical_memory.json"):
        import faiss
        base, ext = os.path.splitext(filepath)
        faiss_path = base + ".faiss"
        faiss.write_index(self.faiss_index, faiss_path)
        save_data = {
            "documents": self.documents,
            "doc_metadata": self.doc_metadata,
            "chunks": self.chunks,
            "chunk_metadata": self.chunk_metadata,
            "faiss_chunk_id_map": self.faiss_chunk_id_map,
            "config": {
                "model_name": self.model_name,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "embedding_dim": self.embedding_dim,
                "use_transformers": getattr(self, 'use_transformers', False)
            },
            "saved_timestamp": datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"💾 Memory saved to {filepath} and FAISS index to {faiss_path}")
        print(f"   Documents: {len(self.documents)}, Chunks: {len(self.chunks)}")

    def load_memory(self, filepath: str = "bioclinical_memory.json"):
        import faiss
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return False
        base, ext = os.path.splitext(filepath)
        faiss_path = base + ".faiss"
        if not os.path.exists(faiss_path):
            print(f"❌ FAISS index file not found: {faiss_path}")
            return False
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.documents = data["documents"]
            self.doc_metadata = data["doc_metadata"]
            self.chunks = data["chunks"]
            self.chunk_metadata = data["chunk_metadata"]
            self.faiss_chunk_id_map = data.get("faiss_chunk_id_map", list(range(len(self.chunks))))
            config = data.get("config", {})
            self.model_name = config.get("model_name", self.model_name)
            self.chunk_size = config.get("chunk_size", self.chunk_size)
            self.overlap = config.get("overlap", self.overlap)
            self.use_transformers = config.get("use_transformers", False)
            self.faiss_index = faiss.read_index(faiss_path)
            print(f"✅ Memory loaded from {filepath} and FAISS index from {faiss_path}")
            print(f"   Documents: {len(self.documents)}, Chunks: {len(self.chunks)}")
            print(f"   Saved: {data.get('saved_timestamp', 'Unknown time')}")
            return True
        except Exception as e:
            print(f"❌ Failed to load memory: {e}")
            return False
