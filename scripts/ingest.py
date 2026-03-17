"""
scripts/ingest.py
─────────────────
Downloads the CUAD legal contract dataset from Hugging Face and applies
recursive character-aware chunking to produce data/processed/chunks.jsonl.

Chunking Strategy: Recursive Character Splitting
─────────────────────────────────────────────────
Legal documents have natural structure: sections → paragraphs → sentences.
We exploit this hierarchy by attempting splits in order:
  1. Double newline  (\n\n) — paragraph boundaries
  2. Single newline  (\n)   — line boundaries
  3. Sentence end    (". ") — sentence boundaries
  4. Word boundary   (" ")  — last resort

This preserves semantic coherence far better than fixed-size splitting,
avoiding scenarios where a critical legal clause is split mid-sentence.

Overlap: 100 characters between consecutive chunks ensures context
continuity at boundaries (a clause's conclusion provides context
for the next chunk's beginning).

Output Format (chunks.jsonl):
  {"doc_id": "contract_001", "chunk_id": "contract_001-0", "text": "..."}
  {"doc_id": "contract_001", "chunk_id": "contract_001-1", "text": "..."}
  ...

Usage:
  python scripts/ingest.py
  python scripts/ingest.py --max-docs 50   # for quick testing
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512    # target chunk size in characters
CHUNK_OVERLAP = 100    # overlap between consecutive chunks
MIN_CHUNK_LEN = 50     # discard chunks shorter than this
MAX_DOCS      = 200    # default maximum documents to process

OUTPUT_PATH = Path("data/processed/chunks.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── Recursive Character Splitter ───────────────────────────────────────────────

def recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text using a hierarchy of separators for semantic coherence.

    Args:
        text:       Input text to split.
        chunk_size: Target maximum characters per chunk.
        overlap:    Number of characters to overlap between chunks.

    Returns:
        List of text chunks.
    """
    separators = ["\n\n", "\n", ". ", " "]

    def _split(text: str, separators: list[str]) -> list[str]:
        if not separators or len(text) <= chunk_size:
            return [text] if text.strip() else []

        sep = separators[0]
        parts = text.split(sep)

        chunks = []
        current = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            candidate = (current + sep + part).strip() if current else part

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                # Save current chunk if non-empty
                if current:
                    chunks.append(current)
                    # Start new chunk with overlap from end of previous
                    overlap_text = current[-overlap:] if len(current) > overlap else current
                    current = (overlap_text + sep + part).strip()
                else:
                    # Single part exceeds chunk_size — recurse with next separator
                    sub_chunks = _split(part, separators[1:])
                    chunks.extend(sub_chunks[:-1])
                    current = sub_chunks[-1] if sub_chunks else ""

        if current:
            chunks.append(current)

        return chunks

    return _split(text, separators)


def clean_text(text: str) -> str:
    """Clean legal document text."""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove page numbers (e.g., "- 12 -" or "Page 12")
    text = re.sub(r"-\s*\d+\s*-|Page\s+\d+", "", text, flags=re.IGNORECASE)
    # Normalize quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text.strip()


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_cuad(max_docs: int) -> list[dict]:
    """
    Load CUAD contracts from Hugging Face datasets.
    Falls back to raw txt files in data/raw/ if download fails.

    Returns:
        List of {"doc_id": str, "text": str} dicts.
    """
    documents = []

    print("[Ingest] Attempting to load CUAD from Hugging Face...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)

        seen_titles = {}
        for row in dataset:
            title = row.get("title", "unknown")
            if title not in seen_titles:
                seen_titles[title] = []
            seen_titles[title].append(row.get("context", ""))

        for i, (title, contexts) in enumerate(seen_titles.items()):
            if len(documents) >= max_docs:
                break
            full_text = "\n\n".join(contexts)
            if len(full_text) < MIN_CHUNK_LEN:
                continue
            safe_title = re.sub(r"[^a-zA-Z0-9_-]", "_", title)[:50]
            doc_id = f"cuad_{i:04d}_{safe_title}"
            documents.append({"doc_id": doc_id, "text": full_text})

        print(f"[Ingest] ✅ Loaded {len(documents)} contracts from CUAD")

    except Exception as e:
        print(f"[Ingest] ⚠️  CUAD download failed: {e}")
        print("[Ingest] Falling back to data/raw/ directory...")
        documents = load_from_raw_dir(max_docs)

    return documents


def load_from_raw_dir(max_docs: int) -> list[dict]:
    """Load .txt files from data/raw/ as fallback."""
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    documents = []

    txt_files = list(raw_dir.glob("*.txt"))[:max_docs]
    if not txt_files:
        print("[Ingest] ⚠️  No .txt files found in data/raw/")
        print("[Ingest] Generating synthetic legal documents for demonstration...")
        documents = generate_synthetic_docs(max_docs)
        return documents

    for i, fpath in enumerate(txt_files):
        text = fpath.read_text(encoding="utf-8", errors="ignore")
        doc_id = f"doc_{i:04d}_{fpath.stem[:40]}"
        documents.append({"doc_id": doc_id, "text": text})

    print(f"[Ingest] ✅ Loaded {len(documents)} documents from data/raw/")
    return documents


def generate_synthetic_docs(n: int) -> list[dict]:
    """
    Generate synthetic legal contract documents for demonstration.
    Used only when real data is unavailable.
    """
    templates = [
        {
            "title": "Software License Agreement",
            "text": """SOFTWARE LICENSE AGREEMENT

This Software License Agreement ("Agreement") is entered into as of the Effective Date
between the Licensor and the Licensee.

1. GRANT OF LICENSE
Subject to the terms and conditions of this Agreement, Licensor hereby grants to Licensee
a non-exclusive, non-transferable, limited license to use the Software solely for Licensee's
internal business purposes.

2. RESTRICTIONS
Licensee shall not: (a) sublicense, sell, resell, transfer, assign, or otherwise dispose
of the Software; (b) modify or make derivative works based upon the Software; (c) reverse
engineer or access the Software in order to build a competitive product.

3. TERM AND TERMINATION
This Agreement shall commence on the Effective Date and continue for one (1) year unless
earlier terminated. Either party may terminate this Agreement with thirty (30) days written
notice. Upon termination, Licensee shall cease all use of the Software and destroy all copies.

4. INTELLECTUAL PROPERTY
The Software and all copies thereof are proprietary to Licensor and title thereto remains
in Licensor. Licensee acknowledges that no title to the intellectual property in the Software
is transferred to Licensee.

5. CONFIDENTIALITY
Each party agrees to hold the other's Confidential Information in strict confidence and not
to disclose such Confidential Information to third parties without prior written consent.

6. LIMITATION OF LIABILITY
IN NO EVENT SHALL LICENSOR BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR IN CONNECTION WITH THIS AGREEMENT.

7. GOVERNING LAW
This Agreement shall be governed by and construed in accordance with the laws of the
State of Delaware, without regard to its conflict of law provisions.

8. INDEMNIFICATION
Licensee shall indemnify and hold harmless Licensor from any claims, damages, losses,
liabilities, costs and expenses arising out of Licensee's use of the Software or breach
of this Agreement."""
        },
        {
            "title": "Non-Disclosure Agreement",
            "text": """NON-DISCLOSURE AGREEMENT

This Non-Disclosure Agreement ("Agreement") is entered into between the Disclosing Party
and the Receiving Party (collectively, the "Parties").

1. DEFINITION OF CONFIDENTIAL INFORMATION
"Confidential Information" means any information disclosed by either party to the other
party, either directly or indirectly, in writing, orally or by inspection of tangible objects,
that is designated as "Confidential," "Proprietary" or some similar designation.

2. OBLIGATIONS OF RECEIVING PARTY
The Receiving Party agrees to: (a) hold the Confidential Information in strict confidence;
(b) not to disclose the Confidential Information to any third parties without prior written
consent of the Disclosing Party; (c) use the Confidential Information solely for evaluating
a potential business relationship between the parties.

3. EXCLUSIONS FROM CONFIDENTIAL INFORMATION
The obligations of confidentiality shall not apply to information that: (a) is or becomes
publicly known through no fault of the Receiving Party; (b) was rightfully known before
receipt from the Disclosing Party; (c) is independently developed without use of Confidential
Information; (d) is required to be disclosed by law or court order.

4. TERM
The obligations under this Agreement shall survive for a period of five (5) years from the
date of disclosure of the Confidential Information.

5. REMEDIES
The parties acknowledge that breach of this Agreement may cause irreparable harm for which
monetary damages would be inadequate, and therefore equitable relief, including injunction,
may be appropriate.

6. RETURN OF INFORMATION
Upon request of the Disclosing Party, the Receiving Party shall promptly return or destroy
all Confidential Information and certify in writing that it has done so.

7. GOVERNING LAW
This Agreement shall be governed by the laws of the State of New York."""
        },
        {
            "title": "Service Agreement",
            "text": """PROFESSIONAL SERVICES AGREEMENT

This Professional Services Agreement ("Agreement") is entered into between the Client
and the Service Provider.

1. SERVICES
Service Provider agrees to perform the services described in each Statement of Work
("SOW") executed by both parties. Each SOW shall describe the services, deliverables,
timeline, and compensation.

2. PAYMENT TERMS
Client shall pay Service Provider within thirty (30) days of receipt of invoice.
Late payments shall accrue interest at the rate of 1.5% per month. Service Provider
may suspend services for non-payment after providing fifteen (15) days written notice.

3. INTELLECTUAL PROPERTY OWNERSHIP
All work product, inventions, discoveries, and improvements created by Service Provider
in connection with the services shall be considered works made for hire and shall be
owned exclusively by Client upon full payment.

4. WARRANTIES
Service Provider warrants that: (a) services will be performed in a professional manner;
(b) Service Provider has the right to enter into this Agreement; (c) services will not
infringe any third party intellectual property rights.

5. INDEMNIFICATION
Each party shall indemnify and hold harmless the other party from claims arising from
its own negligence, willful misconduct, or breach of this Agreement.

6. LIMITATION OF LIABILITY
In no event shall either party be liable for indirect, incidental, or consequential
damages. Each party's total liability shall not exceed the fees paid in the preceding
three months.

7. TERMINATION
Either party may terminate this Agreement for convenience upon thirty (30) days written
notice, or immediately for material breach that remains uncured after fifteen (15) days notice.

8. DISPUTE RESOLUTION
Any dispute arising out of this Agreement shall be resolved through binding arbitration
in accordance with the American Arbitration Association rules."""
        },
        {
            "title": "Employment Agreement",
            "text": """EMPLOYMENT AGREEMENT

This Employment Agreement ("Agreement") is entered into between the Employer and Employee.

1. POSITION AND DUTIES
Employee shall serve in the position of Senior Software Engineer and shall perform such
duties as are customarily associated with such position and as assigned by management.

2. COMPENSATION
Base Salary: Employee shall receive an annual base salary of [AMOUNT], payable in
accordance with Employer's standard payroll practices. Performance bonuses may be awarded
at the discretion of management based on performance objectives.

3. BENEFITS
Employee shall be entitled to participate in all employee benefit plans, including health
insurance, dental insurance, 401(k) plan, and paid time off in accordance with Employer's
standard policies.

4. AT-WILL EMPLOYMENT
Employment is at-will and may be terminated by either party at any time, with or without
cause, upon two weeks written notice. Employer may pay salary in lieu of notice period.

5. CONFIDENTIALITY AND PROPRIETARY INFORMATION
Employee agrees to maintain the confidentiality of all proprietary information and trade
secrets of Employer during and after employment. Employee shall not use or disclose such
information for any purpose other than performing duties for Employer.

6. NON-COMPETE
During employment and for twelve (12) months thereafter, Employee shall not engage in
any business activity that directly competes with Employer's principal business activities
within the same geographic market.

7. INVENTIONS ASSIGNMENT
Employee agrees to assign to Employer all inventions, discoveries, and improvements
made during employment that relate to Employer's business or result from work performed.

8. GOVERNING LAW
This Agreement shall be governed by the laws of the state where Employee primarily works."""
        },
        {
            "title": "Lease Agreement",
            "text": """COMMERCIAL LEASE AGREEMENT

This Commercial Lease Agreement ("Lease") is entered into between the Landlord and Tenant.

1. PREMISES
Landlord hereby leases to Tenant the premises located at [ADDRESS], consisting of
approximately [SQUARE FOOTAGE] square feet of commercial space ("Premises").

2. TERM
The lease term shall commence on the Commencement Date and expire on the Expiration Date,
unless sooner terminated pursuant to the terms hereof.

3. BASE RENT
Tenant shall pay monthly base rent in the amount of [AMOUNT]. Rent shall be due and payable
on the first day of each calendar month. A late fee of 5% shall apply to payments received
after the fifth day of the month.

4. SECURITY DEPOSIT
Upon execution of this Lease, Tenant shall deposit with Landlord a security deposit equal
to two months' rent. The security deposit shall be held as security for Tenant's performance
and returned within thirty (30) days after lease expiration.

5. USE OF PREMISES
Tenant shall use the Premises solely for general office and business purposes and for no
other purpose without prior written consent of Landlord.

6. MAINTENANCE AND REPAIRS
Tenant shall maintain the Premises in good condition and repair. Landlord shall be responsible
for structural repairs, roof, and common areas. Tenant shall be responsible for interior
maintenance and its own equipment.

7. INSURANCE
Tenant shall maintain commercial general liability insurance with limits of not less than
$1,000,000 per occurrence and $2,000,000 aggregate, and shall name Landlord as additional insured.

8. DEFAULT AND REMEDIES
Tenant shall be in default if: (a) rent is unpaid for five days after due; (b) Tenant fails
to perform any covenant after ten days written notice. Upon default, Landlord may terminate
this Lease and recover possession of the Premises."""
        },
    ]

    documents = []
    for i in range(n):
        template = templates[i % len(templates)]
        variation = i // len(templates)
        doc_id = f"synthetic_{i:04d}_{template['title'].replace(' ', '_')}"
        text = template["text"]
        if variation > 0:
            text = f"AMENDMENT {variation}\n\n" + text
        documents.append({"doc_id": doc_id, "text": text})

    print(f"[Ingest] ✅ Generated {len(documents)} synthetic legal documents")
    return documents


# ── Main Ingestion Pipeline ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest legal documents and produce chunks.jsonl"
    )
    parser.add_argument(
        "--max-docs", type=int, default=MAX_DOCS,
        help=f"Maximum number of documents to process (default: {MAX_DOCS})"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help=f"Target chunk size in characters (default: {CHUNK_SIZE})"
    )
    parser.add_argument(
        "--overlap", type=int, default=CHUNK_OVERLAP,
        help=f"Overlap between chunks in characters (default: {CHUNK_OVERLAP})"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Legal RAG Pipeline — Data Ingestion")
    print("=" * 60)
    print(f"  Max documents : {args.max_docs}")
    print(f"  Chunk size    : {args.chunk_size} chars")
    print(f"  Overlap       : {args.overlap} chars")
    print(f"  Output        : {OUTPUT_PATH}")
    print("=" * 60)

    # Step 1: Load documents
    documents = load_cuad(args.max_docs)

    if not documents:
        print("[Ingest] ❌ No documents loaded. Exiting.")
        sys.exit(1)

    # Step 2: Chunk and write
    total_chunks = 0
    skipped = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for doc in tqdm(documents, desc="Chunking documents"):
            doc_id = doc["doc_id"]
            text   = clean_text(doc["text"])

            if len(text) < MIN_CHUNK_LEN:
                skipped += 1
                continue

            chunks = recursive_split(text, args.chunk_size, args.overlap)

            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_text = chunk_text.strip()
                if len(chunk_text) < MIN_CHUNK_LEN:
                    continue

                record = {
                    "doc_id":   doc_id,
                    "chunk_id": f"{doc_id}-{chunk_idx}",
                    "text":     chunk_text,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print("=" * 60)
    print(f"  ✅ Done!")
    print(f"  Documents processed : {len(documents) - skipped}")
    print(f"  Documents skipped   : {skipped}")
    print(f"  Total chunks        : {total_chunks}")
    print(f"  Output file         : {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
