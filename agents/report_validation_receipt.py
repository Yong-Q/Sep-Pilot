"""Validate report mode without claiming model review or scientific truth."""
import hashlib,json,re


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def report_validation_satisfied(result,*,strict_required=False):
    if not isinstance(result,dict) or result.get('status')!='authored_verified':return False
    author=result.get('author_call_id');review=result.get('review_call_id')
    valid_id=lambda value:isinstance(value,str) and bool(re.fullmatch(r'[A-Za-z0-9_-]{1,128}',value))
    if not valid_id(author):return False
    if review is not None:
        if not valid_id(review):return False
        mode=result.get('review_mode');performed=result.get('independent_review_performed')
        if mode is None and performed is None:return True  # Historical reviewed receipt.
        return mode=='independent_model' and performed is True
    if strict_required or result.get('review_mode')!='source_numeric' or result.get('independent_review_performed') is not False:
        return False
    proof=result.get('validation_receipt')
    if not isinstance(proof,dict) or set(proof)!={'schema_version','mode','source_packet_sha256','draft_sha256','original_draft_sha256','report_sha256','author_call_id','independent_review_performed','proof_sha256'}:
        return False
    if (type(proof.get('schema_version')) is not int or proof.get('schema_version')!=1 or proof.get('mode')!='source_numeric'
            or proof.get('independent_review_performed') is not False
            or proof.get('author_call_id')!=result.get('author_call_id')
            or proof.get('source_packet_sha256')!=result.get('source_packet_sha256')
            or proof.get('draft_sha256')!=result.get('draft_sha256')
            or proof.get('original_draft_sha256')!=result.get('original_draft_sha256')
            or proof.get('report_sha256')!=result.get('report_sha256')):
        return False
    if any(not isinstance(proof.get(k),str) or not re.fullmatch(r'[a-f0-9]{64}',proof[k]) for k in ('source_packet_sha256','draft_sha256','original_draft_sha256','report_sha256','proof_sha256')):
        return False
    return proof['proof_sha256']==_digest({k:v for k,v in proof.items() if k!='proof_sha256'})
