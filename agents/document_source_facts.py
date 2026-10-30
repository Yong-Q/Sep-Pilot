"""Hash original scientific documents without loading bytes into a packet."""

MAX_DOCUMENT_BYTES=32*1024**2
MAX_DOCUMENTS=64
MAX_STREAM_BYTES=MAX_DOCUMENTS*MAX_DOCUMENT_BYTES
DOCUMENT_VERIFICATION='native_document_stream_sha256'


def stream_document_fact(root,path,budget):
    from .native_source_stream import stream_owned_source
    stream={'source_files':budget.get('documents',0),'source_bytes':budget.get('document_bytes',0)}
    prefix,fact=stream_owned_source(root,path,stream,max_file_bytes=MAX_DOCUMENT_BYTES,
        max_total_bytes=MAX_STREAM_BYTES,max_files=MAX_DOCUMENTS,verification=DOCUMENT_VERIFICATION)
    budget['documents']=stream['source_files'];budget['document_bytes']=stream['source_bytes']
    return prefix,fact
