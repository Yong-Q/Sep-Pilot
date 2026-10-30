"""Fixed document-text parser. Never executes document actions or user code."""
import json
import resource
import sys
sys.dont_write_bytecode=True

resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2))
resource.setrlimit(resource.RLIMIT_CPU,(10,10))
resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))

# The server supplies one resolved installed-package directory. It is never a
# model argument or document path; -I still excludes ambient cwd/PYTHONPATH.
if len(sys.argv)>4 and sys.argv[4]:
    from pathlib import Path
    package_directory=Path(sys.argv[4])
    if not package_directory.is_absolute()or package_directory.resolve()!=package_directory:raise ValueError('invalid native parser package directory')
    sys.path.insert(0,str(package_directory))


def parse(raw,start,maximum,limit):
    import io
    import re
    if b'%PDF-'in raw[:1024]:
        from pypdf import PdfReader
        import pypdf
        reader=PdfReader(io.BytesIO(raw),strict=False)
        if reader.is_encrypted:raise ValueError('encrypted PDF requires another authorized source; no password bypass')
        total=len(reader.pages)
        if not 1<=start<=total:raise ValueError('requested PDF page is outside document')
        blocks=[];count=0;truncated=False
        for index in range(start-1,min(total,start-1+maximum)):
            text=reader.pages[index].extract_text()or''
            remaining=limit-count
            if len(text)>remaining:text=text[:remaining];truncated=True
            blocks.append({'anchor':'page-'+str(index+1),'page':index+1,'text':text})
            count+=len(text)
            if count>=limit:break
        last=blocks[-1]['page'];next_page=last+1 if last<total else None
        return {'ok':True,'status':'text_extracted','document_type':'pdf','pagination_unit':'page',
            'total_pages':total,'start_page':start,'pages':blocks,'next_page':next_page,
            'truncated':truncated or last<total,'page_text_truncated':truncated,
            'title':str((reader.metadata or{}).get('/Title')or'')[:1000],
            'parser':{'name':'pypdf','version':pypdf.__version__},
            'extractable_characters':count,'image_only_or_ocr_required':count==0,
            'links':[],'embedded_actions_executed':False}
    head=raw[:8192].decode('utf-16'if raw.startswith((b'\xff\xfe',b'\xfe\xff'))else'utf-8',errors='replace')
    if not re.search(r'(?is)<(?:!doctype\s+html|html\b|head\b|body\b|article\b)',head):
        raise ValueError('source is neither an actual PDF nor an HTML document')
    from html.parser import HTMLParser
    class Document(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True);self.skip=0;self.parts=[];self.blocks=[];self.title=[];self.in_title=False;self.links=[];self.link=None
        def flush(self):
            text=' '.join(''.join(self.parts).split());self.parts=[]
            if text:
                if len(self.blocks)>=10000:raise ValueError('HTML block index exceeds bound')
                self.blocks.append(text)
        def handle_starttag(self,tag,attrs):
            if tag in{'script','style','noscript','svg','template','canvas'}:self.skip+=1;return
            if self.skip:return
            values=dict(attrs)
            if tag=='title':self.in_title=True
            if tag in{'p','div','article','section','li','h1','h2','h3','h4','h5','h6','tr','br','blockquote','pre'}:self.flush()
            if tag=='a'and values.get('href'):
                self.link={'href':values['href'][:2048],'text':''}
        def handle_endtag(self,tag):
            if tag in{'script','style','noscript','svg','template','canvas'}:
                self.skip=max(0,self.skip-1);return
            if self.skip:return
            if tag=='title':self.in_title=False
            if tag=='a'and self.link:
                if len(self.links)<200:self.links.append(self.link)
                self.link=None
            if tag in{'p','div','article','section','li','h1','h2','h3','h4','h5','h6','tr','blockquote','pre'}:self.flush()
        def handle_data(self,data):
            if self.skip:return
            self.parts.append(data)
            if self.in_title:self.title.append(data)
            if self.link:self.link['text']=(self.link['text']+data)[:500]
    import codecs
    encoding='utf-8-sig'
    declared=re.search(r'''(?i)charset\s*=\s*["']?([A-Za-z0-9_-]+)''',head)
    if raw.startswith((codecs.BOM_UTF16_LE,codecs.BOM_UTF16_BE)):encoding='utf-16'
    elif declared:
        name=codecs.lookup(declared[1]).name
        if name not in{'utf-8','utf-8-sig','iso8859-1','cp1252','gb18030','gb2312','big5','shift_jis','ascii','utf-16','utf-16-le','utf-16-be'}:
            raise ValueError('unsupported declared HTML encoding')
        encoding=name
    document=Document();document.feed(raw.decode(encoding));document.flush()
    total=len(document.blocks)
    if not total or not 1<=start<=total:raise ValueError('requested HTML block is outside extractable document')
    blocks=[];count=0;truncated=False
    for index in range(start-1,min(total,start-1+maximum)):
        text=document.blocks[index];remaining=limit-count
        if len(text)>remaining:text=text[:remaining];truncated=True
        blocks.append({'anchor':'html-block-'+str(index+1),'block':index+1,'text':text});count+=len(text)
        if count>=limit:break
    last=blocks[-1]['block']
    return {'ok':True,'status':'text_extracted','document_type':'html','pagination_unit':'block',
        'total_blocks':total,'start_block':start,'blocks':blocks,'next_block':last+1 if last<total else None,
        'title':' '.join(document.title)[:1000],'links':document.links,
        'truncated':truncated or last<total,'block_text_truncated':truncated,
        'extractable_characters':count,'embedded_actions_executed':False}


if __name__=='__main__':
    try:
        raw=sys.stdin.buffer.read(32*1024**2+1)
        if len(raw)>32*1024**2:raise ValueError('document byte limit exceeded')
        result=parse(raw,*map(int,sys.argv[1:4]))
    except BaseException as error:
        result={'error':str(error)[:1000],'error_kind':'bounded_document_parser','embedded_actions_executed':False}
    sys.stdout.write(json.dumps(result,ensure_ascii=False,allow_nan=False))
