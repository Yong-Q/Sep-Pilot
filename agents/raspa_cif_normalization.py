"""Deterministic RASPA aliases on CIF tag tokens; values stay byte-for-byte."""
ALIASES=(('_space_group_name_H-M_alt','_symmetry_space_group_name_H-M'),
    ('_space_group_IT_number','_symmetry_Int_Tables_number'),
    ('_space_group_symop_operation_xyz','_symmetry_equiv_pos_as_xyz'))


def _tokens(text):
    """Yield CIF1 tokens with source spans, excluding comments and text bodies."""
    size=len(text);i=0
    while i<size:
        if text[i].isspace():i+=1;continue
        if text[i]=='#':
            end=text.find('\n',i);i=size if end<0 else end+1;continue
        start=i
        if text[i]==';'and(i==0 or text[i-1]=='\n'):
            end=text.find('\n;',i+1)
            if end<0:raise ValueError('unterminated CIF semicolon text')
            i=end+2
            yield '',start,i,True
            continue
        if text[i]in("'",'"'):
            quote=text[i];i+=1
            while i<size:
                if text[i]in'\r\n':raise ValueError('unterminated CIF quoted token')
                if text[i]==quote and(i+1==size or text[i+1].isspace()or text[i+1]=='#'):
                    i+=1;break
                i+=1
            else:raise ValueError('unterminated CIF quoted token')
            yield text[start:i],start,i,True
            continue
        while i<size and not text[i].isspace():i+=1
        yield text[start:i],start,i,False


def normalize_raspa_cif_bytes(raw):
    """Only replace actual scalar/loop-header tag tokens, never data strings.

    This does not add missing scientific information or rewrite CIF geometry,
    coordinates, occupancies, charges or symmetry operations. Syntax/geometry
    admission remains the caller's real CIF parser; malformed lexical forms
    fail rather than being repaired by guessing. Original line endings remain.
    """
    if not isinstance(raw,bytes):raise TypeError('CIF normalization requires raw bytes')
    text=raw.decode('utf-8')
    if '\r' in text.replace('\r\n',''):
        raise ValueError('standalone CR CIF line endings require explicit normalization; original preserved')
    aliases={a.lower():b for a,b in ALIASES}
    changes=[];mode='normal';columns=0;values=0;need_value=False
    for token,start,end,quoted in _tokens(text):
        lower=token.lower()
        if mode=='loop_headers':
            if not quoted and token.startswith('_'):
                columns+=1
                if lower in aliases:changes.append((start,end,aliases[lower]))
                continue
            if columns==0:raise ValueError('CIF loop has no columns')
            mode='loop_values';values=0
        if mode=='loop_values':
            control=not quoted and(token.startswith('_')or lower=='loop_'or lower=='stop_'or lower.startswith(('data_','save_')))
            if not control:
                values+=1;continue
            if values%columns:raise ValueError('incomplete CIF loop row')
            mode='normal';columns=values=0
            if lower=='stop_':continue
        if need_value:
            need_value=False;continue
        if not quoted and lower=='loop_':mode='loop_headers';continue
        if not quoted and token.startswith('_'):
            if lower in aliases:changes.append((start,end,aliases[lower]))
            need_value=True
    if need_value or mode=='loop_headers'or mode=='loop_values'and values%columns:
        raise ValueError('incomplete CIF field or loop')
    for start,end,value in reversed(changes):text=text[:start]+value+text[end:]
    return text.encode('utf-8')
