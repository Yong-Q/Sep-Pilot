"""Conservative recognition of explicit user review and content requirements."""
import re


def _user_texts(goal):
    if not isinstance(goal,dict):return []
    original=goal.get('original_goal');directives=goal.get('directives') or []
    if not isinstance(directives,list) or any(not isinstance(v,str) for v in directives):
        raise ValueError('invalid user report directives')
    return ([original] if isinstance(original,str) else [])+directives


def explicit_strict_review(goal):
    desired=False
    # The user may place modifiers before or after review. A bounded clause
    # and explicit cancellation are required; uncertainty retains strict mode.
    target=(r'(?:严格|独立).{0,40}(?:审查|审核|审稿|复核|评审|核验|检查)'
        r'|(?:审查|审核|审稿|复核|评审|核验|检查).{0,40}(?:严格|独立)'
        r'|\b(?:independent(?:ly)?|strict(?:ly)?|rigorous(?:ly)?)\b.{0,40}\b(?:review|audit)\b'
        r'|\b(?:review|audit)\b.{0,40}\b(?:independent(?:ly)?|strict(?:ly)?|rigorous(?:ly)?)\b')
    cancellation=(r'(?:不需要|无需|不用|不必|取消|关闭|不要)\s*(?:进行|使用|开启)?\s*(?:严格|独立)(?:的|地|仔细|科学|结果|报告|同行|专业){0,6}(?:审查|审核|审稿|复核|评审|核验|检查)'
        r'|\b(?:no|skip|disable)\s+(?:strict\s+)?independent\s+review\b')
    for text in _user_texts(goal):
        if len(text)>200000:raise ValueError('user review directive exceeds bound')
        for clause in re.split(r'[。！？!?；;，,\n]|\b(?:but|however|and)\b|但是|然而|而是|但|不过|并且|并|且',text.lower()):
            matches=list(re.finditer(target,clause,re.I))
            if not matches:continue
            negatives=list(re.finditer(cancellation,clause,re.I))
            desired=not all(any(n.start()<=m.start() and n.end()>=m.end() for n in negatives) for m in matches)
    return desired


def explicit_explanatory_content(goal):
    """Conservatively retain an explicitly requested discussion or explanation."""
    target=r'解释|说明|解读|讨论|分析|论证|\b(?:explain|interpret|discuss|analy[sz]e|discussion|explanation)\b'
    cancellation=r'(?<!不要)(?:不需要|无需|不用|不必|不要|不做)\s*(?:解释|说明|解读|讨论|分析|论证)|\b(?:no|skip)\s+(?:explanation|discussion|analysis)\b'
    for text in _user_texts(goal):
        if len(text)>200000:raise ValueError('user explanation directive exceeds bound')
        remaining=re.sub(cancellation,'',text,flags=re.I)
        if re.search(target,remaining,re.I):return True
    return False


def explicit_source_only(goal):
    """Only a positive, unambiguous user directive waives narrative."""
    target=(r'(?:只要|仅需|只需|只输出|仅输出|仅提供|只提供)(?:来源|原始|结构化)?(?:数据|表格|事实|数值)'
        r'|\b(?:only|just)\s+(?:(?:output|provide)\s+)?(?:the\s+)?(?:source\s+)?(?:data|tables|facts)\b')
    negation=r'(?:不要|不能|不应|拒绝|不是|并非|不许|不可|\bnot\b|\bdon\x27t\b|\bdo not\b)\s*$'
    desired=False
    for text in _user_texts(goal):
        if len(text)>200000:raise ValueError('user output directive exceeds bound')
        if not text.strip():continue
        # A later meaningful user directive may revoke an earlier constraint.
        # Do not carry a source-only waiver across an unparsed new turn.
        desired=False
        for clause in re.split(r'[。！？!?；;，,\n]|\b(?:but|however)\b|但是|然而|而是|但|不过',text.lower()):
            if not clause.strip():continue
            matches=list(re.finditer(target,clause,re.I))
            if not matches:
                # A subsequent unknown clause can revoke the waiver. Only an
                # explicit no-explanation clause is compatible with source-only.
                if not re.fullmatch(r'(?:不需要|无需|不用|不必|不要|不做)\s*(?:解释|说明|解读|讨论|分析|论证)|(?:no|skip)\s+(?:explanation|discussion|analysis)',clause.strip(),re.I):desired=False
                continue
            for match in matches:
                prefix=clause[:match.start()].strip();suffix=clause[match.end():].strip()
                neutral=r'(?:(?:请|请您|最后|本次|本报告|结果)\s*)*'
                desired=bool(re.fullmatch(neutral,prefix,re.I) and not suffix
                    and not re.search(negation,prefix,re.I))
    return desired
