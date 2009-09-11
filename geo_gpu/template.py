import re

end_token = re.compile('{{\s*endif\s*}}')

def next_match(t, s, e):
    m_start = s.search(t)
    if m_start:
        return m_start, e.search(t, m_start.end())
    else:
        return None, None

def tif(t,kv):
    s = re.compile('{{\s*if\s*%s\s*}}'%kv[0])
    value = bool(kv[1])
    e=end_token
    while True:
        m_start, m_end = next_match(t, s, e)
        if m_start and m_end:
            if value:
                t = t[:m_start.start()] + t[m_start.end():m_end.start()] + t[m_end.end():]
            else:
                t = t[:m_start.start()] + t[m_end.end():]
        else:
            break
    return t

def tsubs(t,kv): 
    return re.compile('{{\s*%s\s*}}'%kv[0]).sub(str(kv[1]), t)

    
def templ_subs(t, **kwds):
    ts = reduce(tsubs, kwds.iteritems(), t)
    ti = reduce(tif, kwds.iteritems(), ts)
    return ti
