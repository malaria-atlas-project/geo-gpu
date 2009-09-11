# Copyright (C) 2009  Bernhard Seiser and Anand Patil
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import re

end_token = re.compile('{{\s*endif\s*}}')

def add_line_numbers(s):
    l = s.splitlines()
    ln = []
    for i in xrange(len(l)):
        ln.append('%i\t%s'%(i+2,l[i]))
    return '\n'.join(ln)

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
