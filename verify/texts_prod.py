import json, urllib.request, hashlib
P = "A train leaves at 60 mph. Two hours later a second train leaves the same station at 90 mph on a parallel track. At what distance does the second catch the first? Reason step by step before answering."
for i in (1,2,3):
    body = json.dumps({"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":P}],"max_tokens":512,"temperature":0,"top_p":0.95}).encode()
    r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=300)
    d = json.load(r)
    m = d["choices"][0]["message"]
    txt = (m.get("reasoning_content") or "") + "\n<<<SPLIT>>>\n" + (m.get("content") or "")
    h = hashlib.md5(txt.encode()).hexdigest()[:12]
    open("/tmp/texts_prod_%d.txt" % i,"w").write(txt)
    print("prod-%d: tokens=%d md5=%s" % (i, d["usage"]["completion_tokens"], h))
