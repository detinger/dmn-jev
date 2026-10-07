import httpx, json, sys, time
Q = [l.strip() for l in open(sys.argv[1]) if l.strip()]
res = {}
with httpx.Client(timeout=40, headers={"User-Agent": "jevdmn-refcheck (mailto:darko.etinger@unipu.hr)"}) as c:
    for q in Q:
        for attempt in range(4):
            try:
                r = c.get("https://api.crossref.org/works", params={"query.bibliographic": q, "rows": 3, "select": "DOI,title,container-title,issued,author,type,volume,page"})
                items = r.json()["message"]["items"]; break
            except Exception as e:
                time.sleep(2 + 2*attempt); items = []
        res[q] = [{"doi": it["DOI"], "title": (it.get("title") or [""])[0], "venue": (it.get("container-title") or [""])[0],
                   "year": it.get("issued",{}).get("date-parts",[[None]])[0][0], "authors": [a.get("family","") for a in it.get("author",[])[:4]],
                   "type": it.get("type"), "volume": it.get("volume"), "page": it.get("page")} for it in items]
json.dump(res, open(sys.argv[2], "w"), indent=1)
