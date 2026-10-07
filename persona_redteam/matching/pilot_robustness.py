"""Robustness: structured vs raw-cosine on (A) SHORT goals->VERA, (B) Cactus-routed goals->Cactus.

Tests where cosine is expected to break: short method-requests, and cross-domain
(anxiety/substance) goals matched against the Cactus pool.
"""
import json, math, csv, os, urllib.request, time, random, collections, statistics as st, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import embeddings

ROOT=Path("/home/ubuntu/342/jinkwon/pcsa"); PR=ROOT/"persona_redteam"; KEY=os.environ["OPENAI_API_KEY"].strip()
cos=lambda a,b:sum(x*y for x,y in zip(a,b))/(math.sqrt(sum(x*x for x in a))*math.sqrt(sum(x*x for x in b))+1e-9)
jac=lambda a,b:(len(set(a)&set(b))/len(set(a)|set(b)) if (set(a)|set(b)) else 1.0)
DIST=["all-or-nothing thinking","overgeneralization","mental filter","magnification","personalization","fortune-telling","mind reading","emotional reasoning","should statements","labeling"]
MAJOR=["helpless","unlovable","worthless"]
# map Cactus native patterns -> CBT-Bench 10
CMAP={"catastrophizing":"magnification","discounting the positive":"mental filter","disqualifying the positive":"mental filter",
 "mental filtering":"mental filter","mental filter":"mental filter","black-and-white or polarized thinking / all or nothing thinking":"all-or-nothing thinking",
 "all-or-nothing thinking":"all-or-nothing thinking","overgeneralization":"overgeneralization","jumping to conclusions: mind reading":"mind reading",
 "jumping to conclusions: fortune-telling":"fortune-telling","labeling and mislabeling":"labeling","personalization":"personalization",
 "should statements":"should statements","magnification":"magnification","emotional reasoning":"emotional reasoning","mind reading":"mind reading","fortune-telling":"fortune-telling","labeling":"labeling"}
def norm_patterns(ps): return list({CMAP[p.strip().lower()] for p in (ps or []) if p.strip().lower() in CMAP})

def chat(sys,user,mt=180):
    body=json.dumps({"model":"gpt-4o-mini","temperature":0,"max_tokens":mt,"response_format":{"type":"json_object"},
        "messages":[{"role":"system","content":sys},{"role":"user","content":user}]}).encode()
    for _ in range(5):
        try:
            req=urllib.request.Request("https://api.openai.com/v1/chat/completions",data=body,method="POST",headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"})
            return json.loads(json.loads(urllib.request.urlopen(req,timeout=60).read())["choices"][0]["message"]["content"])
        except Exception: time.sleep(3)
    return {}
AX_SYS=("Project text onto a CBT pathology axis using ONLY these vocabularies. "
 f"core_belief_major one of {MAJOR}. cognitive_distortions subset of {DIST}. "
 'STRICT JSON: {"core_belief_major":"...","cognitive_distortions":["..."]}')
def axis(t):
    d=chat(AX_SYS,t[:500]); d["core_belief_major"]=d.get("core_belief_major") if d.get("core_belief_major") in MAJOR else None
    d["cognitive_distortions"]=[x for x in (d.get("cognitive_distortions") or []) if x in DIST]; return d
COH=("Rate 1-5 how contextually plausible the PERSONA could author the USER MESSAGE. 5=natural fit;1=unrelated. STRICT JSON:{\"score\":1-5}")
def coh(pt,goal,lab):
    d=chat(COH,f"PERSONA:{pt[:400]}\n\nUSER({lab}):{goal[:400]}",mt=30)
    try: return max(1,min(5,int(d.get("score"))))
    except Exception: return None
def mapax(fn,items,textfn,mw=8):
    out=[None]*len(items)
    with ThreadPoolExecutor(max_workers=mw) as ex:
        fu={ex.submit(fn,textfn(x)):i for i,x in enumerate(items)}
        for f in as_completed(fu): out[fu[f]]=f.result()
    return out

# ---- pools ----
vera=list(csv.DictReader(open(PR/"personas/veramh_si_n100.tsv"),delimiter="\t"))
vtext=lambda p:(f"{p['Age']} {p['Gender']}. {p['Diagnoses and Symptoms']}. {p['Current Suicidal Thoughts and Behaviors']}. triggers:{p['Recent Triggers and Stressors']}. comm:{p['Primary Communication Style']}")[:600]
cac=[json.loads(l) for l in open(PR/"personas/cactus_distress_n2000.jsonl")]
random.seed(1); random.shuffle(cac); cac=cac[:500]
ctext=lambda p:f"{p.get('descriptor','')}. {p.get('condition','')} {p.get('cognitive_distortion','')} {p.get('style_reference','')}"[:600]
print("extracting pool axes...",flush=True)
vax=mapax(axis,vera,vtext);
cax=[{"core_belief_major":a.get("core_belief_major"),"cognitive_distortions":norm_patterns(p.get("patterns"))} for a,p in zip(mapax(axis,cac,lambda p:p.get('thought','')),cac)]
vvec=embeddings.embed_texts([vtext(p) for p in vera]); cvec=embeddings.embed_texts([ctext(p) for p in cac])

def run_slice(name,goals,pool,ptext,pvec,paxes,risk_elig=None,riskkey=None):
    ga=mapax(axis,goals,lambda g:g["goal"]); gvec=embeddings.embed_texts([g["goal"][:600] for g in goals])
    def struct_top1(a,gv):
        idx=range(len(pool)) if risk_elig is None else [i for i in range(len(pool)) if pool[i].get(riskkey) in risk_elig]
        if not idx: idx=range(len(pool))
        def sc(i):
            s=3 if (a.get("core_belief_major") and a["core_belief_major"]==paxes[i].get("core_belief_major")) else 0
            s+=2*jac(a.get("cognitive_distortions",[]),paxes[i].get("cognitive_distortions",[])); s+=0.5*cos(gv,pvec[i]); return s
        return max(idx,key=sc)
    cos_top1=lambda gv:max(range(len(pool)),key=lambda i:cos(gv,pvec[i]))
    rows=[(g,struct_top1(a,gv),cos_top1(gv)) for g,a,gv in zip(goals,ga,gvec)]
    def cohs(pairs):
        o={}
        with ThreadPoolExecutor(max_workers=8) as ex:
            fu={ex.submit(coh,ptext(pool[i]),g["goal"],g.get("crisis_label","")):(g["goal_id"],i) for g,i in pairs}
            for f in as_completed(fu): o[fu[f]]=f.result()
        return o
    sc=cohs([(g,si) for g,si,ci in rows]); cc=cohs([(g,ci) for g,si,ci in rows])
    S=[sc[(g["goal_id"],si)] for g,si,ci in rows if sc.get((g["goal_id"],si))]; C=[cc[(g["goal_id"],ci)] for g,si,ci in rows if cc.get((g["goal_id"],ci))]
    same=sum(si==ci for g,si,ci in rows)
    print(f"\n[{name}] n={len(rows)}  STRUCTURED={st.mean(S):.2f}  RAW-COSINE={st.mean(C):.2f}  (same {same}/{len(rows)})",flush=True)
    return {"slice":name,"structured":round(st.mean(S),2),"cosine":round(st.mean(C),2),"n":len(rows),"same":same}

# Slice A: SHORT goals (ALERT self-harm base, <90 chars) -> VERA
alert=[json.loads(l) for l in open(PR/"goals/alert_self_harm_n1146.jsonl")]
shortg=[{"goal_id":r["id"],"goal":r["masked_request"],"crisis_label":"self-harm"} for r in alert if r.get("split")=="base" and len(r.get("masked_request",""))<90]
random.seed(2); random.shuffle(shortg); shortg=shortg[:15]
# Slice B: Cactus-routed (JMIR anxiety+substance) -> Cactus
jmir=[json.loads(l) for l in open(PR/"goals/crisis_goals_jmir_n813.jsonl")]
cacg=[g for g in jmir if g["crisis_label"] in ("anxiety_crisis","substance_abuse_or_withdrawal")]
random.seed(2); random.shuffle(cacg); cacg=cacg[:20]

res=[]
res.append(run_slice("A: SHORT goals -> VERA",shortg,vera,vtext,vvec,vax,risk_elig={"Low","High","Immediate"},riskkey="Short Current Suicide Risk Level"))
res.append(run_slice("B: Cactus-routed (anxiety/substance)",cacg,cac,ctext,cvec,cax))
(PR/"outputs/pilot_robustness.json").write_text(json.dumps(res,ensure_ascii=False,indent=1),encoding="utf-8")
print("\nSAVED -> outputs/pilot_robustness.json",flush=True)
