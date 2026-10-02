"""
День 24. Проверка ответов с источниками и цитатами.

10 вопросов с ответом в базе (8 по проектам + 2 общих):
  - есть ли источники в каждом ответе (source + section + chunk_id);
  - есть ли цитаты в каждом ответе и все ли они найдены во фрагментах дословно;
  - совпадает ли смысл ответа с цитатами: ключевые элементы ответа есть в цитатах
    + вердикт LLM-судьи (supported / partial / unsupported);
  - полнота: обязательные факты из eval_questions.json.
4 вопроса для режима «не знаю» (2 без ответа в базе + 2 расплывчатых):
  - сказал ли ассистент «не знаю» и попросил ли уточнение; на каком этапе остановился.

Запуск:
    python evaluate.py                 # модель по умолчанию, эмбеддинги AITunnel
    python evaluate.py --model gpt-5.6-luna
    python evaluate.py --no-judge      # без LLM-судьи (дешевле)
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime

from grounded import grounded_answer
from rag import BASE, LLM, load_index
from retrieval import load_config
from verify import judge, lexical_grounding

ANSWERABLE, UNKNOWN = ("project", "general"), ("absent", "vague")


def facts(text: str, must: list) -> float:
    low = text.lower()
    return sum(any(v.lower() in low for v in g) for g in must) / len(must) if must else 1.0


def pct(a, b) -> str:
    return f"{a}/{b}" + (f" ({a / b:.0%})" if b else "")


def main(args) -> int:
    with open(os.path.join(BASE, "eval_questions.json"), encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    cfg, llm = load_config(), LLM(args.model)
    index = load_index(args.backend, "structure")
    th = cfg["thresholds"][args.backend]
    if "answer_min" not in th:
        raise SystemExit(f"Нет порога «не знаю» для {args.backend}. Сначала: python calibrate.py --backend {args.backend}")
    print(f"Модель: {llm.model} · эмбеддинги: {index.meta['embedder']['label']} · "
          f"порог «не знаю»: {th['answer_min']} · фильтр: {th['min_score']} / {th['rel_ratio']}\n")

    rows = []
    for q in questions:
        print(f"[{q['id']:>2}/{len(questions)}] {q['q'][:64]}…", flush=True)
        g = grounded_answer(llm, index, q["q"], cfg, args.backend)
        row = {"id": q["id"], "type": q["type"], "q": q["q"], "expected": q["expected"],
               "expected_sources": q["sources"], **asdict(g)}
        row.pop("chunks", None)
        if q["type"] in ANSWERABLE:
            row["has_sources"] = bool(g.sources)
            row["has_quotes"] = bool(g.quotes)
            row["right_source"] = (any(e in s["source"] for s in g.sources for e in q["sources"])
                                   if q["sources"] else None)
            row["facts"] = round(facts(g.answer, q["must"]), 2)
            row["lexical"] = lexical_grounding(g.answer, g.quotes) if g.quotes else None
            row["judge"] = (judge(llm, q["q"], g.answer, g.quotes) if g.quotes and not args.no_judge else None)
            j = row["judge"]["verdict"] if row["judge"] else "—"
            lex = f"{row['lexical']['score']:.0%}" if row["lexical"] and row["lexical"]["score"] is not None else "—"
            print(f"        {g.status:<10} источников {len(g.sources)}, цитат {len(g.quotes)}"
                  f" (отброшено {len(g.rejected_quotes)}), попыток {g.attempts} · элементы в цитатах {lex} · судья: {j}")
        else:
            row["said_unknown"] = g.status != "answered"
            row["asked_clarification"] = bool(g.clarification) and "?" in g.clarification
            print(f"        {'не знаю ✅' if row['said_unknown'] else 'ОТВЕТИЛ ❌'} · этап: {g.reason or '—'} · "
                  f"близость {g.top_score} vs порог {g.gate} · уточнение: {'да' if row['asked_clarification'] else 'нет'}")
        rows.append(row)

    ans = [r for r in rows if r["type"] in ANSWERABLE]
    unk = [r for r in rows if r["type"] in UNKNOWN]
    answered = [r for r in ans if r["status"] == "answered"]
    all_quotes = sum(len(r["quotes"]) for r in ans)
    rejected = sum(len(r["rejected_quotes"]) for r in ans)
    judged = [r for r in ans if r["judge"]]
    lex = [r["lexical"]["score"] for r in ans if r["lexical"] and r["lexical"]["score"] is not None]
    with_src = [r for r in ans if r["right_source"] is not None]
    usage = {k: sum(r["usage"][k] for r in rows) for k in ("prompt_tokens", "completion_tokens")}
    jusage = {k: sum(r["judge"]["usage"][k] for r in judged) for k in usage}

    summary = {
        "Выдан ответ (не «не знаю»)": pct(len(answered), len(ans)),
        "Из выданных ответов — с источниками": pct(sum(r["has_sources"] for r in answered), len(answered)),
        "Из выданных ответов — с цитатами": pct(sum(r["has_quotes"] for r in answered), len(answered)),
        "Нужный файл среди источников": pct(sum(bool(r["right_source"]) for r in with_src), len(with_src)),
        "Цитат принято (дословно найдены)": all_quotes,
        "Цитат отброшено (не найдены дословно)": rejected,
        "Ответов, исправленных со 2-й попытки": sum(r["attempts"] == 2 and r["status"] == "answered" for r in ans),
        "Ключевые элементы ответа есть в цитатах": f"{sum(lex) / len(lex):.0%}" if lex else "—",
        "Судья: подтверждён / частично / не подтверждён": (
            " / ".join(str(sum(r["judge"]["verdict"] == v for r in judged)) for v in ("supported", "partial", "unsupported"))
            if judged else "—"),
        "Обязательные факты в ответе": f"{sum(r['facts'] for r in ans) / len(ans):.0%}",
        "«Не знаю» на вопросах с ответом (модель / не подтверждено цитатами / порог)": " / ".join(
            str(sum(r["reason"] == k for r in ans)) for k in ("model", "unverified", "gate")),
        "«Не знаю» на вопросах без ответа / расплывчатых": pct(sum(r["said_unknown"] for r in unk), len(unk)),
        "…с уточняющим вопросом": pct(sum(r["asked_clarification"] for r in unk), len(unk)),
        "…остановлено до модели (нет такого дня / порог близости)": pct(
            sum(r["reason"] in ("gate", "no_day") for r in unk), len(unk)),
        "Токены ответов: вход / выход": f"{usage['prompt_tokens']} / {usage['completion_tokens']}",
        "Токены судьи: вход / выход": f"{jusage['prompt_tokens']} / {jusage['completion_tokens']}",
    }
    print(f"\n{'=' * 72}\nИТОГ\n{'=' * 72}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    os.makedirs(os.path.join(BASE, "output"), exist_ok=True)
    md = os.path.join(BASE, "output", f"grounded_eval_{args.backend}.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(render(rows, summary, llm.model, index, th))
    with open(os.path.splitext(md)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nОтчёт с ответами, источниками и цитатами: {os.path.relpath(md, BASE)}")
    return 0


def render(rows, summary, model, index, th) -> str:
    out = [f"# Ответы с источниками, цитатами и режимом «не знаю»\n\n"
           f"Модель: `{model}` · эмбеддинги: `{index.meta['embedder']['label']}` · "
           f"порог «не знаю»: `{th['answer_min']}` · {datetime.now():%Y-%m-%d %H:%M}\n\n## Итог\n\n| Метрика | Значение |\n|---|---|"]
    out += [f"| {k} | {v} |" for k, v in summary.items()]
    out.append("\n## Вопросы с ответом\n\n| # | Статус | Источники | Цитаты | Элементы в цитатах | Судья | Факты |\n"
               "|---|---|---|---|---|---|---|")
    for r in rows:
        if r["type"] in ANSWERABLE:
            lex = f"{r['lexical']['score']:.0%}" if r["lexical"] and r["lexical"]["score"] is not None else "—"
            j = r["judge"]["verdict"] if r["judge"] else "—"
            out.append(f"| {r['id']} | {r['status']} | {len(r['sources'])} | {len(r['quotes'])} "
                       f"(−{len(r['rejected_quotes'])}) | {lex} | {j} | {r['facts']:.0%} |")
    out.append("\n## Режим «не знаю»\n\n| # | Тип | Сказал «не знаю» | Этап | Близость / порог | Уточнение |\n|---|---|---|---|---|---|")
    for r in rows:
        if r["type"] in UNKNOWN:
            out.append(f"| {r['id']} | {r['type']} | {'✅' if r['said_unknown'] else '❌'} | {r['reason'] or '—'} | "
                       f"{r['top_score']} / {r['gate']} | {'✅' if r['asked_clarification'] else '❌'} |")
    out.append("\n## Ответы целиком\n")
    for r in rows:
        out.append(f"### {r['id']}. {r['q']}\n\n**Ожидание:** {r['expected']}\n\n**Ответ** ({r['status']}): {r['answer']}\n")
        if r["sources"]:
            out.append("**Источники:**\n" + "\n".join(
                f"- [{s['n']}] `{s['source']}` — {s['section']} (`{s['chunk_id']}`)" for s in r["sources"]))
        if r["quotes"]:
            out.append("\n**Цитаты:**\n" + "\n".join(f"- [{q['n']}] «{q['text']}»" for q in r["quotes"]))
        if r["rejected_quotes"]:
            out.append("\n**Отброшенные цитаты:**\n" + "\n".join(
                f"- [{q['n']}] «{q['text'][:120]}» — {q['why']}" for q in r["rejected_quotes"]))
        if r.get("lexical") and r["lexical"]["unsupported"]:
            out.append(f"\nНе найдено в цитатах: {', '.join(map(str, r['lexical']['unsupported']))}")
        if r.get("judge"):
            out.append(f"\n**Судья:** {r['judge']['verdict']} — {r['judge']['comment']}")
        if r["clarification"]:
            out.append(f"\n**Уточнение:** {r['clarification']}")
        if r["raw_answer"]:
            out.append(f"\n**Исходный ответ модели (заменён):** {r['raw_answer']}")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Проверка ответов с источниками и цитатами")
    ap.add_argument("--model", help="модель AITunnel")
    ap.add_argument("--backend", choices=["aitunnel", "local"], default="aitunnel")
    ap.add_argument("--no-judge", action="store_true", help="без LLM-судьи")
    sys.exit(main(ap.parse_args()))
