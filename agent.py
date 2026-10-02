"""
День 24. Агент: ответы с обязательными источниками и цитатами, «не знаю» при слабом контексте.

    python agent.py                                   # диалог
    python agent.py --ask "Какие слои памяти в дне 11?"

Команды в диалоге:  /trace — как отработал поиск для последнего вопроса;  /exit
"""

import argparse
import sys

from grounded import grounded_answer, render
from rag import LLM, load_index
from retrieval import load_config


def show_trace(g) -> None:
    print(f"  лучшая близость {g.top_score} · порог «не знаю» {g.gate}")
    if g.trace:
        t = g.trace
        print(f"  запросы: {' ‖ '.join(t.get('queries', []))}")
        print(f"  кандидатов {t['candidates']} → после порога {t['after_filter']} → в контексте {t['final']}")
    for q in g.rejected_quotes:
        print(f"  ✗ отброшена цитата к [{q['n']}]: «{q['text'][:80]}» — {q['why']}")
    print(f"  попыток ответа: {g.attempts}")


def main(args) -> None:
    llm, cfg = LLM(args.model), load_config()
    index = load_index(args.backend, "structure")
    if "answer_min" not in cfg["thresholds"][args.backend]:
        raise SystemExit(f"Нет порога «не знаю». Сначала: python calibrate.py --backend {args.backend}")
    print(f"Модель: {llm.model} · база: {index.meta['chunks']} чанков · "
          f"порог «не знаю»: {cfg['thresholds'][args.backend]['answer_min']}")

    if args.ask:
        print("\n" + render(grounded_answer(llm, index, args.ask, cfg, args.backend)))
        return

    last = None
    print("Команды: /trace, /exit\n")
    while True:
        try:
            q = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q == "/exit":
            break
        if q == "/trace":
            show_trace(last) if last else print("Пока не было вопросов.")
            continue
        last = grounded_answer(llm, index, q, cfg, args.backend)
        print("\n" + render(last) + "\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Агент с источниками, цитатами и режимом «не знаю»")
    ap.add_argument("--ask", help="задать один вопрос и выйти")
    ap.add_argument("--model", help="модель AITunnel")
    ap.add_argument("--backend", choices=["aitunnel", "local"], default="aitunnel")
    main(ap.parse_args())
