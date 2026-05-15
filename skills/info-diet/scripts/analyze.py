#!/usr/bin/env python3
"""지식 영양 성분표 분석기 — Solar LLM으로 글의 정보 가치를 영양소로 수치화."""

import os
import sys
import json
import argparse
from pathlib import Path

try:
    from dotenv import load_dotenv
    from openai import OpenAI
except ImportError:
    print("ERROR: 의존성 누락. 다음을 실행하세요: pip install openai python-dotenv", file=sys.stderr)
    sys.exit(1)

# Load .env from assets directory (path relative to this script's location)
_SKILL_DIR = Path(__file__).parent.parent
load_dotenv(_SKILL_DIR / "assets" / ".env")

UPSTAGE_API_KEY = os.environ.get("UPSTAGE_API_KEY")
if not UPSTAGE_API_KEY:
    print(
        "ERROR: UPSTAGE_API_KEY가 설정되지 않았습니다.\n"
        "  1. assets/.env.example 을 assets/.env 로 복사\n"
        "  2. UPSTAGE_API_KEY=<your-key> 로 수정\n"
        "  키 발급: https://console.upstage.ai → API Keys",
        file=sys.stderr,
    )
    sys.exit(1)

client = OpenAI(api_key=UPSTAGE_API_KEY, base_url="https://api.upstage.ai/v1")

MODE_RULES = {
    "bulkup": (
        "벌크업 모드 (Bulk-up) — 전공 공부/논문 읽기용 깊은 학습\n"
        "- 단백질: 팩트(Data) + 통찰(Insight) 모두\n"
        "- 탄수화물: 어려운 내용을 돕는 비유/배경 스토리\n"
        "- 지방: 의미 없는 잉여 문장만\n"
        "- 트랜스지방: 낚시성 제목, 근거 없는 주장 (항상 적용)"
    ),
    "cutting": (
        "컷팅 모드 (Cutting) — 핵심 수치/결론만 빠르게\n"
        "- 단백질: 결론이 담긴 팩트(Data)만 (통찰도 지방으로)\n"
        "- 탄수화물: 없음 (바쁠 땐 탄수화물도 사치)\n"
        "- 지방: 통찰, 스토리, 잉여 모두 지방\n"
        "- 트랜스지방: 낚시성 제목, 근거 없는 주장 (항상 적용)"
    ),
    "maintainer": (
        "유지어터 모드 (Maintainer) — 출퇴근/휴식 중 가벼운 트렌드 파악\n"
        "- 단백질/탄수화물: 트렌드 인사이트 + 재미있는 비유/스토리\n"
        "- 지방: 과도한 전문 용어, 복잡한 수식, 어려운 데이터\n"
        "- 트랜스지방: 낚시성 제목, 근거 없는 주장 (항상 적용)"
    ),
}

SYSTEM_PROMPT = """당신은 글의 정보 가치를 분석하는 '지식 영양사'입니다.

## 원재료 5가지 유형
- 팩트(Data): 수치, 통계, 코드/알고리즘, 구체적 사실
- 통찰(Insight): 핵심 주장, 새로운 관점, 인과관계 분석
- 스토리(Story): 비유, 배경지식, 가벼운 트렌드 설명
- 잉여(Waste): 앞 내용 단순 반복, 무의미한 긴 서론, 과도한 수식어
- 독성(Toxic): 제목-본문 불일치 낚시, 근거 없는 주장, 논리적 비약

## 핵심 원칙
- 독성(Toxic)은 모드에 무관하게 항상 트랜스지방으로 분류
- 각 문단 번호를 정확히 인용하여 환각(hallucination)을 방지
- 전체 문단 수 기준으로 영양소 비율을 계산 (합계 100%)
- 고단백 문단은 반드시 원문 번호와 핵심 내용을 인용

응답은 지정된 JSON 스키마로만 출력합니다."""

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "nutrition_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "paragraph_num": {"type": "integer"},
                            "raw_type": {"type": "string"},
                            "nutrient": {"type": "string"},
                            "preview": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["paragraph_num", "raw_type", "nutrient", "preview", "reason"],
                        "additionalProperties": False,
                    },
                },
                "protein_pct": {"type": "number"},
                "carbs_pct": {"type": "number"},
                "fat_pct": {"type": "number"},
                "trans_fat_pct": {"type": "number"},
                "high_protein_sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "paragraph_num": {"type": "integer"},
                            "content": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["paragraph_num", "content", "reason"],
                        "additionalProperties": False,
                    },
                },
                "overall_verdict": {"type": "string"},
            },
            "required": [
                "segments",
                "protein_pct",
                "carbs_pct",
                "fat_pct",
                "trans_fat_pct",
                "high_protein_sources",
                "overall_verdict",
            ],
            "additionalProperties": False,
        },
    },
}


def split_paragraphs(text: str) -> list[str]:
    """Double-newline 우선, 없으면 단일 줄바꿈으로 분리."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return paragraphs


def analyze(text: str, mode: str) -> dict:
    mode_rules = MODE_RULES.get(mode)
    if not mode_rules:
        raise ValueError(f"유효하지 않은 모드: {mode}. bulkup / cutting / maintainer 중 선택")

    paragraphs = split_paragraphs(text)
    numbered = "\n\n".join(f"[문단 {i + 1}] {p}" for i, p in enumerate(paragraphs))

    user_msg = (
        f"## 현재 모드\n{mode_rules}\n\n"
        f"## 분석할 글 (총 {len(paragraphs)}개 문단)\n{numbered}\n\n"
        "각 문단을 분류하고 영양 성분 비율(%)을 계산해주세요."
    )

    response = client.chat.completions.create(
        model="solar-pro3",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=RESPONSE_SCHEMA,
        temperature=0.1,
        max_tokens=4096,
    )

    return json.loads(response.choices[0].message.content)


def main():
    parser = argparse.ArgumentParser(
        description="지식 영양 성분표 분석기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "사용 예시:\n"
            "  echo '기사 내용...' | python analyze.py bulkup\n"
            "  python analyze.py cutting --file article.txt\n"
            "  python analyze.py maintainer --file article.txt --json"
        ),
    )
    parser.add_argument(
        "mode",
        choices=["bulkup", "cutting", "maintainer"],
        help="읽기 모드 (bulkup=깊은학습, cutting=빠른필터, maintainer=가벼운읽기)",
    )
    parser.add_argument("--file", help="분석할 텍스트 파일 경로 (기본값: stdin)")
    parser.add_argument("--json", action="store_true", help="JSON 원본 출력 (Claude가 포매팅할 때 사용)")
    args = parser.parse_args()

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8").strip()
    else:
        text = sys.stdin.read().strip()

    if not text:
        print("ERROR: 분석할 텍스트를 입력해주세요 (stdin 또는 --file).", file=sys.stderr)
        sys.exit(1)

    result = analyze(text, args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
