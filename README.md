# mental_jail — persona red-team evaluation

현재 유지하는 연구 코드는 [`persona_redteam/`](persona_redteam/)에 있다.

- 전체 설계·데이터 선정·실행 방법: [`persona_redteam/README.md`](persona_redteam/README.md)
- 2026-10-08 확정 결과와 진행 상황: [`persona_redteam/docs/STATUS_2026-10-08.md`](persona_redteam/docs/STATUS_2026-10-08.md)
- 다음 공격 페르소나 생성기 설정: [`persona_redteam/configs/llama33_70b_abliterated.env.example`](persona_redteam/configs/llama33_70b_abliterated.env.example)
- 70B 생성기 서빙 스크립트: [`persona_redteam/scripts/serve_persona_generator_70b.sh`](persona_redteam/scripts/serve_persona_generator_70b.sh)

모델 역할은 분리한다. `huihui-ai/Llama-3.3-70B-Instruct-abliterated`는 공격
페르소나와 다음 내담자 턴을 만드는 생성기이고, 상담 응답 서로게이트는 고정된
`meta-llama/Llama-3.1-8B-Instruct`, PCSA 평가기와 최종 전이 대상은 고정된
GPT-4o-mini다.

원시 상담 응답, API 키, 로컬 `.env`, 다운로드한 모델 가중치, 실행 로그와
원문 데이터 payload는 저장소에 포함하지 않는다. 공개 가능한 코드, 프롬프트
출처, 데이터 체크섬, 집계 결과와 재현 절차만 버전 관리한다.
