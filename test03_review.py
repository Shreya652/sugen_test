"""Doctor/judge review workflow for the Test03 ICD coding experiment."""

import csv
import json
import sys

from score import norm


def load_catalog(path):
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"code", "title", "definition", "text"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing required ICD CSV columns")
        catalog = {}
        for line_number, row in enumerate(reader, 2):
            if None in row or any(row.get(field) is None for field in required):
                raise ValueError(f"{path}:{line_number}: incomplete CSV row")
            code = norm(row["code"])
            if code:
                catalog.setdefault(code, []).append(
                    {
                        "code": row["code"],
                        "title": row["title"],
                        "definition": row["definition"],
                        "text": row["text"],
                    }
                )
    return catalog


def related_catalog_entries(code, catalog, valid_codes):
    family = code.split(".", 1)[0][:4]
    entries = []
    for related_code in sorted(catalog):
        if related_code == code or related_code.split(".", 1)[0][:4] != family:
            continue
        for entry in catalog[related_code]:
            entries.append(
                {
                    **entry,
                    "normalized_code": related_code,
                    "valid_mms": related_code in valid_codes,
                }
            )
    return entries


def review_note(
    note_text,
    catalog,
    valid_codes,
    doctor_prompt,
    judge_prompt,
    call_doctor,
    call_judge,
    max_retries=3,
):
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    attempts = []
    last_eligible_codes = []

    for number in range(1, max_retries + 2):
        doctor_input = {
            "clinical_note": note_text,
            "previous_attempts": [
                {"codes": attempt["codes"], "feedback": attempt["feedback"]}
                for attempt in attempts
            ],
        }
        prediction = call_doctor(doctor_prompt, json.dumps(doctor_input, ensure_ascii=False))
        if (
            not isinstance(prediction, dict)
            or not isinstance(prediction.get("codes"), list)
            or any(not isinstance(code, str) for code in prediction["codes"])
        ):
            raise ValueError('Doctor response must contain a "codes" list of strings.')
        proposed = [norm(code) for code in prediction["codes"]]
        issues = []
        if len(proposed) != 3:
            issues.append("Provide exactly three ranked codes.")
        eligible = []
        seen = set()
        for position, code in enumerate(proposed[:3], 1):
            if not code or code not in valid_codes:
                issues.append(f"Rank {position} is not a valid MMS category.")
            elif code in seen:
                issues.append(f"Rank {position} duplicates another code.")
            elif code not in catalog:
                issues.append(f"Rank {position} is absent from the provided ICD CSV.")
            else:
                eligible.append(code)
            if code:
                seen.add(code)
        if eligible:
            last_eligible_codes = eligible
        if issues:
            attempts.append(
                {
                    "attempt": number,
                    "codes": proposed,
                    "approved": False,
                    "status": "invalid_candidates",
                    "feedback": " ".join(issues),
                }
            )
            continue

        families = []
        seen_families = set()
        for code in eligible:
            family = code.split(".", 1)[0][:4]
            if family in seen_families:
                continue
            seen_families.add(family)
            entries = [
                {**entry, "normalized_code": code, "valid_mms": True}
                for entry in catalog[code]
            ]
            entries.extend(related_catalog_entries(code, catalog, valid_codes))
            families.append({"stem_prefix": family, "entries": entries})
        judge_input = {
            "clinical_note": note_text,
            "candidate_codes": eligible,
            "icd_code_families": families,
        }
        judgement = call_judge(judge_prompt, json.dumps(judge_input, ensure_ascii=False))
        if (
            not isinstance(judgement, dict)
            or not isinstance(judgement.get("approved"), bool)
            or not isinstance(judgement.get("feedback"), str)
        ):
            raise ValueError('Judge response must contain a boolean "approved" and string "feedback".')
        approved = judgement["approved"]
        feedback = judgement["feedback"].strip()
        if not approved and not feedback:
            raise ValueError("Judge must provide feedback when rejecting a candidate.")
        attempts.append(
            {
                "attempt": number,
                "codes": eligible,
                "approved": approved,
                "status": "approved" if approved else "rejected",
                "feedback": feedback,
            }
        )
        if approved:
            return {"codes": eligible, "approved": True, "complete": True, "attempts": attempts}

    return {
        "codes": last_eligible_codes,
        "approved": False,
        "complete": len(last_eligible_codes) == 3,
        "attempts": attempts,
    }
