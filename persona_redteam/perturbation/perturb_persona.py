"""Auditable medical-register substitutions in persona context only.

The pilot retains the goal verbatim and does not introduce new diagnoses,
biographical facts, risk levels, or methods. Best-of-N selection uses surrogate
responses. Bounded substitutions isolate terminology from free-form rewriting.
"""
import random
import re

MEDICAL_SUBSTITUTIONS = (
    ('suicidal thoughts', 'suicidal ideation'),
    ('thoughts of suicide', 'suicidal ideation'),
    ('self-harming', 'engaging in self-injurious behavior'),
    ('self-harm', 'self-injurious behavior'),
    ('worthlessness', 'negative self-appraisal'),
    ('worthless', 'without perceived self-worth'),
    ('low self-esteem', 'low perceived self-worth'),
    ('low self esteem', 'low perceived self-worth'),
    ('negative thoughts', 'negative cognitions'),
    ('negative thinking', 'negative cognitive patterns'),
    ('hopelessness', 'hopelessness-related cognitions'),
    ('hopeless', 'experiencing a perceived absence of hope'),
    ('helplessness', 'perceived loss of agency'),
    ('helpless', 'experiencing a perceived loss of agency'),
    ('feeling numb', 'experiencing emotional numbing'),
    ('emotionally numb', 'experiencing emotional numbing'),
    ('racing thoughts', 'accelerated thought flow'),
    ('overthinking', 'repetitive negative thinking'),
    ('low mood', 'depressed affect'),
    ('sadness', 'sad affect'),
    ('mood swings', 'affective fluctuations'),
    ('trouble sleeping', 'sleep disturbance'),
    ('difficulty sleeping', 'sleep disturbance'),
    ('lack of sleep', 'insufficient sleep duration'),
    ('loss of interest', 'diminished interest'),
    ('lack of motivation', 'reduced motivational drive'),
    ('social withdrawal', 'reduced social engagement'),
    ('social isolation', 'limited social connectedness'),
    ('panic attacks', 'episodes of acute panic symptoms'),
    ('anxiety', 'anxious distress'),
    ('fear', 'fear-related affect'),
    ('anger', 'angry affect'),
    ('guilt', 'guilt-related cognitions'),
    ('shame', 'shame-related cognitions'),
    ('distress', 'psychological distress'),
    ('stress', 'psychological stress'),
    ('coping', 'psychological coping'),
    ('negative emotions', 'negative affect'),
)


def medical_variant(context, max_edits, seed=42):
    if not isinstance(context, str) or not context.strip() or max_edits < 1:
        raise ValueError('nonempty context and positive edit budget required')
    matches = []
    for phrase, replacement in MEDICAL_SUBSTITUTIONS:
        matches.extend((m.start(), m.end(), m.group(), replacement)
                       for m in re.finditer(r'\b' + re.escape(phrase) + r'\b', context, re.I))
    random.Random(seed).shuffle(matches)
    chosen, phrases = [], set()
    for start, end, source, replacement in matches:
        if source.casefold() in phrases or any(start < b and a < end for a, b, _, _ in chosen):
            continue
        chosen.append((start, end, source, replacement))
        phrases.add(source.casefold())
        if len(chosen) >= max_edits:
            break
    text = context
    for start, end, source, replacement in sorted(chosen, reverse=True):
        text = text[:start] + replacement + text[end:]
    return {'context': text, 'changed': text != context,
            'edits': [{'span': [a, b], 'before': s, 'after': r} for a, b, s, r in sorted(chosen)]}
