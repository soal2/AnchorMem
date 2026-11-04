"""Fixed information types shared by entry extraction and directory retrieval."""

TOPICS = {
    'profile': 'The user\'s identity, background and current personal state.',
    'preferences': 'The user\'s global inclinations and preferred choices, rather than explicit requirements.',
    'entities': 'Facts about other people, organizations, projects and objects.',
    'events': 'Events, plans, commitments, changes, progress and outcomes grouped by the same matter.',
    'requirements': 'Explicit conditions the user requires within a stated applicable scope.',
    'methods': 'Reusable procedures, solutions and failure lessons, with applicable conditions and verification.',
}


def candidates():
    return [dict(id=key, path=f'topics/{key}', abstract=value, overview=value) for key,value in TOPICS.items()]


def selected_topics(decisions, identities):
    from .directory_selector import CRITERIA
    from .providers import validate_choice_probabilities

    if (len(decisions) != len(identities) or {d['id'] for d in decisions} != set(identities)
            or any(d['choice'] not in CRITERIA for d in decisions)):
        raise ValueError('topic selection returned invalid candidate decisions')
    for d in decisions:
        validate_choice_probabilities(d['choice'], d['probabilities'], CRITERIA)
    return [d['id'] for d in decisions if d['probabilities']['relevant'] >= .5
            or d['probabilities']['uncertain'] >= .5]
