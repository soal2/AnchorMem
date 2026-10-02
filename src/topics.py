"""Fixed, multi-label parent topics shared by writing and retrieval."""
from .directory_selector import CRITERIA
from .providers import validate_choice_probabilities

TOPICS = {
    'education': 'Degrees, university studies, courses, graduation and educational history.',
    'work': 'Employment, employers, job roles, career changes and workplace projects.',
    'health': 'Health, allergies, dietary preferences, exercise and medical precautions.',
    'travel': 'Trips, destinations, itineraries, travel bookings and journey plans.',
    'finance': 'Budgets, savings, expenses, payments and financial plans.',
    'family': 'Relationships, relatives, friends, family events and responsibilities.',
    'hobbies': 'Music, food, gardening, photography and other leisure interests and preferences.',
    'housing': 'Residences, moving home, rent, household items and home maintenance.',
    'other': 'Personal information and topics not covered by the other eight branches.',
}


def candidates():
    return [dict(id=key, path=f'topics/{key}', abstract=value, overview=value) for key,value in TOPICS.items()]


def selected_topics(decisions, identities):
    if (len(decisions) != len(identities) or {d['id'] for d in decisions} != set(identities)
            or any(d['choice'] not in CRITERIA for d in decisions)):
        raise ValueError('topic selection returned invalid candidate decisions')
    for d in decisions:
        validate_choice_probabilities(d['choice'], d['probabilities'], CRITERIA)
    return [d['id'] for d in decisions if d['probabilities']['relevant'] >= .5
            or d['probabilities']['uncertain'] >= .5]
