# test_search_api.py

import requests

BASE_URL = "http://localhost:8000"

FILTERS = {
    "must": [
        {
            "key": "person_id",
            "match": {"value": "pal_0000"}
        }
    ]
}


def test_query_search():
    print("\n===== TEST: search/query =====")

    payload = {
        "text": "我最近胃有点难受，怎么办？",
        "limit": 5,
        "filters": FILTERS,
    }

    r = requests.post(
        f"{BASE_URL}/search/query",
        json=payload,
        timeout=30,
    )
    r.raise_for_status()

    data = r.json()
    for i, item in enumerate(data["results"], 1):
        payload = item["payload"] or {}
        print(
            f"{i}. score={item['score']:.4f}, "
            f"person_id={payload.get('person_id')}, "
            f"topic={payload.get('topic')}, "
            f"timestamp={payload.get('timestamp')}"
        )


def test_topic_search():
    print("\n===== TEST: search/topic =====")

    payload = {
        "text": "健康问题",
        "limit": 5,
        "filters": FILTERS,
    }

    r = requests.post(
        f"{BASE_URL}/search/topic",
        json=payload,
        timeout=30,
    )
    r.raise_for_status()

    data = r.json()
    for i, item in enumerate(data["results"], 1):
        payload = item["payload"] or {}
        print(
            f"{i}. score={item['score']:.4f}, "
            f"person_id={payload.get('person_id')}, "
            f"topic={payload.get('topic')}, "
            f"timestamp={payload.get('timestamp')}"
        )


if __name__ == "__main__":
    test_query_search()
    test_topic_search()
