import json
import requests
import hydra
import zenml
import argparse


def load_features(name, version, random_state):
    try:
        data = zenml.load_artifact(name_or_id="features_target", version=version)
    except Exception as e:
        print("Error loading zenml artifacts\n")
        raise e
    ...
    data = data.sample(frac=1, random_state=random_state)
    X = data.drop(columns=["price"])
    y = data["price"]
    return X, y


def predict(cfg=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--example_version", type=int, default=1)
    parser.add_argument("--port", type=int, default=5152)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    X, y = load_features(
        name="features_target",
        version=args.example_version,
        random_state=args.random_state,
    )
    example = X.iloc[0, :]
    example_target = y[0]

    example = json.dumps({"inputs": example.to_dict()})

    payload = example

    response = requests.post(
        url=f"http://localhost:{args.port}/invocations",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    print(response.json())
    print("encoded target labels: ", example_target)
    # print("target labels: ", list(cfg.data.labels)[example_target])


if __name__ == "__main__":
    predict()
