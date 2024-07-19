import mlflow


def validate(cfg=None):
    registered_models = mlflow.registered_model.list_registered_models()
    for i, model in enumerate(registered_models):
        model_versions = mlflow.search_registered_model_versions(f"name='{model.name}'")
        for version in model_versions:
            mlflow.registered_model.add_alias(
                name=model.name,
                version=version.version,
                alias=f'challenger{i}'
            )



