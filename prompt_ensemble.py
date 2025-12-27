def build_prompts(object_name):
    return [
        f"a photo of {object_name} with scratch defect",
        f"a photo of {object_name} with broken defect",
        f"a photo of {object_name} with bent defect",
        f"a photo of perfect {object_name}",
    ]
