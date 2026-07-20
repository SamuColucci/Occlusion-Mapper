# Utilizziamo un factory pattern per creare l'adapter, in modo da poterlo facilmente estendere a futuri dataset
def create_adapter(dataset_name: str, dataroot: str):
    if dataset_name == "nuscenes":
        from nuscenes_dataset_adapter import NuscenesDatasetAdapter
        return NuscenesDatasetAdapter(dataroot)