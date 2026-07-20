from factory_dataset import create_adapter
def main():
    # Basta cambiare il main per utilizzare un altro dataset
    NOME_DATASET = "nuscenes"

    DATAROOT = "./nuscenes"

    adapter = create_adapter(NOME_DATASET,DATAROOT)

if __name__ == "__main__":
    main()