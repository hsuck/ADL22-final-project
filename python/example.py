from utils import prepare_dataset



def main():
    loaders = prepare_dataset.prepare_topic_dataloader(False, "../data", "../cache/vocab", 4 )
    print(loaders)

    for i, data in enumerate(loaders['train']):
        if i == 4: break
        for key, val in data.items():
            print(f"{key}:\n{val}")
        break
    print()

    for i, data in enumerate(loaders['test']):
        if i == 4: break
        for key, val in data.items():
            print(f"{key}: {val.shape}\n{val}")

        break
    print()
        


if __name__ == "__main__":
    main()