from utils import prepare_dataset



def main():
    for flag in [False, True]:
        loaders = prepare_dataset.prepare_topic_dataloader(False, "../data", "../cache/vocab", 4, with_bos_eos=flag )
        print(loaders)
        for i, (model_input, target_seq) in enumerate(loaders['eval']):
            if i == 4: break
            for key, val in model_input.items():
                print(f"{key}: {val.shape}\n{val}")
            print(f"target: {target_seq.shape}\n{target_seq}")

            break
    print()
        


if __name__ == "__main__":
    main()