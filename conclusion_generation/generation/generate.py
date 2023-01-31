import argparse
import pathlib
import pandas as pd
import torch

from dataloader import CustomDataset
from torch import cuda
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration


device = "cuda" if cuda.is_available() else "cpu"


# This code is a quickly adapted notebook, here to serve just as a hint
# The values are hardcoded to match our needs
# Happy to merge a PR making it more efficient
def _create_predictions(df, model, tokenizer):
    val_set = CustomDataset(df, tokenizer, 512, 128)

    val_params = {"batch_size": 4, "shuffle": False, "num_workers": 0}
    val_loader = DataLoader(val_set, **val_params)
    predictions = []

    for data in val_loader:
        model.eval()
        with torch.no_grad():
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)
            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=25,
                min_length=4,
                do_sample=True,
                num_beams=4,
                top_k=20,
                temperature=0.75,
                no_repeat_ngram_size=2,
                repetition_penalty=1.25,
            )
            preds = [
                tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for g in generated_ids
            ]
            predictions.extend(preds)

    final_df = pd.DataFrame(predictions)

    return final_df


def main():
    tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=512)

    df = pd.read_csv(args.data_path, encoding="latin-1")
    df_1 = df["sentence_1"]
    df_2 = df["sentence_2"]
    df_1.sentence = "summarize: " + df.sentence_1
    df_2.sentence = "summarize: " + df.sentence_2

    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = model.to(device)
    if args.conclusions:
        model.load_state_dict(
            torch.load("../fine_tuning/models/conclusion_generation_model.pth")
        )
    print("Model loaded")

    print("Generating for sentence 1")
    final_df_1 = _create_predictions(df_1, model, tokenizer)
    print("Generating for sentence 2")
    final_df_2 = _create_predictions(df_2, model, tokenizer)

    final_df_1.columns = ["sentence_1"]
    final_df_2.columns = ["sentence_2"]

    print("Creating joined file")
    new_df = final_df_1.merge(final_df_2, left_index=True, right_index=True)
    if args.summaries:
        new_df.to_csv("Summaries_" + args.data_path.stem + ".csv")
    if args.conclusions:
        new_df.to_csv("Conclusions_" + args.data_path.stem + ".csv")

    print("Generation finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=pathlib.Path,
        help="Path to data",
        required=True,
    )
    parser.add_argument(
        "--summaries",
        action="store_true",
        help="Generate summaries",
    )
    parser.add_argument(
        "--conclusions",
        action="store_true",
        help="Generate conclusions",
    )

    args = parser.parse_args()

    main()
