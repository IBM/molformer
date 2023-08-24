import torch
from .utils import normalize_smiles
import os
from fast_transformers.masking import FullMask


def get_full_attention(molecule, bert_model, config, tokenizer):
    loader = None
    device = config.device
    if loader is not None:
        for batch_number, mols in enumerate(loader):
            batch_to_save = []
            with torch.no_grad():
                # print(batch_number)
                if config.canonical is True:
                    output = [
                        normalize_smiles(smiles, canonical=True, isomeric=False)
                        for smiles in mols["text"]
                        if smiles is not None
                    ]
                else:
                    output = mols["text"]
                batch_ids = tokenizer.batch_encode_plus(
                    output,
                    padding=True,
                    add_special_tokens=True,
                    return_attention_mask=True,
                    return_length=True,
                )

                if config.mask is True:
                    att_mask = FullMask(
                        torch.tensor(batch_ids["attention_mask"], dtype=bool).to(
                            device
                        ),
                        device=device,
                    )
                else:
                    att_mask = FullMask(
                        torch.ones(
                            torch.tensor(batch_ids["input_ids"]).size(), dtype=bool
                        ).to(device),
                        device=device,
                    )

                embeddings, attention_mask = bert_model(
                    torch.tensor(batch_ids["input_ids"]).to(device),
                    att_mask,
                    mode=config.mode,
                )

            for number, mol in enumerate(output):
                batch_to_save.append((embeddings[number].cpu().numpy(), mol))

            # if len(batch_to_save) >= 500:
            batch_name = "batch_num_{}.pth".format(
                batch_number + (50000 * config.chunk_num)
            )
            chunk_name = "chunk_num_{}".format(config.chunk_num)
            if batch_number % 250 == 0:
                print(batch_name)
            torch.save(
                batch_to_save[0],
                os.path.join("./embedding_dump_deterministic", chunk_name, batch_name),
            )

    else:
        with torch.no_grad():
            if config.canonical is True:
                output = [normalize_smiles(molecule, canonical=True, isomeric=False)]
            else:
                output = molecule

            batch_ids = tokenizer.batch_encode_plus(
                [output],
                padding=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_length=True,
            )

            raw_tokens = get_tokens_from_ids(batch_ids["input_ids"], tokenizer)[0]

            if config.mask is True:
                att_mask = FullMask(
                    torch.tensor(batch_ids["attention_mask"], dtype=bool).to(device),
                    device=device,
                )
            else:
                att_mask = FullMask(
                    torch.ones(
                        torch.tensor(batch_ids["input_ids"]).size(), dtype=bool
                    ).to(device),
                    device=device,
                )

            embeddings, attention_mask = bert_model(
                torch.tensor(batch_ids["input_ids"]).to(device),
                att_mask,
                mode=config.mode,
            )
            return attention_mask, raw_tokens


def get_tokens_from_ids(input_ids, tokenizer):
    tokens = []

    for idx_lst in input_ids:
        seq = []
        for idx in idx_lst:
            seq.append(tokenizer.ids_to_tokens[idx])
        tokens.append(seq)
    return tokens
