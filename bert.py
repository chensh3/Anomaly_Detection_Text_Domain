import torch
# import transformers
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states=True,  # Whether the model returns all hidden-states.
                                  )

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()


def bert_embed(data_raw):
    encoded_inputs = tokenizer(data_raw)

    storage = []  # list to store all embeddings
    output = []  # list to store all embeddings
    for i, text in tqdm(enumerate(encoded_inputs['input_ids'])):
        tokens_tensor = torch.tensor([encoded_inputs['input_ids'][i]])
        segments_tensors = torch.tensor([encoded_inputs['attention_mask'][i]])
        # print(tokens_tensor)
        # print(segments_tensors)

        # Run the text through BERT, and collect all of the hidden states produced
        # from all 12 layers.
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)

            # Evaluating the model will return a different number of objects based on
            # how it's  configured in the `from_pretrained` call earlier. In this case,
            # becase we set `output_hidden_states = True`, the third item will be the
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]

        # `hidden_states` has shape [13 x 1 x 22 x 768]

        # `token_vecs` is a tensor with shape [22 x 768]
        token_vecs = hidden_states[-2][0]

        # Calculate the average of all 22 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        # print(sentence_embedding[:10])
        storage.append((text, sentence_embedding))
        output.append(sentence_embedding)
    return output