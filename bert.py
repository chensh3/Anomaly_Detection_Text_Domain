import torch
# import transformers
from transformers import BertTokenizer, BertModel,BertConfig
from tqdm.auto import tqdm
import pandas as pd


df = pd.read_json("News_Category_Dataset_v3.json", lines = True)
df = df[["short_description", "category"]]
df.category = df.category.str.lower()

df.loc[(df.category == "the worldpost") | (df.category == "worldpost"), 'category'] = "world news"
df.loc[(df.category == "style & beauty"), 'category'] = "style"
df.loc[(df.category == "arts") | (df.category == "culture & arts"), 'category'] = "arts & culture"
df.loc[(df.category == "parents"), 'category'] = "parenting"
df.loc[(df.category == "taste"), 'category'] = "food & drink"
df.loc[(df.category == "green"), 'category'] = "environment"
df.loc[(df.category == "healthy living"), 'category'] = "healthy living tips"



print("start bert.py")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("tokonizer ready")
conf=BertConfig.from_pretrained('bert-base-uncased',hidden_size=192,output_hidden_states=True)
model = BertModel(conf)
print("model started")
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

print("model evaluated")
def bert_embed(data_raw):
    print("start function bert")
    encoded_inputs = tokenizer(data_raw)
    print("end words tokinizer")
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


print(" start embed")
embeding = bert_embed(df.short_description.to_list())
print("stop embed")
df["embed_tensor"] = embeding
df["embed"]=df.embed_tensor.apply(lambda x: x.numpy())
df.drop(columns=["embed_tensor"],inplace=True)
a=pd.DataFrame(df.embed.values.tolist())
a["short_description"]=df.short_description
a["category"]=df.category
# df["embed"]=df.embed_tensor.apply(lambda x: x.numpy())
a.to_pickle("short_df_all_192.pickle")
